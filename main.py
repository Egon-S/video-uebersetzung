"""
Video-Uebersetzungs Microservice v1.4.0
========================================
Wandelt englischsprachige Videos (MP4-Upload oder YouTube-URL) in
deutschsprachige Videos um.

Pipeline:
  1. Video empfangen (MP4-Upload oder yt-dlp)
  2. Audio extrahieren (FFmpeg)
  3. Transkription mit Zeitstempeln (OpenAI Whisper verbose_json)
  4. Batch-Uebersetzung Englisch -> Deutsch (GPT-4o, 30 Segmente pro Call)
  5. TTS pro Segment (OpenAI TTS) + atempo-Anpassung (FFmpeg)
  6. Synchronisierte Tonspur (FFmpeg adelay+amix)
  7. Video + neue Tonspur zusammenfuehren (FFmpeg)

Starten:
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import uuid
import asyncio
import json
import math
import tempfile
from pathlib import Path
from typing import Optional, List

import yt_dlp
import openai
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---
# Konfiguration (Umgebungsvariablen)
# ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WORK_DIR = Path(os.getenv("WORK_DIR", "/tmp/video_translate"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ---
# FastAPI App
# ---
app = FastAPI(
    title="Video-Uebersetzungs API",
    description="Uebersetzt englische Videos automatisch ins Deutsche.",
    version="1.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ---
# In-Memory Job-Speicher
# ---
jobs: dict = {}


# ---
# Datenmodelle
# ---
class TranslateRequest(BaseModel):
    video_url: str
    openai_api_key: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    step_label: str
    download_url: Optional[str] = None
    error: Optional[str] = None


# ---
# API-Endpunkte
# ---
@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.4.0"}


@app.post("/translate", response_model=JobStatus)
async def start_translation(req: TranslateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "step_label": "Job wird vorbereitet...",
        "download_url": None,
        "error": None,
    }
    oai_key = req.openai_api_key or OPENAI_API_KEY
    background_tasks.add_task(process_from_url, job_id, req.video_url, oai_key)
    return JobStatus(job_id=job_id, **jobs[job_id])


@app.post("/upload-video", response_model=JobStatus)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    openai_api_key: Optional[str] = Form(None),
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "step_label": "Upload wird verarbeitet...",
        "download_url": None,
        "error": None,
    }

    # Datei sofort speichern (nicht im Background, damit kein Stream-Problem)
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / "input.mp4"

    try:
        content = await file.read()
        video_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload-Fehler: {e}")

    oai_key = openai_api_key or OPENAI_API_KEY
    background_tasks.add_task(process_video, job_id, video_path, oai_key)
    return JobStatus(job_id=job_id, **jobs[job_id])


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")
    return JobStatus(job_id=job_id, **jobs[job_id])


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Video nicht bereit. Status: {job['status']}")
    output_path = WORK_DIR / job_id / "output.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Datei nicht gefunden")
    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=f"uebersetzt_{job_id[:8]}.mp4",
    )


# ---
# Verarbeitungs-Pipeline
# ---
def update_job(job_id: str, status: str, progress: int, label: str):
    jobs[job_id].update({
        "status": status,
        "progress": progress,
        "step_label": label,
    })


async def process_from_url(job_id: str, video_url: str, openai_key: str):
    """Startet die Pipeline fuer YouTube-URL-Videos."""
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        update_job(job_id, "downloading", 5, "Video wird heruntergeladen...")
        video_path = await download_video_file(video_url, job_dir)
        await process_video(job_id, video_path, openai_key)
    except Exception as e:
        jobs[job_id].update({
            "status": "error",
            "progress": 0,
            "step_label": "Fehler beim Download",
            "error": str(e),
        })


async def process_video(job_id: str, video_path: Path, openai_key: str):
    """Kern-Pipeline: Audio -> Transkription -> Uebersetzung -> TTS -> Merge."""
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Schritt 1: Audio extrahieren
        update_job(job_id, "extracting", 15, "Audio wird extrahiert...")
        audio_path = job_dir / "audio.wav"
        await run_ffmpeg([
            "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path), "-y"
        ])

        # Schritt 2: Transkription mit Zeitstempeln (Whisper verbose_json)
        update_job(job_id, "transcribing", 30, "Englisches Audio wird transkribiert...")
        segments = await transcribe_audio_with_timestamps(audio_path, openai_key)

        if not segments:
            raise RuntimeError("Keine Sprache im Audio erkannt.")

        # Schritt 3: Batch-Uebersetzung (30 Segmente pro API-Call)
        update_job(job_id, "translating", 50, "Text wird ins Deutsche uebersetzt...")
        translated_segments = await batch_translate_segments(segments, openai_key)

        # Schritt 4: TTS pro Segment + atempo-Anpassung
        update_job(job_id, "synthesizing", 65, "Deutsche Stimme wird synthetisiert...")
        tts_files = await synthesize_segments(translated_segments, job_dir, openai_key)

        # Schritt 5: Synchronisierte Tonspur zusammenbauen
        update_job(job_id, "mixing", 80, "Tonspur wird synchronisiert...")
        mixed_audio = job_dir / "mixed_audio.aac"
        await build_synchronized_audio(tts_files, translated_segments, mixed_audio, job_dir)

        # Schritt 6: Video + neue Tonspur zusammenfuehren
        update_job(job_id, "merging", 92, "Video wird zusammengefuehrt...")
        output_path = job_dir / "output.mp4"
        await run_ffmpeg([
            "-i", str(video_path),
            "-i", str(mixed_audio),
            "-c:v", "copy",
            "-c:a", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path), "-y"
        ])

        jobs[job_id].update({
            "status": "done",
            "progress": 100,
            "step_label": "Uebersetzung abgeschlossen!",
            "download_url": f"/download/{job_id}",
        })

    except Exception as e:
        jobs[job_id].update({
            "status": "error",
            "progress": 0,
            "step_label": "Fehler aufgetreten",
            "error": str(e),
        })


# ---
# Hilfsfunktionen
# ---
async def run_ffmpeg(args: list):
    cmd = ["ffmpeg"] + args
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg-Fehler: {stderr.decode('utf-8', errors='replace')[-500:]}")


async def download_video_file(url: str, job_dir: Path) -> Path:
    output_template = str(job_dir / "video.%(ext)s")
    ydl_opts = {
        "outtmpl": output_template,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "socket_timeout": 30,
        "retries": 3,
    }
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: _download_sync(url, ydl_opts))
    for f in job_dir.glob("video.*"):
        return f
    raise FileNotFoundError("Video konnte nicht heruntergeladen werden.")


def _download_sync(url: str, opts: dict):
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


async def transcribe_audio_with_timestamps(audio_path: Path, api_key: str) -> List[dict]:
    """Transkribiert Audio mit Zeitstempeln via Whisper verbose_json."""
    client = openai.AsyncOpenAI(api_key=api_key)
    with open(audio_path, "rb") as f:
        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en",
            response_format="verbose_json",
        )
    segments = []
    raw_segments = getattr(response, "segments", None) or []
    for seg in raw_segments:
        start = getattr(seg, "start", 0)
        end = getattr(seg, "end", 0)
        text = getattr(seg, "text", "").strip()
        if text:
            segments.append({"start": start, "end": end, "text": text})
    # Fallback: kein Zeitstempel -> ein einziges Segment
    if not segments:
        full_text = getattr(response, "text", "")
        if full_text.strip():
            segments.append({"start": 0.0, "end": 9999.0, "text": full_text.strip()})
    return segments


async def batch_translate_segments(segments: List[dict], api_key: str) -> List[dict]:
    """Uebersetzt Segmente in Batches von 30 (GPT-4o, JSON-Array)."""
    client = openai.AsyncOpenAI(api_key=api_key)
    batch_size = 30
    result = []

    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        texts = [s["text"] for s in batch]

        prompt = (
            "Uebertrage folgende englische Saetze ins Deutsche. "
            "Antworte NUR mit einem JSON-Objekt mit dem Schlussel \"translations\", "
            "der ein Array der uebersetzten Strings enthaelt. "
            "Behalte dieselbe Reihenfolge. Beispiel: {\"translations\": [\"Satz 1\", \"Satz 2\"]}\n\n"
            f"Eingabe: {json.dumps(texts, ensure_ascii=False)}"
        )

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "translations" in parsed:
                translated_texts = parsed["translations"]
            elif isinstance(parsed, dict):
                translated_texts = next(iter(parsed.values()))
            elif isinstance(parsed, list):
                translated_texts = parsed
            else:
                translated_texts = texts
        except json.JSONDecodeError:
            translated_texts = texts

        for j, seg in enumerate(batch):
            translated_text = translated_texts[j] if j < len(translated_texts) else seg["text"]
            result.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "translated": translated_text,
            })

    return result


async def synthesize_segments(
    segments: List[dict],
    job_dir: Path,
    api_key: str,
) -> List[Path]:
    """TTS fuer jedes Segment + atempo-Anpassung an Original-Timing."""
    client = openai.AsyncOpenAI(api_key=api_key)
    tts_files = []

    for i, seg in enumerate(segments):
        text = seg["translated"]
        if not text.strip():
            # Leeres Segment: Stille erzeugen
            duration = max(seg["end"] - seg["start"], 0.1)
            silence_path = job_dir / f"seg_{i:04d}_tts.mp3"
            await run_ffmpeg([
                "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono",
                "-t", str(duration),
                "-acodec", "mp3",
                str(silence_path), "-y"
            ])
            tts_files.append(silence_path)
            continue

        # TTS generieren
        raw_tts_path = job_dir / f"seg_{i:04d}_raw.mp3"
        response = await client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=text,
        )
        raw_tts_path.write_bytes(response.content)

        # Ziel-Dauer berechnen
        target_duration = seg["end"] - seg["start"]
        if target_duration <= 0:
            target_duration = 1.0

        # Ist-Dauer der TTS ermitteln
        actual_duration = await get_audio_duration(raw_tts_path)

        # atempo-Filter berechnen (FFmpeg: atempo zwischen 0.5 und 2.0)
        final_tts_path = job_dir / f"seg_{i:04d}_tts.mp3"
        if actual_duration > 0:
            speed_ratio = actual_duration / target_duration
            # atempo-Kette bei extremen Werten
            atempo_filter = build_atempo_filter(speed_ratio)
            await run_ffmpeg([
                "-i", str(raw_tts_path),
                "-af", atempo_filter,
                str(final_tts_path), "-y"
            ])
        else:
            raw_tts_path.rename(final_tts_path)

        tts_files.append(final_tts_path)

    return tts_files


async def get_audio_duration(audio_path: Path) -> float:
    """Gibt die Dauer einer Audiodatei in Sekunden zurueck."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(audio_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    try:
        data = json.loads(stdout.decode())
        return float(data["format"]["duration"])
    except Exception:
        return 0.0


def build_atempo_filter(speed_ratio: float) -> str:
    """Baut einen FFmpeg atempo-Filterstring (Bereich 0.5-2.0 pro Filter)."""
    speed_ratio = max(0.25, min(speed_ratio, 4.0))  # Clamp
    filters = []
    remaining = speed_ratio
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


async def build_synchronized_audio(
    tts_files: List[Path],
    segments: List[dict],
    output_path: Path,
    job_dir: Path,
):
    """Baut die synchronisierte Tonspur: jedes Segment an seiner Zeitposition."""
    if not tts_files:
        raise RuntimeError("Keine TTS-Dateien vorhanden.")

    # FFmpeg adelay+amix: Jedes Segment wird mit Verzoegerung positioniert
    inputs = []
    filter_parts = []
    mix_inputs = ""

    for i, (tts_file, seg) in enumerate(zip(tts_files, segments)):
        delay_ms = int(seg["start"] * 1000)
        inputs += ["-i", str(tts_file)]
        filter_parts.append(f"[{i}]adelay={delay_ms}|{delay_ms}[s{i}]")
        mix_inputs += f"[s{i}]"

    n = len(tts_files)
    filter_complex = (
        ";".join(filter_parts)
        + f";{mix_inputs}amix=inputs={n}:normalize=0[aout]"
    )

    await run_ffmpeg([
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-acodec", "aac",
        "-ar", "44100",
        str(output_path), "-y"
    ])
