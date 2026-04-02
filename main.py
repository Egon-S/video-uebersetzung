"""
Video-Uebersetzungs Microservice v1.3.0
=========================================
Wandelt englischsprachige Videos ins Deutsche um - MIT Zeitsynchronisation.

Neu in v1.3.0:
  - Whisper gibt Segmente MIT Zeitstempeln zurueck
  - Jedes Segment wird einzeln uebersetzt (GPT-4o, in Batches)
  - Jedes Segment bekommt eigene TTS-Audiodatei
  - FFmpeg atempo-Filter passt TTS-Laenge an Original-Timing an
  - FFmpeg adelay+amix baut die finale Tonspur zeitsynchron zusammen

Pipeline:
  1. Video herunterladen (yt-dlp) ODER direkt als MP4 hochladen
  2. Audio extrahieren (FFmpeg)
  3. Transkription (OpenAI Whisper) mit Segment-Zeitstempeln
  4. Batch-Uebersetzung aller Segmente (GPT-4o)
  5. TTS pro Segment (OpenAI TTS) + atempo-Anpassung auf Original-Dauer
  6. Tonspur zusammenbauen mit adelay+amix (korrekte Zeitpositionen)
  7. Video + neue Tonspur zusammenfuehren (FFmpeg)
"""

import os
import uuid
import asyncio
import json
from pathlib import Path
from typing import Optional

import yt_dlp
import openai
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -- Konfiguration --------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WORK_DIR       = Path(os.getenv("WORK_DIR", "/tmp/video_translate"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
COOKIES_FILE   = WORK_DIR / "cookies.txt"

# -- FastAPI App ----------------------------------------------------------------
app = FastAPI(
    title="Video-Uebersetzungs API",
    description="Uebersetzt englische Videos automatisch ins Deutsche mit Zeitsynchronisation.",
    version="1.3.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict = {}


# -- Datenmodelle ---------------------------------------------------------------
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


# -- Hilfsfunktionen ------------------------------------------------------------
def _init_job(status="pending", progress=0, label="Job wird vorbereitet..."):
    return {
        "status": status,
        "progress": progress,
        "step_label": label,
        "download_url": None,
        "error": None,
    }


def _upd(job_id, status, progress, label):
    jobs[job_id].update({"status": status, "progress": progress, "step_label": label})


def _ydl_sync(url, opts):
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


# -- API Endpunkte --------------------------------------------------------------
@app.post("/translate", response_model=JobStatus, summary="Video per URL uebersetzen")
async def start_translation(req: TranslateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = _init_job()
    oai_key = req.openai_api_key or OPENAI_API_KEY
    background_tasks.add_task(process_video_from_url, job_id, req.video_url, oai_key)
    return JobStatus(job_id=job_id, **jobs[job_id])


@app.post("/upload-video", response_model=JobStatus, summary="MP4 hochladen und uebersetzen")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    openai_api_key: Optional[str] = Form(None),
):
    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    jobs[job_id] = _init_job("uploading", 5, "Datei wird hochgeladen...")
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    video_path = job_dir / "video.mp4"
    video_path.write_bytes(content)
    jobs[job_id].update({
        "status": "pending",
        "progress": 10,
        "step_label": f"Video empfangen ({size_mb:.1f} MB), Verarbeitung startet...",
    })
    oai_key = openai_api_key or OPENAI_API_KEY
    background_tasks.add_task(process_video_pipeline, job_id, video_path, oai_key)
    return JobStatus(job_id=job_id, **jobs[job_id])


@app.get("/status/{job_id}", response_model=JobStatus, summary="Job-Status abfragen")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")
    return JobStatus(job_id=job_id, **jobs[job_id])


@app.get("/jobs/{job_id}", response_model=JobStatus, summary="Job-Status abfragen (Alias)")
async def get_job(job_id: str):
    return await get_status(job_id)


@app.get("/download/{job_id}", summary="Fertiges Video herunterladen")
async def download_video(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")
    if jobs[job_id]["status"] != "done":
        raise HTTPException(status_code=400, detail="Video noch nicht fertig")
    output_path = WORK_DIR / job_id / "output.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Datei nicht gefunden")
    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=f"uebersetzt_{job_id[:8]}.mp4",
    )


@app.get("/health", summary="Health Check")
async def health():
    return {"status": "ok", "version": "1.3.0", "cookies": COOKIES_FILE.exists()}


# -- Pipeline: URL-Download -----------------------------------------------------
async def process_video_from_url(job_id: str, video_url: str, openai_key: str):
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        _upd(job_id, "downloading", 5, "Video wird heruntergeladen...")
        opts = {
            "outtmpl": str(job_dir / "video.%(ext)s"),
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "quiet": True,
            "socket_timeout": 30,
            "retries": 3,
        }
        if COOKIES_FILE.exists():
            opts["cookiefile"] = str(COOKIES_FILE)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: _ydl_sync(video_url, opts))
        video_path = next(job_dir.glob("video.*"), None)
        if not video_path:
            raise FileNotFoundError("Video konnte nicht heruntergeladen werden.")
        await process_video_pipeline(job_id, video_path, openai_key, base=15)
    except Exception as e:
        jobs[job_id].update({
            "status": "error", "progress": 0,
            "step_label": "Fehler aufgetreten", "error": str(e),
        })


# -- Pipeline: Haupt-Verarbeitungspipeline -------------------------------------
async def process_video_pipeline(job_id: str, video_path: Path, openai_key: str, base: int = 10):
    job_dir = video_path.parent
    try:
        # Schritt 1: Audio extrahieren
        _upd(job_id, "extracting", base + 5, "Audio wird extrahiert...")
        audio_path = job_dir / "audio.mp3"
        await run_ffmpeg([
            "-i", str(video_path),
            "-vn", "-acodec", "mp3", "-ar", "16000", "-ac", "1",
            str(audio_path), "-y",
        ])

        # Schritt 2: Transkription MIT Zeitstempeln
        _upd(job_id, "transcribing", base + 15, "Transkription mit Zeitstempeln laeuft...")
        segments = await transcribe_with_segments(audio_path, openai_key)
        n = len(segments)
        _upd(job_id, "transcribing", base + 25, f"Transkription fertig: {n} Segmente erkannt.")

        # Schritt 3: Alle Segmente in Batches uebersetzen
        _upd(job_id, "translating", base + 30, f"Uebersetzung von {n} Segmenten (Batches)...")
        translated = await batch_translate_segments(segments, openai_key)

        # Schritt 4: TTS pro Segment + atempo-Anpassung
        seg_audio_list = []
        for i, (seg, de_text) in enumerate(zip(segments, translated)):
            pct = base + 45 + int(30 * i / max(n, 1))
            _upd(job_id, "synthesizing", pct, f"Stimme: Segment {i + 1}/{n}...")
            seg_path = job_dir / f"seg_{i:04d}.mp3"
            target_dur = seg["end"] - seg["start"]
            await tts_and_fit(de_text, seg_path, openai_key, target_dur)
            seg_audio_list.append((seg["start"], seg_path))

        # Schritt 5: Synchronisierte Tonspur zusammenbauen
        _upd(job_id, "merging", base + 80, "Synchronisierte Tonspur wird zusammengebaut...")
        synced_audio = job_dir / "synced.mp3"
        await build_synced_audio(seg_audio_list, synced_audio, video_path)

        # Schritt 6: Video + neue Tonspur zusammenfuehren
        _upd(job_id, "merging", base + 93, "Video wird zusammengefuehrt...")
        output_path = job_dir / "output.mp4"
        await run_ffmpeg([
            "-i", str(video_path),
            "-i", str(synced_audio),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path), "-y",
        ])

        jobs[job_id].update({
            "status": "done",
            "progress": 100,
            "step_label": "Uebersetzung abgeschlossen!",
            "download_url": f"/download/{job_id}",
        })

    except Exception as e:
        jobs[job_id].update({
            "status": "error", "progress": 0,
            "step_label": "Fehler aufgetreten", "error": str(e),
        })


# -- Kernfunktionen -------------------------------------------------------------
async def transcribe_with_segments(audio_path: Path, api_key: str) -> list:
    """Transkribiert Audio und gibt Liste von {start, end, text}-Segmenten zurueck."""
    client = openai.AsyncOpenAI(api_key=api_key)
    with open(audio_path, "rb") as f:
        resp = await client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en",
            response_format="verbose_json",
        )
    segs = getattr(resp, "segments", None) or []
    return [
        {
            "start": float(s.get("start") if isinstance(s, dict) else s.start),
            "end":   float(s.get("end")   if isinstance(s, dict) else s.end),
            "text":  (s.get("text") if isinstance(s, dict) else s.text).strip(),
        }
        for s in segs
        if (s.get("text") if isinstance(s, dict) else s.text).strip()
    ]


async def batch_translate_segments(segments: list, api_key: str) -> list:
    """Uebersetzt alle Segmente in Batches von 30 pro GPT-4o-Aufruf."""
    if not segments:
        return []
    client = openai.AsyncOpenAI(api_key=api_key)
    batch_size = 30
    all_translated = []

    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start: batch_start + batch_size]
        items = [{"i": j, "t": s["text"]} for j, s in enumerate(batch)]
        prompt = json.dumps(items, ensure_ascii=True)

        resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional dubbing translator (English to German). "
                        "You receive a JSON array of objects with 'i' (index) and 't' (text). "
                        "Translate each 't' value from English to German. "
                        "Keep translations concise to match the original speech timing. "
                        "Return a JSON array in the EXACT same format with translated 't' values. "
                        "Return ONLY valid JSON, no markdown, no explanations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        # Markdown-Code-Block entfernen falls vorhanden
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                # Manchmal gibt GPT-4o {"segments": [...]} zurueck
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
            lookup = {item["i"]: item["t"] for item in parsed}
            all_translated.extend(lookup.get(j, batch[j]["text"]) for j in range(len(batch)))
        except Exception:
            # Fallback: Original-Text behalten
            all_translated.extend(s["text"] for s in batch)

    return all_translated


async def tts_and_fit(text: str, output_path: Path, api_key: str, target_dur: float):
    """Erstellt TTS-Audio und passt es per atempo an target_dur (Sekunden) an."""
    if not text.strip() or target_dur <= 0.05:
        # Stille erzeugen
        await run_ffmpeg([
            "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
            "-t", str(max(target_dur, 0.1)),
            str(output_path), "-y",
        ])
        return

    client = openai.AsyncOpenAI(api_key=api_key)
    resp = await client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text,
    )
    raw_path = output_path.parent / (output_path.stem + "_raw.mp3")
    raw_path.write_bytes(resp.content)

    tts_dur = await get_audio_duration(raw_path)
    if tts_dur <= 0:
        raw_path.rename(output_path)
        return

    # Geschwindigkeitsfaktor: tts_dur / target_dur
    # > 1.0 = TTS ist laenger als Original  -> schneller abspielen
    # < 1.0 = TTS ist kuerzer als Original  -> langsamer abspielen
    speed = tts_dur / target_dur
    # Bereich beschraenken: min 0.75x, max 2.0x (Qualitaets-Kompromiss)
    speed = max(0.75, min(speed, 2.0))

    if abs(speed - 1.0) < 0.04:
        # Kaum Unterschied, keine Anpassung noetig
        raw_path.rename(output_path)
    else:
        atempo_filter = _build_atempo(speed)
        await run_ffmpeg([
            "-i", str(raw_path),
            "-filter:a", atempo_filter,
            str(output_path), "-y",
        ])
        try:
            raw_path.unlink()
        except OSError:
            pass


def _build_atempo(speed: float) -> str:
    """Baut FFmpeg-atempo-Filterkette fuer beliebige Geschwindigkeit (0.5-2.0 pro Stufe)."""
    parts = []
    s = speed
    while s > 2.0:
        parts.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        parts.append("atempo=0.5")
        s /= 0.5
    parts.append(f"atempo={s:.4f}")
    return ",".join(parts)


async def get_audio_duration(path: Path) -> float:
    """Gibt die Laenge einer Audio-/Videodatei in Sekunden zurueck (via ffprobe)."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    try:
        return float(json.loads(stdout.decode())["format"]["duration"])
    except Exception:
        return 0.0


async def build_synced_audio(seg_list: list, output_path: Path, video_path: Path):
    """
    Kombiniert alle Segment-Audiodateien zu einer vollstaendigen Tonspur.
    Jedes Segment wird per adelay an seine Original-Zeitposition gesetzt,
    dann werden alle Streams per amix zusammengemischt.
    """
    video_dur = await get_audio_duration(video_path)

    if not seg_list:
        await run_ffmpeg([
            "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
            "-t", str(max(video_dur, 1.0)),
            str(output_path), "-y",
        ])
        return

    # Wenn sehr viele Segmente: in Gruppen von 50 zusammenbauen
    # (verhindert zu lange FFmpeg-Kommandos)
    if len(seg_list) > 50:
        output_path = await _build_synced_chunked(seg_list, output_path, video_path, video_dur)
        return

    inputs = []
    filters = []
    labels = []
    for i, (start_sec, seg_path) in enumerate(seg_list):
        inputs += ["-i", str(seg_path)]
        delay_ms = int(start_sec * 1000)
        filters.append(f"[{i}]adelay={delay_ms}|{delay_ms}[d{i}]")
        labels.append(f"[d{i}]")

    n = len(seg_list)
    fc = (
        ";".join(filters)
        + ";"
        + "".join(labels)
        + f"amix=inputs={n}:duration=longest:normalize=0[out]"
    )
    await run_ffmpeg(
        inputs + [
            "-filter_complex", fc,
            "-map", "[out]",
            "-t", str(video_dur),
            str(output_path), "-y",
        ]
    )


async def _build_synced_chunked(seg_list: list, output_path: Path, video_path: Path, video_dur: float):
    """
    Hilfsfunktion fuer sehr viele Segmente: verarbeitet in 50er-Gruppen,
    mischt Zwischenergebnisse zusammen.
    """
    chunk_size = 50
    tmp_files = []
    for chunk_idx in range(0, len(seg_list), chunk_size):
        chunk = seg_list[chunk_idx: chunk_idx + chunk_size]
        tmp_path = output_path.parent / f"tmp_chunk_{chunk_idx}.mp3"
        inputs, filters, labels = [], [], []
        for i, (start_sec, seg_path) in enumerate(chunk):
            inputs += ["-i", str(seg_path)]
            delay_ms = int(start_sec * 1000)
            filters.append(f"[{i}]adelay={delay_ms}|{delay_ms}[d{i}]")
            labels.append(f"[d{i}]")
        n = len(chunk)
        fc = (
            ";".join(filters)
            + ";"
            + "".join(labels)
            + f"amix=inputs={n}:duration=longest:normalize=0[out]"
        )
        await run_ffmpeg(inputs + ["-filter_complex", fc, "-map", "[out]", "-t", str(video_dur), str(tmp_path), "-y"])
        tmp_files.append(tmp_path)

    if len(tmp_files) == 1:
        tmp_files[0].rename(output_path)
    else:
        # Alle Chunks zusammenmischen
        inputs = []
        fc_inputs = []
        for i, f in enumerate(tmp_files):
            inputs += ["-i", str(f)]
            fc_inputs.append(f"[{i}]")
        n = len(tmp_files)
        fc = "".join(fc_inputs) + f"amix=inputs={n}:duration=longest:normalize=0[out]"
        await run_ffmpeg(inputs + ["-filter_complex", fc, "-map", "[out]", "-t", str(video_dur), str(output_path), "-y"])
        for f in tmp_files:
            try:
                f.unlink()
            except OSError:
                pass


async def run_ffmpeg(args: list):
    """Fuehrt FFmpeg asynchron aus."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {stderr.decode()[-600:]}")
