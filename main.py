"""
Video-Übersetzungs Microservice
================================
Wandelt englischsprachige Videos in deutschsprachige Videos um.

Pipeline:
  1. Video herunterladen (yt-dlp)
  2. Audio extrahieren (FFmpeg)
  3. Transkription Englisch (OpenAI Whisper)
  4. Übersetzung Englisch → Deutsch (OpenAI GPT-4o)
  5. Text-to-Speech Deutsch (OpenAI TTS)
  6. Video + neue Tonspur zusammenführen (FFmpeg)

Starten:
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional

import yt_dlp
import openai
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

# ──────────────────────────────────────────────
# Konfiguration (øber Umgebungsvariablen)
# ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WORK_DIR       = Path(os.getenv("WORK_DIR", "/tmp/video_translate"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

openai.api_key = OPENAI_API_KEY

# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(
    title="Video-Übersetzungs API",
    description="Übersetzt englische Videos automatisch ins Deutsche.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In Produktion: nur Base44-Domain eintragen
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# In-Memory Job-Speicher (für Produktion: Redis/DB)
# ──────────────────────────────────────────────
jobs: dict[str, dict] = {}


# ──────────────────────────────────────────────
# Datenmodelle
# ──────────────────────────────────────────────
class TranslateRequest(BaseModel):
    video_url: str
    openai_api_key: Optional[str] = None  # Optional: Override aus Base44

class JobStatus(BaseModel):
    job_id: str
    status: str          # pending | downloading | extracting | transcribing | translating | synthesizing | merging | done | error
    progress: int        # 0–100
    step_label: str
    download_url: Optional[str] = None
    error: Optional[str] = None


# ──────────────────────────────────────────────
# API Endpunkte
# ──────────────────────────────────────────────
@app.post("/translate", response_model=JobStatus, summary="Übersetzungsjob starten")
async def start_translation(req: TranslateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "step_label": "Job wird vorbereitet...",
        "download_url": None,
        "error": None,
    }
    # API Key: Request-Parameter überschreibt Umgebungsvariable
    oai_key = req.openai_api_key or OPENAI_API_KEY

    background_tasks.add_task(process_video, job_id, req.video_url, oai_key)
    return JobStatus(job_id=job_id, **jobs[job_id])


@app.get("/status/{job_id}", response_model=JobStatus, summary="Job-Status abfragen")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")
    return JobStatus(job_id=job_id, **jobs[job_id])


@app.get("/jobs/{job_id}", response_model=JobStatus, summary="Job-Status abfragen (Alias)")
async def get_job(job_id: str):
    """Alias für /status/{job_id} – kompatibel mit Base44."""
    return await get_status(job_id)


@app.get("/download/{job_id}", summary="Fertiges Video herunterladen")
async def download_video(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")
    job = jobs[job_id]
    if job["status"] != "done":
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
    return {"status": "ok"}


# ──────────────────────────────────────────────
# Kernprozess: Video verarbeiten
# ──────────────────────────────────────────────
async def process_video(job_id: str, video_url: str, openai_key: str):
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    def update(status: str, progress: int, label: str):
        jobs[job_id].update({"status": status, "progress": progress, "step_label": label})

    try:
        # ── Schritt 1: Video herunterladen ──
        update("downloading", 5, "Video wird heruntergeladen...")
        video_path = await download_video_file(video_url, job_dir)

        # ── Schritt 2: Audio extrahieren ──
        update("extracting", 20, "Audio wird extrahiert...")
        audio_path = job_dir / "audio.mp3"
        await run_ffmpeg([
            "-i", str(video_path),
            "-vn", "-acodec", "mp3", "-ar", "16000", "-ac", "1",
            str(audio_path), "-y"
        ])

        # ── Schritt 3: Transkription (Whisper) ──
        update("transcribing", 35, "Englisches Audio wird transkribiert...")
        english_text = await transcribe_audio(audio_path, openai_key)

        # ── Schritt 4: Übersetzung (GPT-4o) ──
        update("translating", 55, "Text wird ins Deutsche übersetzt...")
        german_text = await translate_to_german(english_text, openai_key)

        # ── Schritt 5: Text-to-Speech (OpenAI TTS) ──
        update("synthesizing", 70, "Deutsche Stimme wird synthetisiert...")
        german_audio_path = job_dir / "german_audio.mp3"
        await synthesize_speech(german_text, german_audio_path, openai_key)

        # ── Schritt 6: Video + neue Tonspur zusammenführen ──
        update("merging", 85, "Video wird zusammengeführt...")
        output_path = job_dir / "output.mp4"
        await run_ffmpeg([
            "-i", str(video_path),
            "-i", str(german_audio_path),
            "-c:v", "copy",          # Video-Stream unverändert übernehmen
            "-c:a", "aac",           # Audio neu kodieren
            "-map", "0:v:0",         # Video aus Original
            "-map", "1:a:0",         # Audio aus TTS
            "-shortest",             # Am kürzeren Stream enden
            str(output_path), "-y"
        ])

        # ── Fertig ──
        jobs[job_id].update({
            "status": "done",
            "progress": 100,
            "step_label": "Übersetzung abgeschlossen! ✓",
            "download_url": f"/download/{job_id}",
        })

    except Exception as e:
        jobs[job_id].update({
            "status": "error",
            "progress": 0,
            "step_label": "Fehler aufgetreten",
            "error": str(e),
        })


# ──────────────────────────────────────────────
# Hilfsfunktionen
# ──────────────────────────────────────────────
async def download_video_file(url: str, job_dir: Path) -> Path:
    """Lädt das Video mit yt-dlp herunter."""
    output_template = str(job_dir / "video.%(ext)s")
    ydl_opts = {
        "outtmpl": output_template,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        # YouTube-Bot-Detection umgehen (Cloud-Server-IPs werden oft blockiert)
        "extractor_args": {
            "youtube": {
                "player_client": ["ios", "web"],
            }
        },
        "socket_timeout": 30,
        "retries": 3,
    }
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: _download_sync(url, ydl_opts))

    # Datei finden (Endung kann variieren)
    for f in job_dir.glob("video.*"):
        return f
    raise FileNotFoundError("Video konnte nicht heruntergeladen werden.")


def _download_sync(url: str, opts: dict):
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])


async def run_ffmpeg(args: list[str]):
    """Führt FFmpeg asynchron aus."""
    cmd = ["ffmpeg"] + args
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg Fehler: {stderr.decode()}")


async def transcribe_audio(audio_path: Path, api_key: str) -> str:
    """Transkribiert Audio mit OpenAI Whisper."""
    client = openai.AsyncOpenAI(api_key=api_key)
    with open(audio_path, "rb") as f:
        response = await client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en",
        )
    return response.text


async def translate_to_german(text: str, api_key: str) -> str:
    """Übersetzt Text mit OpenAI GPT-4o von Englisch nach Deutsch."""
    client = openai.AsyncOpenAI(api_key=api_key)

    # Lange Texte aufteilen (GPT-4o max. ~120k Tokens, aber kleinere Chunks = bessere Qualität)
    chunks = split_text(text, max_length=3000)
    translated_parts = []

    for chunk in chunks:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du bist ein professioneller Übersetzer. "
                        "Übersetze den folgenden englischen Text ins Deutsche. "
                        "Behalte den Stil, Ton und die Struktur des Originals bei. "
                        "Gib nur den øbersetzten Text zurøck, ohne Erklärungen."
                    ),
                },
                {"role": "user", "content": chunk},
            ],
            temperature=0.3,
        )
        translated_parts.append(response.choices[0].message.content)

    return " ".join(translated_parts)


async def synthesize_speech(text: str, output_path: Path, api_key: str):
    """Erstellt deutsche Sprachausgabe mit OpenAI TTS."""
    client = openai.AsyncOpenAI(api_key=api_key)

    # Lange Texte in Abschnitte aufteilen (OpenAI TTS max. ~4096 Zeichen)
    chunks = split_text(text, max_length=4000)
    audio_parts = []

    for i, chunk in enumerate(chunks):
        response = await client.audio.speech.create(
            model="tts-1",
            voice="onyx",   # Stimmen: alloy, echo, fable, onyx, nova, shimmer
            input=chunk,
        )
        chunk_path = output_path.parent / f"tts_chunk_{i}.mp3"
        chunk_path.write_bytes(response.content)
        audio_parts.append(chunk_path)

    if len(audio_parts) == 1:
        audio_parts[0].rename(output_path)
    else:
        # Mehrere Teile zusammenfügen
        list_file = output_path.parent / "chunks.txt"
        list_file.write_text("\n".join(f"file '{p}'" for p in audio_parts))
        await run_ffmpeg([
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy", str(output_path), "-y"
        ])


def split_text(text: str, max_length: int = 4000) -> list[str]:
    """Teilt langen Text in Sätze auf."""
    if len(text) <= max_length:
        return [text]
    sentences = text.replace("! ", "!\n").replace("? ", "?\n").replace(". ", ".\n").split("\n")
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < max_length:
            current += sentence + " "
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + " "
    if current:
        chunks.append(current.strip())
    return chunks
