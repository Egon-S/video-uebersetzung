"""
Video-Ãbersetzungs Microservice
================================
Wandelt englischsprachige Videos in deutschsprachige Videos um.

Pipeline:
  1. Video herunterladen (yt-dlp)
  2. Audio extrahieren (FFmpeg)
  3. Transkription Englisch (OpenAI Whisper)
  4. Ãbersetzung Englisch â Deutsch (OpenAI GPT-4o)
  5. Text-to-Speech Deutsch (OpenAI TTS)
  6. Video + neue Tonspur zusammenfÃ¼hren (FFmpeg)

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

# ââââââââââââââââââââââââââââââââââââââââââââââ
# Konfiguration (Ã¸ber Umgebungsvariablen)
# ââââââââââââââââââââââââââââââââââââââââââââââ
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WORK_DIR       = Path(os.getenv("WORK_DIR", "/tmp/video_translate"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

openai.api_key = OPENAI_API_KEY

# ââââââââââââââââââââââââââââââââââââââââââââââ
# FastAPI App
# ââââââââââââââââââââââââââââââââââââââââââââââ
app = FastAPI(
    title="Video-Ãbersetzungs API",
    description="Ãbersetzt englische Videos automatisch ins Deutsche.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In Produktion: nur Base44-Domain eintragen
    allow_methods=["*"],
    allow_headers=["*"],
)

# ââââââââââââââââââââââââââââââââââââââââââââââ
# In-Memory Job-Speicher (fÃ¼r Produktion: Redis/DB)
# ââââââââââââââââââââââââââââââââââââââââââââââ
jobs: dict[str, dict] = {}


# ââââââââââââââââââââââââââââââââââââââââââââââ
# Datenmodelle
# ââââââââââââââââââââââââââââââââââââââââââââââ
class TranslateRequest(BaseModel):
    video_url: str
    openai_api_key: Optional[str] = None  # Optional: Override aus Base44

class JobStatus(BaseModel):
    job_id: str
    status: str          # pending | downloading | extracting | transcribing | translating | synthesizing | merging | done | error
    progress: int        # 0â100
    step_label: str
    download_url: Optional[str] = None
    error: Optional[str] = None


# ââââââââââââââââââââââââââââââââââââââââââââââ
# API Endpunkte
# ââââââââââââââââââââââââââââââââââââââââââââââ
@app.post("/translate", response_model=JobStatus, summary="Ãbersetzungsjob starten")
async def start_translation(req: TranslateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "step_label": "Job wird vorbereitet...",
        "download_url": None,
        "error": None,
    }
    # API Key: Request-Parameter Ã¼berschreibt Umgebungsvariable
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
    """Alias fÃ¼r /status/{job_id} â kompatibel mit Base44."""
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


# ââââââââââââââââââââââââââââââââââââââââââââââ
# Kernprozess: Video verarbeiten
# ââââââââââââââââââââââââââââââââââââââââââââââ
async def process_video(job_id: str, video_url: str, openai_key: str):
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    def update(status: str, progress: int, label: str):
        jobs[job_id].update({"status": status, "progress": progress, "step_label": label})

    try:
        # ââ Schritt 1: Video herunterladen ââ
        update("downloading", 5, "Video wird heruntergeladen...")
        video_path = await download_video_file(video_url, job_dir)

        # ââ Schritt 2: Audio extrahieren ââ
        update("extracting", 20, "Audio wird extrahiert...")
        audio_path = job_dir / "audio.mp3"
        await run_ffmpeg([
            "-i", str(video_path),
            "-vn", "-acodec", "mp3", "-ar", "16000", "-ac", "1",
            str(audio_path), "-y"
        ])

        # ââ Schritt 3: Transkription (Whisper) ââ
        update("transcribing", 35, "Englisches Audio wird transkribiert...")
        english_text = await transcribe_audio(audio_path, openai_key)

        # ââ Schritt 4: Ãbersetzung (GPT-4o) ââ
        update("translating", 55, "Text wird ins Deutsche Ã¼bersetzt...")
        german_text = await translate_to_german(english_text, openai_key)

        # ââ Schritt 5: Text-to-Speech (OpenAI TTS) ââ
        update("synthesizing", 70, "Deutsche Stimme wird synthetisiert...")
        german_audio_path = job_dir / "german_audio.mp3"
        await synthesize_speech(german_text, german_audio_path, openai_key)

        # ââ Schritt 6: Video + neue Tonspur zusammenfÃ¼hren ââ
        update("merging", 85, "Video wird zusammengefÃ¼hrt...")
        output_path = job_dir / "output.mp4"
        await run_ffmpeg([
            "-i", str(video_path),
            "-i", str(german_audio_path),
            "-c:v", "copy",          # Video-Stream unverÃ¤ndert Ã¼bernehmen
            "-c:a", "aac",           # Audio neu kodieren
            "-map", "0:v:0",         # Video aus Original
            "-map", "1:a:0",         # Audio aus TTS
            "-shortest",             # Am kÃ¼rzeren Stream enden
            str(output_path), "-y"
        ])

        # ââ Fertig ââ
        jobs[job_id].update({
            "status": "done",
            "progress": 100,
            "step_label": "Ãbersetzung abgeschlossen! â",
            "download_url": f"/download/{job_id}",
        })

    except Exception as e:
        jobs[job_id].update({
            "status": "error",
            "progress": 0,
            "step_label": "Fehler aufgetreten",
            "error": str(e),
        })


# ââââââââââââââââââââââââââââââââââââââââââââââ
# Hilfsfunktionen
# ââââââââââââââââââââââââââââââââââââââââââââââ
async def download_video_file(url: str, job_dir: Path) -> Path:
    """LÃ¤dt das Video mit yt-dlp herunter."""
    output_template = str(job_dir / "video.%(ext)s")
    ydl_opts = {
        "outtmpl": output_template,
        "format": "best[height<=720]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        # YouTube-Bot-Detection umgehen (Cloud-Server-IPs werden oft blockiert)
        "extractor_args": {
            "youtube": {
                "player_client": ["android_creator", "android", "web"],
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
    """FÃ¼hrt FFmpeg asynchron aus."""
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
    """Ãbersetzt Text mit OpenAI GPT-4o von Englisch nach Deutsch."""
    client = openai.AsyncOpenAI(api_key=api_key)

    # Lange Texte aufteilen (GPT-4o max. ~120k Tokens, aber kleinere Chunks = bessere QualitÃ¤t)
    chunks = split_text(text, max_length=3000)
    translated_parts = []

    for chunk in chunks:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du bist ein professioneller Ãbersetzer. "
                        "Ãbersetze den folgenden englischen Text ins Deutsche. "
                        "Behalte den Stil, Ton und die Struktur des Originals bei. "
                        "Gib nur den Ã¸bersetzten Text zurÃ¸ck, ohne ErklÃ¤rungen."
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
        # Mehrere Teile zusammenfÃ¼gen
        list_file = output_path.parent / "chunks.txt"
        list_file.write_text("\n".join(f"file '{p}'" for p in audio_parts))
        await run_ffmpeg([
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy", str(output_path), "-y"
        ])


def split_text(text: str, max_length: int = 4000) -> list[str]:
    """Teilt langen Text in SÃ¤tze auf."""
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
