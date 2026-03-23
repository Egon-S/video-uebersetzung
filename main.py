"""
Video-Übersetzungs Microservice
===================================
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
from pydantic import BaseModel

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WORK_DIR = Path(os.getenv("WORK_DIR", "/tmp/video_translate"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
openai.api_key = OPENAI_API_KEY

app = FastAPI(title="Video-Übersetzungs API", description="Übersetzt englische Videos ins Deutsche.", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
jobs: dict[str, dict] = {}

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

@app.post("/translate", response_model=JobStatus)
async def start_translation(req: TranslateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0, "step_label": "Job wird vorbereitet...", "download_url": None, "error": None}
    oai_key = req.openai_api_key or OPENAI_API_KEY
    background_tasks.add_task(process_video, job_id, req.video_url, oai_key)
    return JobStatus(job_id=job_id, **jobs[job_id])

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    if job_id not in jobs: raise HTTPException(status_code=404, detail="Job nicht gefunden")
    return JobStatus(job_id=job_id, **jobs[job_id])

@app.get("/download/{job_id}")
async def download_video(job_id: str):
    if job_id not in jobs: raise HTTPException(status_code=404, detail="Job nicht gefunden")
    job = jobs[job_id]
    if job["status"] != "done": raise HTTPException(status_code=400, detail="Video noch nicht fertig")
    output_path = WORK_DIR / job_id / "output.mp4"
    if not output_path.exists(): raise HTTPException(status_code=404, detail="Datei nicht gefunden")
    return FileResponse(path=str(output_path), media_type="video/mp4", filename=f"uebersetzt_{job_id[:8]}.mp4")

@app.get("/health")
async def health(): return {"status": "ok"}

async def process_video(job_id, video_url, openai_key):
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    def update(s, p, l): jobs[job_id].update({"status":s," progress":p,"step_label":l})
    try:
        update("downloading",5,"Video wird heruntergeladen...")
        video_path = await download_video_file(video_url, job_dir)
        update("extracting",20,"Audio wird extrahiert...")
        audio_path = job_dir / "audio.mp3"
        await run_ffmpeg(["-i",str(video_path),"-vn","-acodec","mp3","-ar","16000","-ac","1",str(audio_path),"-y"])
        update("transcribing",35,"Englisches Audio wird transkribiert...")
        english_text = await transcribe_audio(audio_path, openai_key)
        update("translating",55,"Text wird ins Deutsche übersetzt...")
        german_text = await translate_to_german(english_text, openai_key)
        update("synthesizing",70,"Deutsche Stimme wird synthetisiert...")
        german_audio_path = job_dir / "german_audio.mp3"
        await synthesize_speech(german_text, german_audio_path, openai_key)
        update("merging",85,"Video wird zusammengeführt...")
        output_path = job_dir / "output.mp4"
        await run_ffmpeg(["-i",str(video_path),"-i",str(german_audio_path),"-c:v","copy","-c:a","aac","-map","0:v:0","-map","1:a:0","-shortest",str(output_path),"-y"])
        jobs[job_id].update({"status":"done","progress":100,"step_label":"Übersetzung abgeschlossen!","download_url":f"/download/{job_id}"})
    except Exception as e:
        jobs[job_id].update({"status":"error","progress":0,"step_label":"Fehler aufgetreten","error":str(e)})

async def download_video_file(url, job_dir):
    opts = {"outtmpl":str(job_dir)/"video.%(ext)s","format":"bestvideo[ext=mp4]+bestaudio[ext=m4a]/best","merge_output_format":"mp4","quiet":True}
    loop=asyncio.get_event_loop()
    await loop.run_in_executor(None,lambda:_download_sync(url,opts))
    for f in job_dir.glob("video.*"): return f
    raise FileNotFoundError()

def _download_sync(url,opts):
    with yt_dlp.YoutubeDL(opts) as y: y.download([url])

async def run_ffmpeg(args):
    p = await asyncio.create_subprocess_exec(*["ffmpeg"]+args,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    _,e=await p.communicate()
    if p.returncode!=0: raise RuntimeError(f.decode())

async def transcribe_audio(path, key):
    c=openai.AsyncOpenAI(api_key=key)
    with open(path,"rb") as f: r=await c.audio.transcriptions.create(model="whisper-1",file=f,language="en")
    return r.text

async def translate_to_german(text, key):
    c=openai.AsyncOpenAI(api_key=key)
    parts=[]
    for chunk in split_text(text,3000):
        r=await c.chat.completions.create(model="gpt-4o",messages=[{"role":"system","content":"Professioneller Übersetzer EN>DE. Nur Übersetzung ohne Erklärungen."},{"role":"user","content":chunk}],temperature=0.3)
        parts.append(r.choices[0].message.content)
    return " ".join(parts)

async def synthesize_speech(text, out, key):
    c=openai.AsyncOpenAI(api_key=key)
    parts=[]
    for i,chunk in enumerate(split_text(text,4000)):
        r=await c.audio.speech.create(model="tts-1",voice="onyx",input=chunk)
        p=out.parent/f"tts_{i}.mp3"; p.write_bytes(r.content); parts.append(p)
    if len(parts)==1: parts[0].rename(out)
    else:
        lf=out.parent/"chunks.txt"; lf.write_text("\n".join([f"file '{p}'" for p in parts]))
        await run_ffmpeg(["-f","concat","-safe","0","-i",str(lf),"-c","copy",str(out),"-y"])

def split_text(t,m=4000):
    if len(t)<=m: return [t]
    ss=t.replace("! ","!\n").replace("? ","?\n").replace(". ",".\n").split("\n")
    chs,cur=[],""
    for s in ss:
        if len(cur)+len(s)<m: cur+=s+" "
        else:
            if cur: chs.append(cur.strip())
            cur=s+" "
    if cur: chs.append(cur.strip())
    return chs
