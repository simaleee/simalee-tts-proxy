"""
Simalee TTS proxy — Edge TTS (Microsoft) Russian neural voice for ESP32 robots.
ESP32 hits a plain HTTP endpoint with text; we ask Microsoft Edge TTS for MP3
and stream it straight to the device. No API key on Microsoft's side, no
billing — Edge's public voice endpoint is free.

A shared password is required so a leaked URL doesn't burn quota.
"""

import os
import time as _time
import html
import xml.etree.ElementTree as ET
from urllib.parse import quote
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, PlainTextResponse, Response
import edge_tts
import httpx
import numpy as np

PASSWORD = os.environ.get("TTS_PASSWORD", "Simalee00221922")
DEFAULT_VOICE = "ru-RU-SvetlanaNeural"   # friendly female; or ru-RU-DmitryNeural for male

app = FastAPI(title="Simalee TTS Proxy")


@app.get("/", response_class=PlainTextResponse)
def root():
    return (
        "Simalee TTS proxy — alive.\n"
        "GET /tts?key=PASSWORD&text=привет&voice=ru-RU-SvetlanaNeural -> audio/mpeg\n"
        "GET /voices -> list of Russian voices\n"
    )


@app.get("/voices", response_class=PlainTextResponse)
def voices():
    return (
        "ru-RU-SvetlanaNeural   (female, default — warm, natural)\n"
        "ru-RU-DmitryNeural     (male)\n"
        "ru-RU-DariyaNeural     (female, multilingual)\n"
    )


@app.get("/tts")
async def tts(
    text: str = Query(..., min_length=1, max_length=900),
    key: str = Query(...),
    voice: str = Query(DEFAULT_VOICE),
    rate: str = Query("+0%"),
    pitch: str = Query("+0Hz"),
):
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")

    if not voice.startswith("ru-RU-"):
        voice = DEFAULT_VOICE   # force a Russian voice to keep robot consistent

    # Generate the WHOLE MP3 first, then send it in one response. Render's free CPU
    # makes edge-tts slower than real-time; streaming → the robot's player drains its
    # buffer mid-sentence and cuts off. Sending the complete file lets the robot
    # download it at network speed and play it from its buffer without underruns.
    comm = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    audio = bytearray()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return Response(content=bytes(audio), media_type="audio/mpeg")


# ---- internet info for the robot (the ESP can't reach some hosts directly; we fetch server-side) ----
WMO = {0: "ясно", 1: "облачно", 2: "облачно", 3: "пасмурно", 45: "туман", 48: "туман",
       51: "морось", 53: "морось", 55: "морось", 56: "морось", 57: "морось",
       61: "дождь", 63: "дождь", 65: "сильный дождь", 66: "дождь", 67: "дождь",
       71: "снег", 73: "снег", 75: "сильный снег", 77: "снег", 85: "снег", 86: "снег",
       80: "ливень", 81: "ливень", 82: "сильный ливень", 95: "гроза", 96: "гроза", 99: "гроза"}


@app.post("/gemini")
async def gemini_relay(request: Request, key: str = Query(...), model: str = Query("gemini-2.0-flash")):
    # Relay the robot's Gemini request to Google. Render isn't geo-blocked from Google, so this
    # works from Russia without a VPN. The API key travels in the x-goog-api-key header (not stored here).
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    gk = request.headers.get("x-goog-api-key", "")
    body = await request.body()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(url, headers={"x-goog-api-key": gk, "Content-Type": "application/json"}, content=body)
        return Response(content=r.content, media_type="application/json", status_code=r.status_code)
    except Exception:
        return Response(content=b'{"error":"relay failed"}', media_type="application/json", status_code=502)


@app.get("/time", response_class=PlainTextResponse)
def srv_time(key: str = Query(...)):
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    return str(int(_time.time()))   # UTC epoch; the robot has TZ=MSK so getLocalTime() shows Moscow time


@app.get("/weather", response_class=PlainTextResponse)
async def weather(key: str = Query(...), lat: float = Query(45.33), lon: float = Query(42.86)):
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           "&current=temperature_2m&daily=precipitation_probability_max,precipitation_sum,weathercode"
           "&forecast_days=1&timezone=auto")
    try:
        async with httpx.AsyncClient(timeout=12) as c:
            d = (await c.get(url)).json()
    except Exception:
        return "ERR погода недоступна"
    cur, daily = d.get("current", {}), d.get("daily", {})
    t = cur.get("temperature_2m")
    pprob = (daily.get("precipitation_probability_max") or [0])[0] or 0
    wc = int((daily.get("weathercode") or [0])[0] or 0)
    sky = WMO.get(wc, "ясно")
    rain = pprob >= 55 or wc >= 51
    body = (f"сейчас {round(t)} градусов, {sky}, вероятность осадков {pprob} процентов"
            if t is not None else f"{sky}, осадки {pprob} процентов")
    return ("RAIN " if rain else "OK ") + body


async def _heads(url: str, n: int):
    async with httpx.AsyncClient(timeout=12, follow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0"}) as c:
        root = ET.fromstring((await c.get(url)).text)
    out = []
    for it in root.findall(".//item")[:n]:
        title = html.unescape(it.findtext("title") or "").strip()
        if " - " in title:
            title = title.rsplit(" - ", 1)[0].strip()
        if title:
            out.append(title)
    return out


GEN_NEWS = "https://news.google.com/rss/headlines/section/geo/Russia?hl=ru&gl=RU&ceid=RU:ru"


@app.get("/search", response_class=PlainTextResponse)
async def search(key: str = Query(...), q: str = Query(..., max_length=160)):
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True,
                                     headers={"User-Agent": "Mozilla/5.0"}) as c:
            d = (await c.get(f"https://api.duckduckgo.com/?q={quote(q)}&format=json&no_html=1&skip_disambig=1&kl=ru-ru")).json()
        for k in ("Answer", "AbstractText", "Definition"):
            v = d.get(k)
            if v:
                return str(v)[:400]
        rt = d.get("RelatedTopics") or []
        if rt and isinstance(rt[0], dict) and rt[0].get("Text"):
            return str(rt[0]["Text"])[:400]
    except Exception:
        pass
    try:                                   # fallback: fresh news headlines for the query
        heads = await _heads(f"https://news.google.com/rss/search?q={quote(q)}&hl=ru&gl=RU&ceid=RU:ru", 2)
        if heads:
            return ". ".join(heads) + "."
    except Exception:
        pass
    return "Точного ответа не нашла в интернете."


# ---- speaker recognition (voiceprint) -------------------------------------------------
# The ESP can't run a speaker model, so it ships the raw mic PCM here. We compute a small
# MFCC "voiceprint" (pure numpy, no torch/scipy — fits Render's free 512MB). The robot keeps
# the enrolled voiceprints on its SD card and sends them back on /identify, so the proxy stays
# stateless (Render's free disk is wiped on every redeploy). See [[robot-voice-stack]].

def _voiceprint(pcm_bytes, sr=16000, n_mfcc=13, n_filt=26, nfft=512):
    """Raw 16-bit mono PCM -> 12-dim MFCC mean voiceprint (skips c0/energy). None if too short."""
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if x.size < sr // 3:                      # < ~0.33s of audio -> useless
        return None
    x = x / 32768.0
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])    # pre-emphasis
    flen, fstep = int(0.025 * sr), int(0.010 * sr)
    if x.size < flen:
        return None
    nfr = 1 + (x.size - flen) // fstep
    idx = np.arange(flen)[None, :] + fstep * np.arange(nfr)[:, None]
    frames = x[idx] * np.hamming(flen)
    pspec = (np.abs(np.fft.rfft(frames, nfft)) ** 2) / nfft     # power spectrum, nfr x (nfft/2+1)
    # mel filterbank
    hi = 2595 * np.log10(1 + (sr / 2) / 700)
    mel = np.linspace(0, hi, n_filt + 2)
    hz = 700 * (10 ** (mel / 2595) - 1)
    bins = np.floor((nfft + 1) * hz / sr).astype(int)
    fbank = np.zeros((n_filt, nfft // 2 + 1))
    for m in range(1, n_filt + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fbank[m - 1, k] = (k - l) / (c - l + 1e-9)
        for k in range(c, r):
            fbank[m - 1, k] = (r - k) / (r - c + 1e-9)
    feat = np.log(np.maximum(np.dot(pspec, fbank.T), 1e-10))    # nfr x n_filt
    # DCT-II -> MFCC
    dct = np.cos(np.pi * np.arange(n_mfcc)[None, :] * (2 * np.arange(n_filt)[:, None] + 1) / (2 * n_filt))
    mfcc = np.dot(feat, dct)                  # nfr x n_mfcc
    # voice-activity: keep only louder frames (drop silence/breaths) for a stable print
    energy = pspec.sum(axis=1)
    voiced = energy > energy.mean() * 0.35
    if voiced.sum() >= 5:
        mfcc = mfcc[voiced]
    return mfcc[:, 1:].mean(axis=0)           # 12-dim timbre print (skip c0 = volume)


def _parse_profiles(prof):
    """'name:v1,v2,...;name2:...' -> [(name, np.array)]"""
    out = []
    for entry in prof.split(";"):
        if ":" not in entry:
            continue
        name, vec = entry.split(":", 1)
        name = name.strip()
        try:
            v = np.array([float(t) for t in vec.split(",") if t != ""])
        except ValueError:
            continue
        if name and v.size:
            out.append((name, v))
    return out


@app.post("/enroll", response_class=PlainTextResponse)
async def enroll(request: Request, key: str = Query(...)):
    """POST raw 16k mono PCM -> comma-joined voiceprint the robot stores on SD."""
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    vp = _voiceprint(await request.body())
    if vp is None:
        return "ERR short"
    return ",".join(f"{v:.3f}" for v in vp)


@app.post("/identify", response_class=PlainTextResponse)
async def identify(request: Request, key: str = Query(...),
                   prof: str = Query("", max_length=4000),
                   thr: float = Query(11.0)):
    """POST raw PCM + ?prof=enrolled profiles -> 'name|distance'. Distance>thr => 'unknown'."""
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    vp = _voiceprint(await request.body())
    if vp is None:
        return "unknown|0"
    best_name, best_d = "unknown", 1e9
    for name, pv in _parse_profiles(prof):
        if pv.shape != vp.shape:
            continue
        d = float(np.linalg.norm(vp - pv))    # euclidean over 12 MFCC means
        if d < best_d:
            best_d, best_name = d, name
    if best_d > thr:
        return f"unknown|{best_d:.2f}"
    return f"{best_name}|{best_d:.2f}"


@app.get("/news", response_class=PlainTextResponse)
async def news(key: str = Query(...), q: str = Query("", max_length=140), n: int = Query(2)):
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    n = max(1, min(n, 4))
    heads = []
    try:
        if q.strip():
            heads = await _heads(f"https://news.google.com/rss/search?q={quote(q)}&hl=ru&gl=RU&ceid=RU:ru", n)
        if not heads:                       # long sentence search often empty -> fall back to top headlines
            heads = await _heads(GEN_NEWS, n)
    except Exception:
        return "ERR новости недоступны"
    return (". ".join(heads) + ".") if heads else "Новостей не нашлось."
