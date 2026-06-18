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
from fastapi.responses import StreamingResponse, PlainTextResponse, Response, JSONResponse, HTMLResponse
import io
import json as _json
import edge_tts
import httpx
import numpy as np
from PIL import Image, ImageDraw
from pywebpush import webpush, WebPushException

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


@app.post("/groq")
async def groq_relay(request: Request, key: str = Query(...)):
    # Relay the robot's chat request to Groq (OpenAI-compatible). Groq = fast (LPU) + a generous free tier.
    # The API key travels in the Authorization header (not stored here). Render(US) -> Groq has no geo issue.
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    auth = request.headers.get("authorization", "")
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post("https://api.groq.com/openai/v1/chat/completions",
                             headers={"Authorization": auth, "Content-Type": "application/json"}, content=body)
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


# ---- remote link: a tiny mailbox so the user can reach the robot from ANYWHERE (no Telegram, no port-forward) ----
# Browser (any network) -> /remote page -> /say_remote (queue a phrase) + /status_remote (read live status).
# Robot (polls every ~5s) -> /robot_poll (fetch a queued phrase) + /robot_status (push its state).
# In-memory only: Render free is a single instance; the queue resets on redeploy/sleep — fine for a mailbox.
REMOTE_CMD = []          # queue for the robot: [{"mode":"say"|"chat","text":...}]
REMOTE_OUT = []          # the chat thread: [{"id":n,"from":"you"|"robot","text":...,"t":epoch}]
REMOTE_OUT_ID = 0        # monotonic message id (page tracks the last it has seen)
REMOTE_STATUS = {}       # last status the robot pushed (temp, hum, battery, ...)
REMOTE_TS = 0.0          # epoch of the robot's last check-in (poll or status)


def _out_add(frm, text):
    global REMOTE_OUT_ID
    text = (text or "").strip()
    if not text:
        return
    REMOTE_OUT_ID += 1
    REMOTE_OUT.append({"id": REMOTE_OUT_ID, "from": frm, "text": text[:600], "t": int(_time.time())})
    del REMOTE_OUT[:-40]                       # keep the last 40 messages


@app.get("/say_remote", response_class=PlainTextResponse)
def say_remote(key: str = Query(...), text: str = Query(..., max_length=300)):
    # speak this aloud AT HOME (the original feature)
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    t = text.strip()
    if t:
        REMOTE_CMD.append({"mode": "say", "text": t})
        del REMOTE_CMD[:-10]
    return "ok"


@app.get("/chat_remote", response_class=PlainTextResponse)
def chat_remote(key: str = Query(...), text: str = Query(..., max_length=600)):
    # TEXT chat: shows in the thread + the robot replies in text (its full AI brain)
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    t = text.strip()
    if t:
        _out_add("you", t)
        REMOTE_CMD.append({"mode": "chat", "text": t})
        del REMOTE_CMD[:-10]
    return "ok"


@app.get("/robot_poll", response_class=PlainTextResponse)
def robot_poll(key: str = Query(...)):
    # robot fetches one queued command -> "mode|text" (split on the FIRST '|')
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    global REMOTE_TS
    REMOTE_TS = _time.time()                   # polling => robot is alive
    if REMOTE_CMD:
        c = REMOTE_CMD.pop(0)
        return c["mode"] + "|" + c["text"]
    return ""


@app.get("/robot_msg", response_class=PlainTextResponse)
def robot_msg(key: str = Query(...), text: str = Query(..., max_length=600)):
    # robot posts an outbound message (a chat reply OR a fired reminder) into the thread
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    global REMOTE_TS
    _out_add("robot", text)
    REMOTE_TS = _time.time()
    return "ok"


@app.get("/robot_status", response_class=PlainTextResponse)
def robot_status(request: Request, key: str = Query(...)):
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    global REMOTE_STATUS, REMOTE_TS
    st = dict(request.query_params)
    st.pop("key", None)
    REMOTE_STATUS = st
    REMOTE_TS = _time.time()
    return "ok"


@app.get("/status_remote")
def status_remote(key: str = Query(...)):
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    age = int(_time.time() - REMOTE_TS) if REMOTE_TS else -1
    return JSONResponse({"age": age, "pending": len(REMOTE_CMD), **REMOTE_STATUS})


@app.get("/outbox_remote")
def outbox_remote(key: str = Query(...), since: int = Query(0)):
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    msgs = [m for m in REMOTE_OUT if m["id"] > since]
    return JSONResponse({"msgs": msgs, "last": REMOTE_OUT_ID})


# ---- remote SETTINGS (the PWA changes settings; the robot polls + applies) ----
REMOTE_SET = {}          # pending setting changes from the app -> {param: value}


@app.get("/set_remote", response_class=PlainTextResponse)
def set_remote(request: Request, key: str = Query(...)):
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    for k, v in request.query_params.items():
        if k != "key":
            REMOTE_SET[k] = v
    return "ok"


@app.get("/settings_poll")
def settings_poll(key: str = Query(...)):
    # robot polls this; returns pending changes and clears them
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    global REMOTE_SET, REMOTE_TS
    REMOTE_TS = _time.time()
    out = dict(REMOTE_SET)
    REMOTE_SET = {}
    return JSONResponse(out)


# ---- Web Push notifications (PWA -> phone notification shade, even when app is closed) ----
VAPID_PUBLIC = "BFdQApV-VGZj4Oy_ZrBp2eo1Jt3XLJkQheGfRNXkMOI132x97YUw_Df98UeCkmb3auW98Mp1uKk1bXjiqROF1sE"
VAPID_PRIVATE = "gxYrwPWIH2TV4f83an5nyXTKqySWl7JJALlyCxqSmfg"
VAPID_SUB = "mailto:bobekmunyer619@gmail.com"
PUSH_SUBS = []           # web-push subscriptions from the PWA (in-memory; user re-subscribes by opening the app)


@app.get("/vapid_public", response_class=PlainTextResponse)
def vapid_public():
    return VAPID_PUBLIC


@app.post("/push_subscribe", response_class=PlainTextResponse)
async def push_subscribe(request: Request, key: str = Query(...)):
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    sub = await request.json()
    if sub.get("endpoint") and sub not in PUSH_SUBS:
        PUSH_SUBS.append(sub)
        del PUSH_SUBS[:-20]
    return "ok"


LAST_PUSH_ERR = ""


def _send_push(text, title="Simalee"):
    global LAST_PUSH_ERR
    payload = _json.dumps({"title": title, "body": (text or "")[:300]})
    sent, failed, dead = 0, 0, []
    for sub in list(PUSH_SUBS):
        try:
            webpush(subscription_info=sub, data=payload,
                    vapid_private_key=VAPID_PRIVATE, vapid_claims={"sub": VAPID_SUB})
            sent += 1
        except WebPushException as e:
            failed += 1
            LAST_PUSH_ERR = str(e)[:160]
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (404, 410):
                dead.append(sub)
        except Exception as e:
            failed += 1
            LAST_PUSH_ERR = str(e)[:160]
    for d in dead:
        if d in PUSH_SUBS:
            PUSH_SUBS.remove(d)
    return sent, failed


@app.get("/push")
def push(key: str = Query(...), text: str = Query(..., max_length=300), title: str = Query("Simalee")):
    # robot (or anything) calls this to push a notification to the phone(s)
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    sent, failed = _send_push(text.strip(), title)
    return JSONResponse({"subs": len(PUSH_SUBS), "sent": sent, "failed": failed, "err": LAST_PUSH_ERR})


@app.get("/push_count")
def push_count(key: str = Query(...)):
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    return JSONResponse({"subs": len(PUSH_SUBS), "err": LAST_PUSH_ERR})


# ---- installable PWA (status + settings), served at /app ----
def _make_icon(sz):
    img = Image.new("RGB", (sz, sz), (14, 18, 48))
    d = ImageDraw.Draw(img)
    r, ry, cy = sz * 0.17, sz * 0.12, sz * 0.46
    for cx in (sz * 0.35, sz * 0.65):                       # two glowing almond eyes
        d.ellipse([cx - r, cy - ry, cx + r, cy + ry], fill=(95, 200, 255))
        d.ellipse([cx - r * 0.45, cy - ry * 0.45, cx + r * 0.45, cy + ry * 0.45], fill=(14, 18, 48))
    d.arc([sz * 0.40, sz * 0.52, sz * 0.60, sz * 0.72], 20, 160, fill=(233, 196, 106), width=max(2, sz // 36))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


ICON512 = _make_icon(512)
ICON192 = _make_icon(192)


@app.get("/app/icon-512.png")
def icon512():
    return Response(content=ICON512, media_type="image/png")


@app.get("/app/icon-192.png")
def icon192():
    return Response(content=ICON192, media_type="image/png")


USER_ICON = None         # custom notification icon (uploaded from the app; in-memory)


@app.post("/set_icon", response_class=PlainTextResponse)
async def set_icon(request: Request, key: str = Query(...)):
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    global USER_ICON
    body = await request.body()
    try:
        im = Image.open(io.BytesIO(body)).convert("RGB").resize((192, 192))
        buf = io.BytesIO()
        im.save(buf, "PNG")
        USER_ICON = buf.getvalue()
    except Exception:
        raise HTTPException(status_code=400, detail="bad image")
    return "ok"


@app.get("/app/usericon.png")
def usericon():                                     # custom icon if uploaded, else the default
    return Response(content=USER_ICON if USER_ICON else ICON192, media_type="image/png")


@app.get("/app/manifest.webmanifest")
def manifest():
    return JSONResponse({
        "name": "Simalee", "short_name": "Simalee", "start_url": "/app", "scope": "/app",
        "display": "standalone", "background_color": "#0b0e26", "theme_color": "#141a44",
        "icons": [
            {"src": "/app/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/app/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ],
    }, media_type="application/manifest+json")


@app.get("/app/sw.js")
def service_worker():
    js = (
        "self.addEventListener('install',e=>self.skipWaiting());"
        "self.addEventListener('activate',e=>self.clients.claim());"
        "self.addEventListener('fetch',e=>{});"
        "self.addEventListener('push',e=>{let d={title:'Simalee',body:''};"
        "try{d=e.data.json()}catch(x){try{d.body=e.data.text()}catch(y){}}"
        "e.waitUntil(self.registration.showNotification(d.title||'Simalee',"
        "{body:d.body||'',icon:'/app/usericon.png',badge:'/app/usericon.png',vibrate:[120,60,120],tag:'simalee'}));});"
        "self.addEventListener('notificationclick',e=>{e.notification.close();"
        "e.waitUntil(clients.matchAll({type:'window'}).then(cs=>{for(const c of cs){if('focus'in c)return c.focus();}"
        "if(clients.openWindow)return clients.openWindow('/app');}));});"
    )
    return Response(content=js, media_type="application/javascript")


APP_HTML = """<!doctype html><html lang=ru><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name=theme-color content="#141a44"><title>Simalee</title>
<link rel=manifest href="/app/manifest.webmanifest">
<link rel=apple-touch-icon href="/app/icon-192.png">
<style>
:root{--ink:#eef;--card:#171c44;--line:#2a316a;--gold:#e9c46a;--mut:#8a93c8;--grn:#4cc78a;--red:#e15b4c}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:linear-gradient(160deg,#0b0e26,#141a44);color:var(--ink);min-height:100dvh;padding:14px env(safe-area-inset-right) calc(14px + env(safe-area-inset-bottom)) env(safe-area-inset-left)}
.wrap{max-width:480px;margin:0 auto}
h1{font-size:20px;margin:4px 2px 12px}h1 b{color:var(--gold)}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:14px;margin-bottom:12px}
.big{font-size:17px;font-weight:700;margin-bottom:8px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;vertical-align:middle}
.row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #ffffff10;font-size:15px}
.row:last-child{border:0}.row .k{color:var(--mut)}.row .v{font-weight:600}
.set{margin:12px 0}.set label{display:flex;justify-content:space-between;font-size:14px;color:var(--mut);margin-bottom:5px}
.set label b{color:var(--ink)}
input[type=range]{width:100%;accent-color:var(--gold)}
select{width:100%;padding:10px;border-radius:10px;background:#0c1030;color:var(--ink);border:1px solid var(--line);font-size:15px}
.tog{display:flex;align-items:center;justify-content:space-between;padding:8px 0;font-size:15px}
.sw{width:46px;height:26px;border-radius:13px;background:#33396a;position:relative;transition:.15s}
.sw.on{background:var(--grn)}.sw i{position:absolute;top:3px;left:3px;width:20px;height:20px;border-radius:50%;background:#fff;transition:.15s}.sw.on i{left:23px}
.kc{margin:18vh auto 0;max-width:330px;text-align:center}
.kc input{width:100%;display:block;padding:14px;margin:12px 0;border-radius:12px;border:1px solid var(--line);background:#0c1030;color:var(--ink);font-size:16px}
.kc button{width:100%;padding:14px;border:0;border-radius:12px;background:var(--gold);color:#1a1530;font-weight:700;font-size:16px}
</style></head><body><div class=wrap>
<div class=kc id=keycard style=display:none>
  <h1>сяма<b>лии</b></h1><div style="color:var(--mut)">Пароль доступа (один раз)</div>
  <input id=key type=password placeholder=пароль autocomplete=current-password onkeydown="if(event.key=='Enter')saveKey()">
  <button onclick=saveKey()>Войти</button>
</div>
<div id=app style=display:none>
  <h1>сяма<b>лии</b></h1>
  <div class=card>
    <div class=big id=online>…</div>
    <div class=row><span class=k>🌡 Температура</span><span class=v id=t>—</span></div>
    <div class=row><span class=k>💧 Влажность</span><span class=v id=h>—</span></div>
    <div class=row><span class=k>⚡ Энергия</span><span class=v id=en>—</span></div>
    <div class=row><span class=k>🔋 Батарея</span><span class=v id=bat>—</span></div>
    <div class=row><span class=k>💤 Состояние</span><span class=v id=slp>—</span></div>
    <div class=row><span class=k>⏱ Аптайм</span><span class=v id=up>—</span></div>
    <div class=row><span class=k>🌐 IP в доме</span><span class=v id=ip>—</span></div>
  </div>
  <div class=card>
    <div class=big>Настройки</div>
    <div class=set><label>🔊 Громкость <b id=volv>—</b></label><input type=range id=vol min=0 max=21 oninput="lv('volv',this.value)" onchange="setp('vol',this.value)"></div>
    <div class=set><label>🎙 Чувствительность мика <b id=micv>—</b></label><input type=range id=mic min=0 max=100 oninput="lv('micv',this.value,'%')" onchange="setp('mic',this.value)"></div>
    <div class=set><label>🔆 Яркость экрана <b id=briv>—</b></label><input type=range id=bri min=5 max=100 oninput="lv('briv',this.value,'%')" onchange="setp('bri',this.value)"></div>
    <div class=set><label>👁 Свечение глаз <b id=eglowv>—</b></label><input type=range id=eglow min=5 max=100 oninput="lv('eglowv',this.value,'%')" onchange="setp('eglow',this.value)"></div>
    <div class=set><label>🕐 Часы во сне <b id=clockbriv>—</b></label><input type=range id=clockbri min=5 max=100 oninput="lv('clockbriv',this.value,'%')" onchange="setp('clockbri',this.value)"></div>
    <div class=set><label>🗣 Голос</label><select id=voice onchange="setp('voice',this.value)">
      <option value=ru-RU-SvetlanaNeural>Светлана</option><option value=ru-RU-DmitryNeural>Дмитрий</option><option value=ru-RU-DariyaNeural>Дария</option></select></div>
    <div class=tog><span>🔁 Голос наоборот</span><div class=sw id=gender onclick="tg('gender','gender')"><i></i></div></div>
    <div class=tog><span>🐤 Звуки в покое</span><div class=sw id=chirp onclick="tg('chirp','chirp')"><i></i></div></div>
    <div class=tog><span>🩺 Авто-диагностика</span><div class=sw id=adiag onclick="tg('adiag','adiag')"><i></i></div></div>
    <div style="color:var(--mut);font-size:12px;margin-top:6px" id=setnote>Изменения долетают до робота за ~5 сек.</div>
  </div>
  <div class=card>
    <div class=big>🔔 Уведомления на телефон</div>
    <div style="color:var(--mut);font-size:13px;margin-bottom:8px" id=pushst>Включи, чтобы напоминания приходили в шторку (даже когда приложение закрыто).</div>
    <div style="font-family:monospace;font-size:11px;color:#9aa3d6;background:#0c1030;border-radius:8px;padding:6px 8px;margin-bottom:10px" id=pushdiag>диагностика…</div>
    <button style="width:100%;padding:13px;border:0;border-radius:12px;background:var(--gold);color:#1a1530;font-weight:700;font-size:16px" onclick=enablePush()>Включить уведомления</button>
    <button style="width:100%;padding:11px;margin-top:8px;border:0;border-radius:12px;background:#23306a;color:#fff;font-size:15px" onclick=testPush()>Проверить (тест)</button>
    <label style="display:block;margin-top:12px;color:var(--mut);font-size:13px">🖼 Своя иконка уведомлений:</label>
    <input type=file id=iconf accept="image/*" onchange=uploadIcon() style="width:100%;margin-top:6px;color:var(--mut);font-size:13px">
  </div>
</div>
<script>
let KEY=localStorage.getItem('simkey')||'';
function show(){keycard.style.display=KEY?'none':'block';app.style.display=KEY?'block':'none';}
function saveKey(){KEY=document.getElementById('key').value.trim();localStorage.setItem('simkey',KEY);show();tick();}
function fmtUp(s){s=+s||0;let h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?h+'ч '+m+'м':m+'м';}
function lv(id,v,suf){document.getElementById(id).textContent=v+(suf||'');}
function setp(k,v){fetch('/set_remote?key='+encodeURIComponent(KEY)+'&'+k+'='+encodeURIComponent(v));document.getElementById('setnote').textContent='Отправлено: '+k+' = '+v;}
let st={};
function tg(id,k){let on=!document.getElementById(id).classList.contains('on');document.getElementById(id).classList.toggle('on',on);setp(k,on?1:0);}
let touched=0;
async function tick(){if(!KEY)return;
 try{let r=await fetch('/status_remote?key='+encodeURIComponent(KEY));
  if(r.status==403){localStorage.removeItem('simkey');KEY='';show();return;}
  let d=await r.json();st=d;let on=d.age>=0&&d.age<14;
  online.innerHTML='<span class=dot style="background:'+(on?'#4cc78a':'#e15b4c')+'"></span>'+(on?'На связи':(d.age<0?'Ещё не выходил на связь':'Был '+d.age+'с назад'));
  t.textContent=(d.t&&d.t!='-99')?d.t+' °C':'—';h.textContent=(d.h&&d.h!='-99')?d.h+' %':'—';
  en.textContent=(d.en!=null&&d.en!=='')?d.en+' %':'—';
  bat.textContent=(d.bat&&d.bat!='-1')?d.bat+' %':'от сети';
  slp.textContent=(d.slp=='1')?'спит':'бодрствует';up.textContent=fmtUp(d.up);ip.textContent=d.ip||'—';
  if(Date.now()-touched>4000){ // don't fight the user mid-drag
   sv('vol','volv',d.vol,'');sv('mic','micv',d.mic,'%');sv('bri','briv',d.bri,'%');sv('eglow','eglowv',d.eglow,'%');sv('clockbri','clockbriv',d.clockbri,'%');
   if(d.voice&&document.activeElement!=voice)voice.value=d.voice;
   tgset('gender',d.gender);tgset('chirp',d.chirp);tgset('adiag',d.adiag);
  }
 }catch(e){}}
function sv(id,lab,v,suf){if(v==null||v==='')return;let el=document.getElementById(id);if(el!==document.activeElement){el.value=v;document.getElementById(lab).textContent=v+suf;}}
function tgset(id,v){document.getElementById(id).classList.toggle('on',v=='1'||v===1||v===true||v==='true');}
document.addEventListener('input',e=>{if(e.target.type=='range')touched=Date.now();},true);
if('serviceWorker'in navigator)navigator.serviceWorker.register('/app/sw.js');
function urlB64(s){let p='='.repeat((4-s.length%4)%4);let b=atob((s+p).replace(/-/g,'+').replace(/_/g,'/'));return Uint8Array.from([...b].map(c=>c.charCodeAt(0)));}
async function enablePush(){try{
 pushst.textContent='Проверяю поддержку…';
 if(!('serviceWorker'in navigator)){pushst.textContent='❌ Нет serviceWorker — открой как УСТАНОВЛЕННОЕ приложение в Chrome.';return;}
 if(!('PushManager'in window)){pushst.textContent='❌ Телефон без Web Push (нет Google-сервисов / не Chrome).';return;}
 pushst.textContent='Запрашиваю разрешение…';
 let perm=await Notification.requestPermission();
 if(perm!=='granted'){pushst.textContent='❌ Разрешение: '+perm+'. Включи уведомления приложению в настройках телефона.';return;}
 pushst.textContent='Жду сервис-воркер…';
 let reg=await Promise.race([navigator.serviceWorker.ready,new Promise((_,rej)=>setTimeout(()=>rej(new Error('сервис-воркер не активировался')),8000))]);
 pushst.textContent='Подписываюсь в FCM…';
 let pub=(await (await fetch('/vapid_public')).text()).trim();
 let sub=await reg.pushManager.getSubscription();
 if(!sub)sub=await reg.pushManager.subscribe({userVisibleOnly:true,applicationServerKey:urlB64(pub)});
 pushst.textContent='Сохраняю подписку…';
 let r=await fetch('/push_subscribe?key='+encodeURIComponent(KEY),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(sub)});
 let c=await(await fetch('/push_count?key='+encodeURIComponent(KEY))).json();
 pushst.textContent=(r.ok&&c.subs>0)?('✅ Готово! Подписок на сервере: '+c.subs+'. Жми «Проверить».'):('⚠ Не сохранилось (ответ '+r.status+', subs='+c.subs+').');
}catch(e){pushst.textContent='❌ Сломалось на шаге: '+(e.message||e)+' — часто это VPN, который режет Google/FCM. Попробуй БЕЗ VPN.';}}
async function testPush(){try{let r=await(await fetch('/push?key='+encodeURIComponent(KEY)+'&title=Simalee&text='+encodeURIComponent('Тестовое уведомление 🔔'))).json();
 if(r.subs==0)pushst.textContent='⚠ Нет подписки. Нажми «Включить уведомления» и разреши их.';
 else if(r.sent>0)pushst.textContent='✅ Отправлено на '+r.sent+' устр. — должно прийти в шторку.';
 else pushst.textContent='Подписка есть ('+r.subs+'), но доставка не прошла'+(r.err?(': '+r.err):'')+'. Скорее всего мешает VPN/блокировка Google — попробуй без VPN.';
}catch(e){pushst.textContent='Ошибка теста: '+e;}}
async function uploadIcon(){let f=document.getElementById('iconf').files[0];if(!f)return;try{let b=await f.arrayBuffer();let r=await fetch('/set_icon?key='+encodeURIComponent(KEY),{method:'POST',body:b});pushst.textContent=r.ok?'✅ Иконка установлена — появится в следующем уведомлении.':'Не удалось загрузить иконку.';}catch(e){pushst.textContent='Ошибка иконки: '+e;}}
async function pushStatus(){try{let r=await(await fetch('/push_count?key='+encodeURIComponent(KEY))).json();if(r.subs>0)pushst.textContent='✅ Подписано устройств: '+r.subs+'. Напоминания придут в шторку.';}catch(e){}}
function pushDiag(){try{
 let sw=('serviceWorker'in navigator), pm=('PushManager'in window);
 let perm=(window.Notification?Notification.permission:'нет Notification');
 let inst=(matchMedia('(display-mode: standalone)').matches||navigator.standalone)?'да':'НЕТ(в браузере)';
 let el=document.getElementById('pushdiag');
 if(el)el.textContent='SW:'+(sw?'да':'НЕТ')+' · Push:'+(pm?'да':'НЕТ')+' · Разрешение:'+perm+' · Установлено:'+inst;
}catch(e){}}
window.addEventListener('error',ev=>{let el=document.getElementById('pushdiag');if(el)el.textContent='JS-ошибка: '+(ev.message||ev);});
show();tick();if(KEY){pushStatus();pushDiag();}setInterval(tick,3000);
</script></div></body></html>"""


@app.get("/app", response_class=HTMLResponse)
def app_page():
    return APP_HTML


REMOTE_HTML = """<!doctype html><html lang=ru><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Simalee — связь</title>
<style>
:root{--ink:#eef;--card:#171c44;--line:#2a316a;--gold:#e9c46a;--mut:#8a93c8}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:linear-gradient(160deg,#0b0e26,#141a44);color:var(--ink);display:flex;flex-direction:column;height:100dvh}
.wrap{max-width:480px;width:100%;margin:0 auto;display:flex;flex-direction:column;height:100%;padding:10px}
h1{font-size:17px;margin:2px 4px 8px}h1 b{color:var(--gold)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
.bar{font-size:13px;color:var(--mut);background:#141a3e;border:1px solid var(--line);border-radius:12px;padding:8px 11px;margin-bottom:8px;cursor:pointer}
.det{font-size:13px;color:var(--mut);margin-top:6px;display:none}.det.open{display:block}
.det span{display:inline-block;margin-right:12px}
#thread{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:8px;padding:4px}
.msg{max-width:82%;padding:9px 12px;border-radius:14px;font-size:15px;line-height:1.35;word-wrap:break-word}
.you{align-self:flex-end;background:var(--gold);color:#1a1530;border-bottom-right-radius:4px}
.rob{align-self:flex-start;background:var(--card);border:1px solid var(--line);border-bottom-left-radius:4px}
.msg .tm{display:block;font-size:10px;opacity:.6;margin-top:3px}
.hint{align-self:center;color:var(--mut);font-size:12px;padding:6px}
.compose{display:flex;gap:7px;padding-top:8px}
input{flex:1;padding:12px;border-radius:12px;border:1px solid var(--line);background:#0c1030;color:var(--ink);font-size:16px}
.snd{padding:0 15px;border:0;border-radius:12px;background:var(--gold);color:#1a1530;font-weight:700;font-size:18px}
.snd:active{transform:scale(.95)}.say{background:#23306a;color:var(--ink)}
.kc{margin:auto;width:88%;max-width:330px;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;text-align:center}
.kc h2{margin:0 0 4px;font-size:20px}.kc h2 b{color:var(--gold)}
.kc p{margin:0 0 14px;color:var(--mut);font-size:13px}
.kc input{width:100%;display:block;padding:14px;margin:0 0 12px;border-radius:12px;border:1px solid var(--line);background:#0c1030;color:var(--ink);font-size:16px}
.kc button{width:100%;padding:14px;border:0;border-radius:12px;background:var(--gold);color:#1a1530;font-weight:700;font-size:16px}
</style></head><body><div class=wrap>
<div class=kc id=keycard style=display:none>
  <h2>сяма<b>лии</b></h2>
  <p>Введи пароль доступа (один раз)</p>
  <input id=key type=password placeholder="пароль" autocomplete="current-password" onkeydown="if(event.key=='Enter')saveKey()">
  <button onclick=saveKey()>Войти</button>
</div>
<div id=app style="display:none;flex-direction:column;height:100%">
  <h1>сяма<b>лии</b> · чат</h1>
  <div class=bar id=bar onclick="document.getElementById('det').classList.toggle('open')">
    <span id=online>…</span>
    <div class=det id=det>
      <span>🌡 <b id=t>—</b></span><span>💧 <b id=h>—</b></span><span>🔋 <b id=bat>—</b></span>
      <span>🎙 <b id=spk>—</b></span><span>💤 <b id=slp>—</b></span><span>🔊 <b id=vm>—</b></span>
      <span>⚡ <b id=en>—</b></span><span>⏱ <b id=up>—</b></span><span>🌐 <b id=ip>—</b></span>
    </div>
  </div>
  <div id=thread><div class=hint>Напиши роботу — он ответит. Напоминания он пришлёт сюда сам.</div></div>
  <div class=compose>
    <input id=say placeholder="Сообщение роботу…" onkeydown="if(event.key=='Enter')sendChat()">
    <button class=snd title="Сказать вслух дома" onclick=sendSay()>📢</button>
    <button class=snd onclick=sendChat()>➤</button>
  </div>
</div>
<script>
let KEY=localStorage.getItem('simkey')||'',LAST=0,seen={};
function show(){document.getElementById('keycard').style.display=KEY?'none':'block';document.getElementById('app').style.display=KEY?'flex':'none';}
function saveKey(){KEY=document.getElementById('key').value.trim();localStorage.setItem('simkey',KEY);show();poll();status();}
function fmtUp(s){s=+s||0;let h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?h+'ч '+m+'м':m+'м';}
function add(m){if(seen[m.id])return;seen[m.id]=1;
 let th=document.getElementById('thread'),h=th.querySelector('.hint');if(h)h.remove();
 let d=document.createElement('div');d.className='msg '+(m.from=='you'?'you':'rob');
 let tm=new Date((m.t||0)*1000).toLocaleTimeString('ru',{hour:'2-digit',minute:'2-digit'});
 d.innerHTML='';d.textContent=m.text;let s=document.createElement('span');s.className='tm';s.textContent=tm;d.appendChild(s);
 th.appendChild(d);th.scrollTop=th.scrollHeight;}
async function poll(){if(!KEY)return;
 try{let r=await fetch('/outbox_remote?key='+encodeURIComponent(KEY)+'&since='+LAST);
  if(r.status==403){localStorage.removeItem('simkey');KEY='';show();return;}
  let d=await r.json();(d.msgs||[]).forEach(add);LAST=d.last||LAST;}catch(e){}}
async function status(){if(!KEY)return;
 try{let d=await (await fetch('/status_remote?key='+encodeURIComponent(KEY))).json();
  let on=d.age>=0&&d.age<14;
  document.getElementById('online').innerHTML='<span class=dot style="background:'+(on?'#4cc78a':'#e15b4c')+'"></span>'+(on?'На связи':(d.age<0?'Ещё не выходил на связь':'Был '+d.age+'с назад'));
  t.textContent=(d.t&&d.t!='-99')?d.t+'°C':'—';h.textContent=(d.h&&d.h!='-99')?d.h+'%':'—';
  bat.textContent=(d.bat&&d.bat!='-1')?d.bat+'%':'сеть';spk.textContent=(d.spk&&d.spk!='-')?d.spk:'—';
  slp.textContent=(d.slp=='1')?'спит':'не спит';vm.textContent=(d.vol||'?')+'/'+(d.mic||'?')+'%';
  en.textContent=(d.en!=null&&d.en!=='')?d.en+'%':'—';
  up.textContent=fmtUp(d.up);ip.textContent=d.ip||'—';}catch(e){}}
async function sendChat(){let el=document.getElementById('say'),x=el.value.trim();if(!x)return;el.value='';
 await fetch('/chat_remote?key='+encodeURIComponent(KEY)+'&text='+encodeURIComponent(x));poll();}
async function sendSay(){let el=document.getElementById('say'),x=el.value.trim();if(!x)return;el.value='';
 await fetch('/say_remote?key='+encodeURIComponent(KEY)+'&text='+encodeURIComponent(x));
 add({id:'l'+Date.now(),from:'you',text:'📢 (вслух дома) '+x,t:Date.now()/1000});}
show();poll();status();setInterval(poll,2500);setInterval(status,4000);
</script></div></body></html>"""


@app.get("/remote", response_class=HTMLResponse)
def remote_page():
    return REMOTE_HTML


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
