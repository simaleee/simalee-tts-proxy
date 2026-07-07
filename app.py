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


@app.get("/forecast")
async def forecast(key: str = Query(...), lat: float = Query(45.33), lon: float = Query(42.86)):
    """3-day forecast proxied through the server (Open-Meteo is reachable from here even when
    the robot's local DNS is poisoned by a VPN). Returns a compact JSON the ESP can parse cheaply:
    [{"c":code,"n":tmin,"x":tmax}, ...] for today/tomorrow/day-after."""
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           "&daily=weather_code,temperature_2m_max,temperature_2m_min&timezone=auto&forecast_days=3")
    try:
        async with httpx.AsyncClient(timeout=12) as c:
            d = (await c.get(url)).json()
    except Exception:
        return {"error": "forecast unavailable"}
    daily = d.get("daily", {})
    codes = daily.get("weather_code") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    days = []
    for i in range(min(3, len(codes))):
        days.append({
            "c": int(codes[i]) if i < len(codes) else 0,
            "x": round(tmax[i]) if i < len(tmax) else 0,
            "n": round(tmin[i]) if i < len(tmin) else 0,
            "t": WMO.get(int(codes[i]) if i < len(codes) else 0, "облачно"),
        })
    return {"city": "Светлоград", "days": days}


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


def _send_push(text, title="Simalee", icon="/app/usericon.png", badge="/app/usericon.png", tag="simalee"):
    global LAST_PUSH_ERR
    payload = _json.dumps({
        "title": title,
        "body": (text or "")[:300],
        "icon": icon,
        "badge": badge,
        "tag": tag,
    })
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
def push(key: str = Query(...), text: str = Query(..., max_length=300), title: str = Query("Simalee"),
         icon: str = Query("/app/usericon.png"), badge: str = Query("/app/usericon.png"), tag: str = Query("simalee")):
    # robot (or anything) calls this to push a notification to the phone(s)
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    sent, failed = _send_push(text.strip(), title, icon, badge, tag)
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


@app.get("/sw.js")
def service_worker():
    js = (
        "self.addEventListener('install',e=>self.skipWaiting());"
        "self.addEventListener('activate',e=>self.clients.claim());"
        "self.addEventListener('fetch',e=>{});"
        "self.addEventListener('push',e=>{let d={title:'Simalee',body:''};"
        "try{d=e.data.json()}catch(x){try{d.body=e.data.text()}catch(y){}}"
        "e.waitUntil(self.registration.showNotification(d.title||'Simalee',"
        "{body:d.body||'',icon:d.icon||'/app/usericon.png',badge:d.badge||d.icon||'/app/usericon.png',"
        "vibrate:d.vibrate||[120,60,120],tag:d.tag||'simalee'}));});"
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
@keyframes fadeup{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
@keyframes pulse{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.35);opacity:.55}}
@keyframes pop{0%{transform:scale(.6);opacity:0}70%{transform:scale(1.12)}100%{transform:scale(1);opacity:1}}
#app .card{animation:fadeup .38s both}
#app .card:nth-of-type(2){animation-delay:.05s}#app .card:nth-of-type(3){animation-delay:.1s}#app .card:nth-of-type(4){animation-delay:.15s}#app .card:nth-of-type(5){animation-delay:.2s}
button{transition:transform .09s ease,filter .12s}button:active{transform:scale(.93);filter:brightness(1.12)}
.snd:active{transform:scale(.88)}
.dot{animation:pulse 1.6s ease-in-out infinite}
.row .v,.big b{display:inline-block}
.big{animation:pop .4s both}
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
    <div class=big>🔌 Реле (удалёнка) <span id=rstate style="float:right;font-size:14px;font-weight:400;color:var(--mut)">—</span></div>
    <div style="display:flex;gap:8px;margin-top:10px">
      <button style="flex:1;padding:13px;border:0;border-radius:12px;background:var(--grn);color:#06210f;font-weight:700;font-size:15px" onclick="setp('relay',1)">Включить</button>
      <button style="flex:1;padding:13px;border:0;border-radius:12px;background:#3a2530;color:#fff;font-weight:700;font-size:15px" onclick="setp('relay',0)">Выключить</button>
    </div>
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
    <div class=tog><span>🎙 Микрофон (слух)</span><div class=sw id=micsw onclick=tgMic()><i></i></div></div>
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
function tgMic(){let on=!document.getElementById('micsw').classList.contains('on');document.getElementById('micsw').classList.toggle('on',on);setp('micoff',on?0:1);}
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
  if(d.ron!=null)document.getElementById('rstate').textContent=(d.ron=='1'?(d.rrelay=='1'?'● ВКЛ':'○ выкл'):'нет связи');
  if(Date.now()-touched>4000){ // don't fight the user mid-drag
   sv('vol','volv',d.vol,'');sv('mic','micv',d.mic,'%');sv('bri','briv',d.bri,'%');sv('eglow','eglowv',d.eglow,'%');sv('clockbri','clockbriv',d.clockbri,'%');
   if(d.voice&&document.activeElement!=voice)voice.value=d.voice;
   tgset('micsw',d.micon);tgset('gender',d.gender);tgset('chirp',d.chirp);tgset('adiag',d.adiag);
  }
 }catch(e){}}
function sv(id,lab,v,suf){if(v==null||v==='')return;let el=document.getElementById(id);if(el!==document.activeElement){el.value=v;document.getElementById(lab).textContent=v+suf;}}
function tgset(id,v){document.getElementById(id).classList.toggle('on',v=='1'||v===1||v===true||v==='true');}
document.addEventListener('input',e=>{if(e.target.type=='range')touched=Date.now();},true);
if('serviceWorker'in navigator){
 navigator.serviceWorker.getRegistrations().then(rs=>{for(const r of rs){if(r.scope.endsWith('/app/'))r.unregister();}});
 navigator.serviceWorker.register('/sw.js').then(r=>r.update()).catch(e=>{let el=document.getElementById('pushdiag');if(el)el.textContent='SW register error: '+e;});
}
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
let _ac;function beep(f,d){try{_ac=_ac||new(window.AudioContext||window.webkitAudioContext)();let o=_ac.createOscillator(),g=_ac.createGain();o.type='triangle';o.frequency.value=f||780;g.gain.value=.05;o.connect(g);g.connect(_ac.destination);let t=_ac.currentTime;o.start(t);g.gain.exponentialRampToValueAtTime(.0001,t+(d||.07));o.stop(t+(d||.1));}catch(e){}}
document.addEventListener('click',e=>{let el=e.target.closest('button,.sw');if(el)beep(el.classList.contains('sw')?520:(el.classList.contains('snd')?920:780),.06);},true);
show();tick();if(KEY){pushStatus();pushDiag();}setInterval(tick,3000);
</script></div></body></html>"""


# New /app shell. The legacy APP_HTML above is kept as a fallback reference while the
# Render PWA moves to the glass-style control panel.
APP_HTML = r"""<!doctype html><html lang=ru><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name=theme-color content="#e9f0f2"><title>Simalee</title>
<link rel=manifest href="/app/manifest.webmanifest">
<link rel=apple-touch-icon href="/app/icon-192.png">
<style>
:root{--bg:#e8eff1;--paper:rgba(255,255,255,.62);--paper2:rgba(255,255,255,.84);--ink:#172126;--mut:#65737a;--line:rgba(69,86,94,.18);--accent:#2563eb;--accent2:#05a8aa;--warm:#e7b95b;--ok:#21a67a;--bad:#d4505f;--shadow:10px 14px 30px rgba(75,92,99,.20),-8px -8px 24px rgba(255,255,255,.68);--r:8px}
body[data-theme=night]{--bg:#22282a;--paper:rgba(45,51,52,.72);--paper2:rgba(50,57,58,.92);--ink:#edf7f7;--mut:#aab7b8;--line:rgba(255,255,255,.10);--accent:#2ed3ff;--accent2:#e8892e;--warm:#e8892e;--shadow:9px 9px 20px rgba(0,0,0,.38),-5px -5px 16px rgba(255,255,255,.045)}
body[data-theme=mono]{--bg:#f4f2ed;--paper:rgba(255,255,255,.68);--paper2:rgba(255,255,255,.9);--ink:#24211b;--mut:#766f62;--line:rgba(88,76,55,.16);--accent:#178f8d;--accent2:#c05f46;--warm:#d0a44f;--shadow:9px 12px 24px rgba(91,80,60,.18),-8px -8px 20px rgba(255,255,255,.72)}
*{box-sizing:border-box}html,body{min-height:100%;margin:0}body{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;background:linear-gradient(145deg,var(--bg),color-mix(in srgb,var(--bg),#ffffff 22%));color:var(--ink);letter-spacing:0}
button,input,select{font:inherit}button{border:0;color:inherit;cursor:pointer}button:active{transform:translateY(1px) scale(.985);filter:brightness(.96)}.wrap{width:min(980px,100%);margin:0 auto;padding:14px 14px calc(86px + env(safe-area-inset-bottom))}
.top{display:flex;align-items:center;gap:12px;margin:2px 0 14px}.avatar{width:52px;height:52px;border-radius:50%;object-fit:cover;box-shadow:var(--shadow);border:1px solid var(--line);background:var(--paper2)}.brand{min-width:0;flex:1}.brand h1{font-size:22px;line-height:1.05;margin:0;font-weight:760}.brand p{margin:4px 0 0;color:var(--mut);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.pill{display:inline-flex;align-items:center;gap:7px;padding:8px 10px;border-radius:999px;background:var(--paper2);border:1px solid var(--line);box-shadow:var(--shadow);font-size:13px;font-weight:650}.dot{width:9px;height:9px;border-radius:50%;background:var(--bad);box-shadow:0 0 0 5px color-mix(in srgb,var(--bad),transparent 82%)}.dot.ok{background:var(--ok);box-shadow:0 0 0 5px color-mix(in srgb,var(--ok),transparent 82%)}
.themes{display:flex;gap:6px}.themes button{width:28px;height:28px;border-radius:50%;background:var(--paper2);border:1px solid var(--line);box-shadow:var(--shadow)}.themes button[data-v=glass]{background:linear-gradient(135deg,#f8ffff,#cfe6ee)}.themes button[data-v=night]{background:linear-gradient(135deg,#1f2426,#3a4243)}.themes button[data-v=mono]{background:linear-gradient(135deg,#fff8e8,#d7c29b)}
.panel{display:none;animation:rise .28s ease both}.panel.active{display:block}.grid{display:grid;grid-template-columns:1fr;gap:10px}.card{background:var(--paper);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow);backdrop-filter:blur(18px);padding:14px;min-width:0}.hero{overflow:hidden;position:relative}.hero:before{content:"";position:absolute;inset:0;background:linear-gradient(120deg,transparent,color-mix(in srgb,var(--accent),transparent 86%),transparent);transform:translateX(-120%);animation:sweep 6s ease-in-out infinite}.hero>*{position:relative}.hero h2{font-size:18px;margin:0 0 10px}.hero .big{font-size:34px;font-weight:780;margin:0}.hero .sub{color:var(--mut);font-size:13px;margin-top:4px}.stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.stat{padding:12px;border-radius:var(--r);background:color-mix(in srgb,var(--paper2),transparent 8%);border:1px solid var(--line);min-height:78px}.stat b{display:block;font-size:21px;margin-top:8px}.stat span{color:var(--mut);font-size:12px}.stat.good b{color:var(--ok)}.stat.warn b{color:var(--bad)}
.section-title{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:10px}.section-title h2{font-size:16px;margin:0}.section-title small{color:var(--mut)}.row{display:flex;justify-content:space-between;gap:10px;padding:9px 0;border-bottom:1px solid var(--line);font-size:14px}.row:last-child{border-bottom:0}.row .k{color:var(--mut)}.row .v{text-align:right;font-weight:670;word-break:break-word}.set{margin:12px 0}.set label{display:flex;justify-content:space-between;color:var(--mut);font-size:13px;margin-bottom:7px}.set label b{color:var(--ink)}input[type=range]{width:100%;accent-color:var(--accent);height:28px}select,input[type=text],input[type=password]{width:100%;padding:12px;border-radius:var(--r);border:1px solid var(--line);background:var(--paper2);color:var(--ink);outline:none}.tog{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 0;border-bottom:1px solid var(--line)}.tog:last-child{border-bottom:0}.sw{width:48px;height:28px;border-radius:999px;background:color-mix(in srgb,var(--mut),transparent 72%);position:relative;flex:0 0 auto;transition:.18s}.sw i{position:absolute;width:22px;height:22px;left:3px;top:3px;background:#fff;border-radius:50%;box-shadow:0 2px 7px rgba(0,0,0,.18);transition:.18s}.sw.on{background:var(--ok)}.sw.on i{left:23px}
.btns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.btn,.chip{border-radius:var(--r);padding:12px;background:var(--paper2);border:1px solid var(--line);box-shadow:var(--shadow);font-weight:700;text-align:center}.btn.primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff}.btn.warm{background:linear-gradient(135deg,var(--warm),#f4d78b);color:#31230d}.btn.bad{background:linear-gradient(135deg,#d4505f,#9c3544);color:#fff}.chips{display:flex;gap:8px;overflow:auto;padding:2px 1px 8px;scrollbar-width:none}.chips::-webkit-scrollbar{display:none}.chip{white-space:nowrap;box-shadow:none;padding:9px 11px;font-size:13px}.thread{height:290px;overflow:auto;display:flex;flex-direction:column;gap:8px;padding:2px}.msg{max-width:86%;border-radius:var(--r);padding:10px 12px;font-size:14px;line-height:1.35;border:1px solid var(--line)}.msg.you{align-self:flex-end;background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff}.msg.rob{align-self:flex-start;background:var(--paper2)}.msg .tm{display:block;font-size:10px;opacity:.62;margin-top:4px}.composer{display:grid;grid-template-columns:1fr 46px 46px;gap:8px;margin-top:10px}.composer button{border-radius:var(--r);background:var(--paper2);border:1px solid var(--line);font-size:18px}
.login{min-height:100dvh;display:grid;place-items:center;padding:18px}.login-card{width:min(360px,100%);padding:22px;background:var(--paper);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow);backdrop-filter:blur(18px);text-align:center}.login-card img{width:72px;height:72px;border-radius:50%;box-shadow:var(--shadow);margin-bottom:10px}.login-card h1{margin:0;font-size:24px}.login-card p{color:var(--mut);font-size:13px;margin:6px 0 16px}.login-card button{width:100%;margin-top:10px;border-radius:var(--r);padding:13px;background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;font-weight:760}.filepick{display:flex;align-items:center;gap:10px;margin-top:7px;padding:8px;border-radius:var(--r);background:var(--paper2);border:1px solid var(--line);cursor:pointer;overflow:hidden}.filepick input{position:absolute;opacity:0;pointer-events:none}.filepick span{padding:9px 11px;border-radius:var(--r);background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;font-weight:700}.filepick em{font-style:normal;color:var(--mut);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.tabbar{position:fixed;left:50%;bottom:calc(10px + env(safe-area-inset-bottom));transform:translateX(-50%);width:min(560px,calc(100% - 18px));display:grid;grid-template-columns:repeat(4,1fr);gap:6px;padding:7px;border-radius:var(--r);background:color-mix(in srgb,var(--paper2),transparent 4%);border:1px solid var(--line);box-shadow:var(--shadow);backdrop-filter:blur(18px)}.tabbar button{border-radius:var(--r);padding:9px 4px;background:transparent;color:var(--mut);font-size:12px}.tabbar button.active{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff}.note{color:var(--mut);font-size:12px}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}.diag{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:11px;color:var(--mut);background:color-mix(in srgb,var(--ink),transparent 92%);border:1px solid var(--line);border-radius:var(--r);padding:8px;margin-top:8px;overflow:auto}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}@keyframes sweep{0%,45%{transform:translateX(-130%)}75%,100%{transform:translateX(130%)}}@media (min-width:760px){.grid{grid-template-columns:1.1fr .9fr}.stats{grid-template-columns:repeat(4,1fr)}.panel[data-panel=controls] .grid{grid-template-columns:1fr 1fr}.panel[data-panel=notify] .grid{grid-template-columns:1fr 1fr}.thread{height:390px}}@media (prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
</style></head><body><div id=login class=login style=display:none><div class=login-card>
  <img src="/app/usericon.png" alt=""><h1>Simalee</h1><p>Удалённая панель</p>
  <input id=key type=password placeholder="Пароль" autocomplete=current-password onkeydown="if(event.key==='Enter')saveKey()">
  <button onclick=saveKey()>Войти</button>
</div></div>
<div id=app style=display:none><div class=wrap>
  <div class=top>
    <img class=avatar src="/app/usericon.png" alt="">
    <div class=brand><h1>Simalee</h1><p id=subtitle>удалённая связь с роботом</p></div>
    <div class=pill id=online><span class=dot></span><span>...</span></div>
    <div class=themes><button data-v=glass title=Glass onclick="theme('glass')"></button><button data-v=night title=Night onclick="theme('night')"></button><button data-v=mono title=Soft onclick="theme('mono')"></button></div>
  </div>

  <section class="panel active" data-panel=status>
    <div class=grid>
      <div class="card hero"><h2>Состояние</h2><p class=big id=mainState>...</p><p class=sub id=mainSub>ожидаю данные</p></div>
      <div class=card><div class=section-title><h2>Система</h2><small id=fw>—</small></div>
        <div class=row><span class=k>IP дома</span><span class=v id=ip>—</span></div>
        <div class=row><span class=k>Собеседник</span><span class=v id=spk>—</span></div>
        <div class=row><span class=k>PC Bridge</span><span class=v id=pc>—</span></div>
        <div class=row><span class=k>AI модель</span><span class=v id=aimodel>—</span></div>
      </div>
    </div>
    <div class=stats style="margin-top:10px">
      <div class=stat><span>Температура</span><b id=t>—</b></div><div class=stat><span>Влажность</span><b id=h>—</b></div>
      <div class=stat><span>Энергия</span><b id=en>—</b></div><div class=stat><span>Батарея</span><b id=bat>—</b></div>
    </div>
    <div class=grid style="margin-top:10px">
      <div class=card><div class=section-title><h2>Тело</h2><small id=touchState>—</small></div>
        <div class=row><span class=k>Сон</span><span class=v id=slp>—</span></div>
        <div class=row><span class=k>Реле</span><span class=v id=rstate>—</span></div>
        <div class=row><span class=k>Касание</span><span class=v id=touch>—</span></div>
        <div class=row><span class=k>Аптайм</span><span class=v id=up>—</span></div>
      </div>
      <div class=card><div class=section-title><h2>Производительность</h2><small>STT / AI / TTS</small></div>
        <div class=row><span class=k>Распознавание</span><span class=v id=sttms>—</span></div>
        <div class=row><span class=k>Мозг</span><span class=v id=aims>—</span></div>
        <div class=row><span class=k>Голос</span><span class=v id=ttsms>—</span></div>
        <div class=row><span class=k>Ошибки AI</span><span class=v id=aifails>—</span></div>
      </div>
    </div>
  </section>

  <section class=panel data-panel=controls>
    <div class=grid>
      <div class=card><div class=section-title><h2>Настройки</h2><small id=setnote>готово</small></div>
        <div class=set><label>Громкость <b id=volv>—</b></label><input type=range id=vol min=0 max=21 oninput="lv('volv',this.value)" onchange="setp('vol',this.value)"></div>
        <div class=set><label>Микрофон <b id=micv>—</b></label><input type=range id=mic min=0 max=100 oninput="lv('micv',this.value,'%')" onchange="setp('mic',this.value)"></div>
        <div class=set><label>Экран <b id=briv>—</b></label><input type=range id=bri min=5 max=100 oninput="lv('briv',this.value,'%')" onchange="setp('bri',this.value)"></div>
        <div class=set><label>Прозрачность лица <b id=eglowv>—</b></label><input type=range id=eglow min=5 max=100 oninput="lv('eglowv',this.value,'%')" onchange="setp('eglow',this.value)"></div>
        <div class=set><label>OLED помощника <b id=obriv>—</b></label><input type=range id=obri min=1 max=100 oninput="lv('obriv',this.value,'%')" onchange="setp('obri',this.value)"></div>
        <div class=set><label>Часы во сне <b id=clockbriv>—</b></label><input type=range id=clockbri min=5 max=100 oninput="lv('clockbriv',this.value,'%')" onchange="setp('clockbri',this.value)"></div>
        <div class=set><label>Голос</label><select id=voice onchange="setp('voice',this.value)"><option value=ru-RU-SvetlanaNeural>Светлана</option><option value=ru-RU-DmitryNeural>Дмитрий</option><option value=ru-RU-DariyaNeural>Дария</option></select></div>
      </div>
      <div class=card><div class=section-title><h2>Переключатели</h2><small>живые</small></div>
        <div class=tog><span>Микрофон</span><div class=sw id=micsw onclick=tgMic()><i></i></div></div>
        <div class=tog><span>Голос наоборот</span><div class=sw id=gender onclick="tg('gender','gender')"><i></i></div></div>
        <div class=tog><span>Звуки в покое</span><div class=sw id=chirp onclick="tg('chirp','chirp')"><i></i></div></div>
        <div class=tog><span>Авто-диагностика</span><div class=sw id=adiag onclick="tg('adiag','adiag')"><i></i></div></div>
        <div class=tog><span>Реакция на касание</span><div class=sw id=touchreact onclick="tg('touchreact','touchreact')"><i></i></div></div>
        <div class=section-title style="margin-top:14px"><h2>Реле</h2><small id=relayHint>—</small></div>
        <div class=btns><button class="btn primary" onclick="setp('relay',1)">Включить</button><button class="btn bad" onclick="setp('relay',0)">Выключить</button></div>
      </div>
    </div>
  </section>

  <section class=panel data-panel=chat>
    <div class=card><div class=section-title><h2>Связь</h2><small>текст / голос дома</small></div>
      <div class=chips>
        <button class=chip onclick="quick('моргни одним глазом')">подмигни</button><button class=chip onclick="quick('покажи сердечко')">сердце</button>
        <button class=chip onclick="quick('какой статус')">статус</button><button class=chip onclick="quick('запусти музыку')">музыка</button>
        <button class=chip onclick="quick('останови музыку')">стоп</button><button class=chip onclick="quick('покажи сети вай фай рядом')">WiFi</button>
      </div>
      <div id=thread class=thread><div class=note>Сообщения Simalee появятся здесь.</div></div>
      <div class=composer><input id=say type=text placeholder="Сообщение роботу..." onkeydown="if(event.key==='Enter')sendChat()"><button title="Сказать дома" onclick=sendSay()>▶</button><button title="Отправить в чат" onclick=sendChat()>↗</button></div>
    </div>
  </section>

  <section class=panel data-panel=notify>
    <div class=grid>
      <div class=card><div class=section-title><h2>Уведомления</h2><small id=pushCount>—</small></div>
        <p class=note id=pushst>Проверка...</p><div class=diag id=pushdiag>диагностика...</div>
        <div class=btns style="margin-top:10px"><button class="btn primary" onclick=enablePush()>Включить</button><button class="btn warm" onclick=testPush()>Тест</button></div>
        <label class=note style="display:block;margin-top:12px">Иконка уведомлений</label><label class=filepick><input type=file id=iconf accept="image/*" onchange=uploadIcon()><span>Выбрать</span><em id=iconname>файл не выбран</em></label>
      </div>
      <div class=card><div class=section-title><h2>Облако</h2><small>Render</small></div>
        <div class=row><span class=k>Подписок</span><span class=v id=subs>—</span></div>
        <div class=row><span class=k>Последняя ошибка</span><span class=v id=pusherr>—</span></div>
        <div class=row><span class=k>Очередь команд</span><span class=v id=pending>—</span></div>
        <div class=row><span class=k>Связь</span><span class=v id=cloudage>—</span></div>
      </div>
    </div>
  </section>
</div><nav class=tabbar><button class=active data-tab=status onclick="nav('status')">Статус</button><button data-tab=controls onclick="nav('controls')">Пульт</button><button data-tab=chat onclick="nav('chat')">Связь</button><button data-tab=notify onclick="nav('notify')">Push</button></nav></div>
<script>
let KEY=localStorage.getItem('simkey')||'',LAST=0,seen={},touched=0,st={};
document.body.dataset.theme=localStorage.getItem('simTheme')||'glass';
function $(id){return document.getElementById(id)}
function show(){$('login').style.display=KEY?'none':'grid';$('app').style.display=KEY?'block':'none'}
function saveKey(){KEY=$('key').value.trim();localStorage.setItem('simkey',KEY);show();tick();poll();pushStatus();pushDiag()}
function theme(v){document.body.dataset.theme=v;localStorage.setItem('simTheme',v);beep(620,.05)}
function nav(p){document.querySelectorAll('.panel').forEach(x=>x.classList.toggle('active',x.dataset.panel===p));document.querySelectorAll('.tabbar button').forEach(x=>x.classList.toggle('active',x.dataset.tab===p));beep(740,.05)}
function fmtUp(s){s=+s||0;let h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?h+'ч '+m+'м':m+'м'}
function lv(id,v,suf=''){$(id).textContent=v+suf}
function setp(k,v){fetch('/set_remote?key='+encodeURIComponent(KEY)+'&'+k+'='+encodeURIComponent(v));$('setnote').textContent=k+' = '+v;beep(820,.04)}
function sv(id,lab,v,suf=''){if(v==null||v==='')return;let el=$(id);if(el&&el!==document.activeElement){el.value=v;$(lab).textContent=v+suf}}
function tgset(id,v){let el=$(id);if(el)el.classList.toggle('on',v==1||v==='1'||v===true||v==='true')}
function tg(id,k){let on=!$(id).classList.contains('on');$(id).classList.toggle('on',on);setp(k,on?1:0)}
function tgMic(){let on=!$('micsw').classList.contains('on');$('micsw').classList.toggle('on',on);setp('micoff',on?0:1)}
async function tick(){if(!KEY)return;try{let r=await fetch('/status_remote?key='+encodeURIComponent(KEY));if(r.status===403){localStorage.removeItem('simkey');KEY='';show();return}let d=await r.json();st=d;let on=d.age>=0&&d.age<14;$('online').innerHTML='<span class="dot '+(on?'ok':'')+'"></span><span>'+(on?'online':(d.age<0?'new':'offline'))+'</span>';$('mainState').textContent=on?(d.slp==='1'?'спит':'на связи'):'не на связи';$('mainSub').textContent=on?('последний пакет '+d.age+'с назад'):(d.age<0?'робот ещё не присылал статус':'молчание '+d.age+'с');$('subtitle').textContent=(d.speaker&&d.speaker!=='not recognized')?'говорит: '+d.speaker:'удалённая связь с роботом';
val('t',d.t&&d.t!='-99'?d.t+' °C':'—');val('h',d.h&&d.h!='-99'?d.h+' %':'—');val('en',d.en!=null&&d.en!==''?d.en+' %':'—');val('bat',d.bat&&d.bat!='-1'?d.bat+' %':'от сети');val('ip',d.ip||'—');val('slp',d.slp==='1'?'спит':'бодрствует');val('up',fmtUp(d.up));val('spk',(d.speaker&&d.speaker!=='not recognized')?d.speaker:'—');val('pc',d.pconline==='true'||d.pconline==='1'?'online':(d.pcon==='true'||d.pcon==='1'?'нет ответа':'выкл'));val('aimodel',d.aimodel||'—');val('fw',d.fw||'—');val('rstate',d.ron==='1'?(d.rrelay==='1'?'включено':'выключено'):'нет связи');val('relayHint',d.ron==='1'?'модуль на связи':'нет связи');val('touch',d.touch==='1'?'есть':'—');val('touchState',d.touchsens?('сенсор '+d.touchsens+'%'):'—');val('sttms',ms(d.sttms));val('aims',ms(d.aims));val('ttsms',ms(d.ttsms));val('aifails',d.aifails||'0');val('pending',d.pending||0);val('cloudage',d.age>=0?d.age+'с':'—');
if(Date.now()-touched>3500){sv('vol','volv',d.vol);sv('mic','micv',d.mic,'%');sv('bri','briv',d.bri,'%');sv('eglow','eglowv',d.eglow,'%');sv('obri','obriv',d.obri,'%');sv('clockbri','clockbriv',d.clockbri,'%');if(d.voice&&document.activeElement!==$('voice'))$('voice').value=d.voice;tgset('micsw',d.micon);tgset('gender',d.gender);tgset('chirp',d.chirp);tgset('adiag',d.adiag);tgset('touchreact',d.touchreact)}}catch(e){}}
function val(id,v){let el=$(id);if(el)el.textContent=v}
function ms(v){v=+v||0;return v?v+' мс':'—'}
document.addEventListener('input',e=>{if(e.target.type==='range')touched=Date.now()},true);
function add(m){if(seen[m.id])return;seen[m.id]=1;let th=$('thread'),hint=th.querySelector('.note');if(hint)hint.remove();let d=document.createElement('div');d.className='msg '+(m.from==='you'?'you':'rob');d.textContent=m.text;let tm=document.createElement('span');tm.className='tm';tm.textContent=new Date((m.t||Date.now()/1000)*1000).toLocaleTimeString('ru',{hour:'2-digit',minute:'2-digit'});d.appendChild(tm);th.appendChild(d);th.scrollTop=th.scrollHeight}
async function poll(){if(!KEY)return;try{let d=await(await fetch('/outbox_remote?key='+encodeURIComponent(KEY)+'&since='+LAST)).json();(d.msgs||[]).forEach(add);LAST=d.last||LAST}catch(e){}}
async function sendChat(){let x=$('say').value.trim();if(!x)return;$('say').value='';add({id:'l'+Date.now(),from:'you',text:x,t:Date.now()/1000});await fetch('/chat_remote?key='+encodeURIComponent(KEY)+'&text='+encodeURIComponent(x));poll()}
async function sendSay(){let x=$('say').value.trim();if(!x)return;$('say').value='';add({id:'s'+Date.now(),from:'you',text:'дома: '+x,t:Date.now()/1000});await fetch('/say_remote?key='+encodeURIComponent(KEY)+'&text='+encodeURIComponent(x))}
function quick(x){$('say').value=x;sendChat()}
if('serviceWorker'in navigator){navigator.serviceWorker.register('/sw.js').then(r=>r.update()).catch(e=>{let el=$('pushdiag');if(el)el.textContent='SW: '+e})}
function urlB64(s){let p='='.repeat((4-s.length%4)%4);let b=atob((s+p).replace(/-/g,'+').replace(/_/g,'/'));return Uint8Array.from([...b].map(c=>c.charCodeAt(0)))}
async function enablePush(){try{$('pushst').textContent='Проверяю...';if(!('serviceWorker'in navigator)){pushText('Нет serviceWorker');return}if(!('PushManager'in window)){pushText('Нет Web Push');return}let perm=await Notification.requestPermission();if(perm!=='granted'){pushText('Разрешение: '+perm);return}let reg=await Promise.race([navigator.serviceWorker.ready,new Promise((_,rej)=>setTimeout(()=>rej(new Error('service worker timeout')),8000))]);let pub=(await(await fetch('/vapid_public')).text()).trim();let sub=await reg.pushManager.getSubscription();if(!sub)sub=await reg.pushManager.subscribe({userVisibleOnly:true,applicationServerKey:urlB64(pub)});let r=await fetch('/push_subscribe?key='+encodeURIComponent(KEY),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(sub)});await pushStatus();pushText(r.ok?'Уведомления включены':'Подписка не сохранилась')}catch(e){pushText('Ошибка: '+(e.message||e))}}
function pushText(x){$('pushst').textContent=x}
async function testPush(){try{let r=await(await fetch('/push?key='+encodeURIComponent(KEY)+'&title=Simalee&text='+encodeURIComponent('Тестовое уведомление'))).json();pushText(r.sent>0?'Отправлено: '+r.sent:(r.subs==0?'Нет подписки':'Не доставлено'));await pushStatus()}catch(e){pushText('Ошибка теста: '+e)}}
async function uploadIcon(){let f=$('iconf').files[0];if(!f)return;$('iconname').textContent=f.name;try{let b=await f.arrayBuffer();let r=await fetch('/set_icon?key='+encodeURIComponent(KEY),{method:'POST',body:b});pushText(r.ok?'Иконка обновлена':'Не удалось загрузить')}catch(e){pushText('Ошибка иконки: '+e)}}
async function pushStatus(){if(!KEY)return;try{let c=await(await fetch('/push_count?key='+encodeURIComponent(KEY))).json();val('subs',c.subs||0);val('pushCount',(c.subs||0)+' устр.');val('pusherr',c.err||'—');if(c.subs>0)pushText('Подписано устройств: '+c.subs)}catch(e){}}
function pushDiag(){try{let sw=('serviceWorker'in navigator),pm=('PushManager'in window),perm=(window.Notification?Notification.permission:'нет'),inst=(matchMedia('(display-mode: standalone)').matches||navigator.standalone)?'да':'нет';$('pushdiag').textContent='SW '+(sw?'ok':'no')+' · Push '+(pm?'ok':'no')+' · '+perm+' · PWA '+inst}catch(e){}}
let _ac;function beep(f=720,d=.05){try{_ac=_ac||new(window.AudioContext||window.webkitAudioContext)();let o=_ac.createOscillator(),g=_ac.createGain();o.type='sine';o.frequency.value=f;g.gain.value=.045;o.connect(g);g.connect(_ac.destination);let t=_ac.currentTime;o.start(t);g.gain.exponentialRampToValueAtTime(.0001,t+d);o.stop(t+d+.02)}catch(e){}}
document.addEventListener('click',e=>{if(e.target.closest('button,.sw'))beep()},true);window.addEventListener('error',e=>{let d=$('pushdiag');if(d)d.textContent='JS: '+(e.message||e)});
show();tick();poll();if(KEY){pushStatus();pushDiag()}setInterval(tick,3000);setInterval(poll,2500);
</script></body></html>"""

APP_HTML_JP = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#e9f0ed"><title>Simalee</title>
<link rel="manifest" href="/app/manifest.webmanifest"><link rel="apple-touch-icon" href="/app/icon-192.png">
<style>
:root{--bg:#e8efed;--card:rgba(255,255,255,.66);--card2:rgba(255,255,255,.88);--ink:#192423;--mut:#667472;--line:rgba(52,74,72,.16);--a:#1f7a8c;--b:#d36b3d;--ok:#2e9d70;--bad:#c94c5d;--sh:10px 12px 28px rgba(56,76,74,.18),-8px -8px 24px rgba(255,255,255,.78)}
body[data-theme=night]{--bg:#202625;--card:rgba(38,47,47,.72);--card2:rgba(45,56,56,.9);--ink:#eff8f6;--mut:#a8b8b5;--line:rgba(255,255,255,.1);--a:#42d9ff;--b:#e68a3f;--sh:9px 10px 24px rgba(0,0,0,.38),-5px -5px 16px rgba(255,255,255,.045)}
body[data-theme=paper]{--bg:#f2eee5;--card:rgba(255,255,255,.68);--card2:rgba(255,255,255,.92);--ink:#28241d;--mut:#776f62;--line:rgba(88,76,55,.16);--a:#147c7a;--b:#b85d45}
*{box-sizing:border-box}html,body{margin:0;min-height:100%}body{font-family:"Yu Mincho","Noto Serif",Georgia,"Segoe UI",serif;background:radial-gradient(circle at 16% 2%,color-mix(in srgb,var(--a),transparent 82%),transparent 30%),linear-gradient(145deg,var(--bg),color-mix(in srgb,var(--bg),#fff 18%));color:var(--ink);letter-spacing:0}
button,input,select{font:inherit}button{border:0;color:inherit;cursor:pointer}button:active{transform:scale(.98);filter:brightness(.96)}.wrap{width:min(860px,100%);margin:0 auto;padding:12px 12px calc(76px + env(safe-area-inset-bottom))}
.login{min-height:100dvh;display:grid;place-items:center;padding:18px}.login .box{width:min(340px,100%);padding:22px;border-radius:10px;background:var(--card);border:1px solid var(--line);box-shadow:var(--sh);backdrop-filter:blur(18px);text-align:center}.login img,.avatar{width:54px;height:54px;border-radius:50%;object-fit:cover;border:1px solid var(--line);box-shadow:var(--sh)}.login h1{margin:8px 0 4px}.login p,.note{color:var(--mut);font-size:13px}.login input{width:100%;padding:13px;border-radius:8px;border:1px solid var(--line);background:var(--card2);color:var(--ink);outline:none}.login button,.btn{padding:12px;border-radius:8px;background:var(--card2);border:1px solid var(--line);box-shadow:var(--sh);font-weight:760}.login button,.btn.pri{background:linear-gradient(135deg,var(--a),var(--b));color:white}
.top{display:flex;align-items:center;gap:10px;margin:2px 0 10px}.brand{min-width:0;flex:1}.brand h1{font-size:23px;margin:0;line-height:1}.brand p{margin:4px 0 0;color:var(--mut);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.seal{font-size:21px;color:var(--b);font-weight:800}.pill{display:inline-flex;align-items:center;gap:7px;padding:8px 10px;border-radius:999px;background:var(--card2);border:1px solid var(--line);box-shadow:var(--sh);font:700 13px system-ui}.dot{width:9px;height:9px;border-radius:50%;background:var(--bad);box-shadow:0 0 0 5px color-mix(in srgb,var(--bad),transparent 82%)}.dot.ok{background:var(--ok);box-shadow:0 0 0 5px color-mix(in srgb,var(--ok),transparent 82%)}
.themes{display:flex;gap:6px}.themes button{width:29px;height:29px;border-radius:50%;background:var(--card2);border:1px solid var(--line);box-shadow:var(--sh)}.themes button:nth-child(1){background:linear-gradient(135deg,#f8ffff,#cfe8e3)}.themes button:nth-child(2){background:linear-gradient(135deg,#1d2324,#3a4546)}.themes button:nth-child(3){background:linear-gradient(135deg,#fff8e8,#d7c29b)}
.panel{display:none}.panel.on{display:block;animation:rise .24s ease both}.grid{display:grid;grid-template-columns:1fr;gap:10px}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;box-shadow:var(--sh);backdrop-filter:blur(18px);padding:13px;min-width:0}.hero{position:relative;overflow:hidden}.hero:before{content:"";position:absolute;inset:0;background:linear-gradient(120deg,transparent,color-mix(in srgb,var(--a),transparent 84%),transparent);transform:translateX(-120%);animation:sweep 5.5s infinite}.hero>*{position:relative}.big{font-size:31px;font-weight:800;margin:3px 0}.sub{color:var(--mut);font-size:13px}.head{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:8px}.head h2{font-size:16px;margin:0}.head small{color:var(--mut)}
.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:9px}.stat{background:var(--card2);border:1px solid var(--line);border-radius:8px;padding:11px;min-height:72px}.stat span{display:block;color:var(--mut);font-size:12px}.stat b{display:block;font-size:21px;margin-top:7px}.row{display:flex;justify-content:space-between;gap:10px;padding:8px 0;border-bottom:1px solid var(--line);font-size:14px}.row:last-child{border:0}.k{color:var(--mut)}.v{text-align:right;font-weight:760;word-break:break-word}.set{margin:12px 0}.set label{display:flex;justify-content:space-between;color:var(--mut);font-size:13px;margin-bottom:6px}.set b{color:var(--ink)}input[type=range]{width:100%;accent-color:var(--a)}select,input[type=text],input[type=password]{width:100%;padding:12px;border-radius:8px;border:1px solid var(--line);background:var(--card2);color:var(--ink);outline:none}
.tog{display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--line)}.sw{width:48px;height:28px;border-radius:999px;background:color-mix(in srgb,var(--mut),transparent 70%);position:relative}.sw i{position:absolute;left:3px;top:3px;width:22px;height:22px;border-radius:50%;background:white;transition:.18s;box-shadow:0 2px 7px rgba(0,0,0,.18)}.sw.on{background:var(--ok)}.sw.on i{left:23px}.btns{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.btn.bad{background:linear-gradient(135deg,#c94c5d,#87303b);color:white}.chips{display:flex;gap:8px;overflow:auto;padding:2px 1px 8px}.chip{white-space:nowrap;padding:9px 11px;border-radius:8px;background:var(--card2);border:1px solid var(--line);font-weight:700}.thread{height:310px;overflow:auto;display:flex;flex-direction:column;gap:8px}.msg{max-width:86%;padding:10px 12px;border-radius:8px;border:1px solid var(--line);font:14px/1.35 system-ui}.msg.you{align-self:flex-end;background:linear-gradient(135deg,var(--a),var(--b));color:white}.msg.rob{align-self:flex-start;background:var(--card2)}.tm{display:block;font-size:10px;opacity:.62;margin-top:4px}.composer{display:grid;grid-template-columns:1fr 45px 45px;gap:8px;margin-top:10px}.composer button{border-radius:8px;background:var(--card2);border:1px solid var(--line)}
.tabs{position:fixed;left:50%;bottom:calc(9px + env(safe-area-inset-bottom));transform:translateX(-50%);width:min(560px,calc(100% - 18px));display:grid;grid-template-columns:repeat(4,1fr);gap:6px;padding:7px;border-radius:8px;background:color-mix(in srgb,var(--card2),transparent 4%);border:1px solid var(--line);box-shadow:var(--sh);backdrop-filter:blur(18px)}.tabs button{border-radius:7px;padding:9px 3px;background:transparent;color:var(--mut);font:700 12px system-ui}.tabs button.on{background:linear-gradient(135deg,var(--a),var(--b));color:white}.diag{font:11px/1.35 ui-monospace,Consolas,monospace;color:var(--mut);background:color-mix(in srgb,var(--ink),transparent 92%);border:1px solid var(--line);border-radius:8px;padding:8px;margin-top:8px;overflow:auto}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}@keyframes sweep{0%,45%{transform:translateX(-130%)}75%,100%{transform:translateX(130%)}}@media(min-width:760px){.grid{grid-template-columns:1.05fr .95fr}.stats{grid-template-columns:repeat(4,1fr)}.thread{height:390px}}
</style></head><body>
<div id="login" class="login" style="display:none"><div class="box"><img src="/app/usericon.png" alt=""><h1>Simalee</h1><p>удалённая панель</p><input id="key" type="password" placeholder="пароль" autocomplete="current-password" onkeydown="if(event.key==='Enter')saveKey()"><button onclick="saveKey()">Войти</button></div></div>
<div id="app" style="display:none"><div class="wrap">
<div class="top"><img class="avatar" src="/app/usericon.png" alt=""><div class="brand"><h1>Simalee <span class="seal">和</span></h1><p id="subtitle">удалённая связь с роботом</p></div><div class="pill" id="online"><span class="dot"></span><span>...</span></div><div class="themes"><button onclick="theme('')"></button><button onclick="theme('night')"></button><button onclick="theme('paper')"></button></div></div>
<section class="panel on" data-panel="home"><div class="grid"><div class="card hero"><div class="head"><h2>Состояние</h2><small id="fw">—</small></div><p class="big" id="mainState">...</p><p class="sub" id="mainSub">жду данные</p></div><div class="card"><div class="head"><h2>Система</h2><small id="clock">—</small></div><div class="row"><span class="k">IP дома</span><span class="v" id="ip">—</span></div><div class="row"><span class="k">Собеседник</span><span class="v" id="spk">—</span></div><div class="row"><span class="k">PC Bridge</span><span class="v" id="pc">—</span></div><div class="row"><span class="k">AI модель</span><span class="v" id="aimodel">—</span></div></div></div><div class="stats" style="margin-top:10px"><div class="stat"><span>Температура</span><b id="t">—</b></div><div class="stat"><span>Влажность</span><b id="h">—</b></div><div class="stat"><span>Энергия</span><b id="en">—</b></div><div class="stat"><span>Батарея</span><b id="bat">—</b></div></div><div class="grid" style="margin-top:10px"><div class="card"><div class="head"><h2>Привычки</h2><small id="pet">—</small></div><div class="row"><span class="k">Голос</span><span class="v" id="hvoice">—</span></div><div class="row"><span class="k">Касания</span><span class="v" id="htouches">—</span></div><div class="row"><span class="k">Громкость</span><span class="v" id="hvolavg">—</span></div><div class="row"><span class="k">Микрофон / экран</span><span class="v" id="hpref">—</span></div></div><div class="card"><div class="head"><h2>Скорость</h2><small>STT / AI / TTS</small></div><div class="row"><span class="k">Распознавание</span><span class="v" id="sttms">—</span></div><div class="row"><span class="k">Мозг</span><span class="v" id="aims">—</span></div><div class="row"><span class="k">Голос</span><span class="v" id="ttsms">—</span></div><div class="row"><span class="k">Очередь команд</span><span class="v" id="pending">—</span></div></div></div></section>
<section class="panel" data-panel="ctrl"><div class="grid"><div class="card"><div class="head"><h2>Настройки</h2><small id="setnote">готово</small></div><div class="set"><label>Громкость <b id="volv">—</b></label><input type="range" id="vol" min="0" max="21" oninput="lv('volv',this.value)" onchange="setp('vol',this.value)"></div><div class="set"><label>Микрофон <b id="micv">—</b></label><input type="range" id="mic" min="0" max="100" oninput="lv('micv',this.value,'%')" onchange="setp('mic',this.value)"></div><div class="set"><label>Экран <b id="briv">—</b></label><input type="range" id="bri" min="5" max="100" oninput="lv('briv',this.value,'%')" onchange="setp('bri',this.value)"></div><div class="set"><label>Прозрачность лица <b id="eglowv">—</b></label><input type="range" id="eglow" min="5" max="100" oninput="lv('eglowv',this.value,'%')" onchange="setp('eglow',this.value)"></div><div class="set"><label>OLED помощника <b id="obriv">—</b></label><input type="range" id="obri" min="1" max="100" oninput="lv('obriv',this.value,'%')" onchange="setp('obri',this.value)"></div><div class="set"><label>Голос</label><select id="voice" onchange="setp('voice',this.value)"><option value="ru-RU-SvetlanaNeural">Светлана</option><option value="ru-RU-DmitryNeural">Дмитрий</option><option value="ru-RU-DariyaNeural">Дарья</option></select></div></div><div class="card"><div class="head"><h2>Быстрые</h2><small>удалённо</small></div><div class="tog"><span>Микрофон</span><div class="sw" id="micsw" onclick="tgMic()"><i></i></div></div><div class="tog"><span>Звуки в покое</span><div class="sw" id="chirp" onclick="tg('chirp','chirp')"><i></i></div></div><div class="tog"><span>Авто-диагностика</span><div class="sw" id="adiag" onclick="tg('adiag','adiag')"><i></i></div></div><div class="tog"><span>Реакция на касание</span><div class="sw" id="touchreact" onclick="tg('touchreact','touchreact')"><i></i></div></div><div class="btns" style="margin-top:12px"><button class="btn pri" onclick="setp('relay',1)">Реле ВКЛ</button><button class="btn bad" onclick="setp('relay',0)">Реле ВЫКЛ</button><button class="btn" onclick="quick('включи микрофон и слушай')">Слушай</button><button class="btn" onclick="quick('покажи статус')">Статус</button></div></div></div></section>
<section class="panel" data-panel="chat"><div class="card"><div class="head"><h2>Связь</h2><small>текст до робота</small></div><div class="chips"><button class="chip" onclick="quick('моргни одним глазом')">подмигни</button><button class="chip" onclick="quick('покажи сердечко')">сердце</button><button class="chip" onclick="quick('запусти музыку')">музыка</button><button class="chip" onclick="quick('останови музыку')">стоп</button><button class="chip" onclick="quick('покажи сети вай фай рядом')">WiFi</button></div><div id="thread" class="thread"><div class="note">Сообщения Simalee появятся здесь.</div></div><div class="composer"><input id="say" type="text" placeholder="Сообщение роботу..." onkeydown="if(event.key==='Enter')sendChat()"><button onclick="sendSay()" title="Сказать дома">▶</button><button onclick="sendChat()" title="В чат">↗</button></div></div></section>
<section class="panel" data-panel="push"><div class="grid"><div class="card"><div class="head"><h2>Уведомления</h2><small id="pushCount">—</small></div><p class="note" id="pushst">Проверка...</p><div class="diag" id="pushdiag">диагностика...</div><div class="btns" style="margin-top:10px"><button class="btn pri" onclick="enablePush()">Включить</button><button class="btn" onclick="testPush()">Тест</button></div><label class="note" style="display:block;margin-top:12px">Иконка уведомлений</label><input type="file" id="iconf" accept="image/*" onchange="uploadIcon()"></div><div class="card"><div class="head"><h2>Облако</h2><small>Render</small></div><div class="row"><span class="k">Подписок</span><span class="v" id="subs">—</span></div><div class="row"><span class="k">Ошибка</span><span class="v" id="pusherr">—</span></div><div class="row"><span class="k">Связь</span><span class="v" id="cloudage">—</span></div></div></div></section>
</div><nav class="tabs"><button class="on" data-tab="home" onclick="nav('home')">Статус</button><button data-tab="ctrl" onclick="nav('ctrl')">Пульт</button><button data-tab="chat" onclick="nav('chat')">Связь</button><button data-tab="push" onclick="nav('push')">Push</button></nav></div>
<script>
let KEY=localStorage.getItem('simkey')||'',LAST=0,seen={},touched=0,st={};
document.body.dataset.theme=localStorage.getItem('simTheme')||'';
function $(id){return document.getElementById(id)}function show(){$('login').style.display=KEY?'none':'grid';$('app').style.display=KEY?'block':'none'}function saveKey(){KEY=$('key').value.trim();localStorage.setItem('simkey',KEY);show();tick();poll();pushStatus();pushDiag()}function theme(v){document.body.dataset.theme=v;localStorage.setItem('simTheme',v);beep(620,.05)}
function nav(p){document.querySelectorAll('.panel').forEach(x=>x.classList.toggle('on',x.dataset.panel===p));document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('on',x.dataset.tab===p));beep(740,.05)}
function lv(id,v,suf=''){$(id).textContent=v+suf}function val(id,v){let e=$(id);if(e)e.textContent=v}function ms(v){v=+v||0;return v?v+' мс':'—'}function fmtUp(s){s=+s||0;let h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?h+'ч '+m+'м':m+'м'}
function setp(k,v){fetch('/set_remote?key='+encodeURIComponent(KEY)+'&'+k+'='+encodeURIComponent(v));$('setnote').textContent=k+' = '+v;beep(820,.04)}
function sv(id,lab,v,suf=''){if(v==null||v==='')return;let e=$(id);if(e&&e!==document.activeElement){e.value=v;val(lab,v+suf)}}function tgset(id,v){let e=$(id);if(e)e.classList.toggle('on',v==1||v==='1'||v===true||v==='true')}function tg(id,k){let on=!$(id).classList.contains('on');$(id).classList.toggle('on',on);setp(k,on?1:0)}function tgMic(){let on=!$('micsw').classList.contains('on');$('micsw').classList.toggle('on',on);setp('micoff',on?0:1)}
async function tick(){if(!KEY)return;try{let r=await fetch('/status_remote?key='+encodeURIComponent(KEY));if(r.status===403){localStorage.removeItem('simkey');KEY='';show();return}let d=await r.json();st=d;let on=d.age>=0&&d.age<15;$('online').innerHTML='<span class="dot '+(on?'ok':'')+'"></span><span>'+(on?'online':(d.age<0?'new':'offline'))+'</span>';val('mainState',on?(d.slp==='1'?'спит':'на связи'):'не на связи');val('mainSub',on?'последний пакет '+d.age+'с назад':(d.age<0?'робот ещё не присылал статус':'молчание '+d.age+'с'));val('subtitle',(d.speaker&&d.speaker!=='not recognized')?'говорит: '+d.speaker:'удалённая связь с роботом');val('t',d.t&&d.t!='-99'?d.t+' °C':'—');val('h',d.h&&d.h!='-99'?d.h+' %':'—');val('en',d.en!=null&&d.en!==''?d.en+' %':'—');val('bat',d.bat&&d.bat!='-1'?d.bat+' %':'от сети');val('ip',d.ip||'—');val('spk',(d.speaker&&d.speaker!=='not recognized')?d.speaker:'—');val('pc',d.pconline==='1'||d.pconline==='true'?'online':(d.pcon==='1'||d.pcon==='true'?'нет ответа':'выкл'));val('aimodel',d.aimodel||'—');val('fw',d.fw||'—');val('clock',new Date().toLocaleTimeString('ru',{hour:'2-digit',minute:'2-digit'}));val('pet',d.pet?d.pet+'/100':'—');val('hvoice',d.hvoice||0);val('htouches',d.htouches||0);val('hvolavg',(+d.hvolavg>=0)?d.hvolavg+' из 21':'—');val('hpref',((+d.hmicavg>=0)?d.hmicavg+'%':'—')+' / '+((+d.hbriavg>=0)?d.hbriavg+'%':'—'));val('sttms',ms(d.sttms));val('aims',ms(d.aims));val('ttsms',ms(d.ttsms));val('pending',d.pending||0);val('cloudage',d.age>=0?d.age+'с':'—');if(Date.now()-touched>3500){sv('vol','volv',d.vol);sv('mic','micv',d.mic,'%');sv('bri','briv',d.bri,'%');sv('eglow','eglowv',d.eglow,'%');sv('obri','obriv',d.obri,'%');if(d.voice&&document.activeElement!==$('voice'))$('voice').value=d.voice;tgset('micsw',d.micon);tgset('chirp',d.chirp);tgset('adiag',d.adiag);tgset('touchreact',d.touchreact)}}catch(e){}}
document.addEventListener('input',e=>{if(e.target.type==='range')touched=Date.now()},true);
function add(m){if(seen[m.id])return;seen[m.id]=1;let th=$('thread'),hint=th.querySelector('.note');if(hint)hint.remove();let d=document.createElement('div');d.className='msg '+(m.from==='you'?'you':'rob');d.textContent=m.text;let tm=document.createElement('span');tm.className='tm';tm.textContent=new Date((m.t||Date.now()/1000)*1000).toLocaleTimeString('ru',{hour:'2-digit',minute:'2-digit'});d.appendChild(tm);th.appendChild(d);th.scrollTop=th.scrollHeight}
async function poll(){if(!KEY)return;try{let d=await(await fetch('/outbox_remote?key='+encodeURIComponent(KEY)+'&since='+LAST)).json();(d.msgs||[]).forEach(add);LAST=d.last||LAST}catch(e){}}async function sendChat(){let x=$('say').value.trim();if(!x)return;$('say').value='';add({id:'l'+Date.now(),from:'you',text:x,t:Date.now()/1000});await fetch('/chat_remote?key='+encodeURIComponent(KEY)+'&text='+encodeURIComponent(x));poll()}async function sendSay(){let x=$('say').value.trim();if(!x)return;$('say').value='';add({id:'s'+Date.now(),from:'you',text:'дома: '+x,t:Date.now()/1000});await fetch('/say_remote?key='+encodeURIComponent(KEY)+'&text='+encodeURIComponent(x))}function quick(x){$('say').value=x;sendChat()}
if('serviceWorker'in navigator){navigator.serviceWorker.register('/sw.js').then(r=>r.update()).catch(e=>{val('pushdiag','SW: '+e)})}function urlB64(s){let p='='.repeat((4-s.length%4)%4),b=atob((s+p).replace(/-/g,'+').replace(/_/g,'/'));return Uint8Array.from([...b].map(c=>c.charCodeAt(0)))}
async function enablePush(){try{val('pushst','Проверяю...');if(!('serviceWorker'in navigator)){val('pushst','Нет serviceWorker');return}if(!('PushManager'in window)){val('pushst','Нет Web Push');return}let perm=await Notification.requestPermission();if(perm!=='granted'){val('pushst','Разрешение: '+perm);return}let reg=await Promise.race([navigator.serviceWorker.ready,new Promise((_,rej)=>setTimeout(()=>rej(new Error('service worker timeout')),8000))]);let pub=(await(await fetch('/vapid_public')).text()).trim();let sub=await reg.pushManager.getSubscription();if(!sub)sub=await reg.pushManager.subscribe({userVisibleOnly:true,applicationServerKey:urlB64(pub)});let r=await fetch('/push_subscribe?key='+encodeURIComponent(KEY),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(sub)});await pushStatus();val('pushst',r.ok?'Уведомления включены':'Подписка не сохранилась')}catch(e){val('pushst','Ошибка: '+(e.message||e))}}async function testPush(){try{let r=await(await fetch('/push?key='+encodeURIComponent(KEY)+'&title=Simalee&text='+encodeURIComponent('Тестовое уведомление'))).json();val('pushst',r.sent>0?'Отправлено: '+r.sent:(r.subs==0?'Нет подписки':'Не доставлено'));await pushStatus()}catch(e){val('pushst','Ошибка теста: '+e)}}async function uploadIcon(){let f=$('iconf').files[0];if(!f)return;try{let b=await f.arrayBuffer();let r=await fetch('/set_icon?key='+encodeURIComponent(KEY),{method:'POST',body:b});val('pushst',r.ok?'Иконка обновлена':'Не удалось загрузить')}catch(e){val('pushst','Ошибка иконки: '+e)}}async function pushStatus(){if(!KEY)return;try{let c=await(await fetch('/push_count?key='+encodeURIComponent(KEY))).json();val('subs',c.subs||0);val('pushCount',(c.subs||0)+' устр.');val('pusherr',c.err||'—');if(c.subs>0)val('pushst','Подписано устройств: '+c.subs)}catch(e){}}function pushDiag(){try{val('pushdiag','SW '+(('serviceWorker'in navigator)?'ok':'no')+' · Push '+(('PushManager'in window)?'ok':'no')+' · '+(window.Notification?Notification.permission:'нет'))}catch(e){}}
let _ac;function beep(f=720,d=.05){try{_ac=_ac||new(window.AudioContext||window.webkitAudioContext)();let o=_ac.createOscillator(),g=_ac.createGain();o.type='sine';o.frequency.value=f;g.gain.value=.045;o.connect(g);g.connect(_ac.destination);let t=_ac.currentTime;o.start(t);g.gain.exponentialRampToValueAtTime(.0001,t+d);o.stop(t+d+.02)}catch(e){}}document.addEventListener('click',e=>{if(e.target.closest('button,.sw'))beep()},true);
show();tick();poll();if(KEY){pushStatus();pushDiag()}setInterval(tick,3000);setInterval(poll,2500);
</script></body></html>"""

@app.get("/app", response_class=HTMLResponse)
def app_page():
    return APP_HTML_JP


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
