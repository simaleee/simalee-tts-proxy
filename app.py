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
