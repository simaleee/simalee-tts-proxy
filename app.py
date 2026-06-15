"""
Simalee TTS proxy — Edge TTS (Microsoft) Russian neural voice for ESP32 robots.
ESP32 hits a plain HTTP endpoint with text; we ask Microsoft Edge TTS for MP3
and stream it straight to the device. No API key on Microsoft's side, no
billing — Edge's public voice endpoint is free.

A shared password is required so a leaked URL doesn't burn quota.
"""

import os
import html
import xml.etree.ElementTree as ET
from urllib.parse import quote
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, PlainTextResponse
import edge_tts
import httpx

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
    text: str = Query(..., min_length=1, max_length=600),
    key: str = Query(...),
    voice: str = Query(DEFAULT_VOICE),
    rate: str = Query("+0%"),
    pitch: str = Query("+0Hz"),
):
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")

    if not voice.startswith("ru-RU-"):
        voice = DEFAULT_VOICE   # force a Russian voice to keep robot consistent

    async def stream():
        comm = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    return StreamingResponse(stream(), media_type="audio/mpeg")


# ---- internet info for the robot (the ESP can't reach some hosts directly; we fetch server-side) ----
WMO = {0: "ясно", 1: "облачно", 2: "облачно", 3: "пасмурно", 45: "туман", 48: "туман",
       51: "морось", 53: "морось", 55: "морось", 56: "морось", 57: "морось",
       61: "дождь", 63: "дождь", 65: "сильный дождь", 66: "дождь", 67: "дождь",
       71: "снег", 73: "снег", 75: "сильный снег", 77: "снег", 85: "снег", 86: "снег",
       80: "ливень", 81: "ливень", 82: "сильный ливень", 95: "гроза", 96: "гроза", 99: "гроза"}


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


@app.get("/news", response_class=PlainTextResponse)
async def news(key: str = Query(...), q: str = Query("", max_length=140), n: int = Query(2)):
    if key != PASSWORD:
        raise HTTPException(status_code=403, detail="bad key")
    n = max(1, min(n, 4))
    if q.strip():
        url = f"https://news.google.com/rss/search?q={quote(q)}&hl=ru&gl=RU&ceid=RU:ru"
    else:
        url = "https://news.google.com/rss/headlines/section/geo/Russia?hl=ru&gl=RU&ceid=RU:ru"
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True,
                                     headers={"User-Agent": "Mozilla/5.0"}) as c:
            root = ET.fromstring((await c.get(url)).text)
    except Exception:
        return "ERR новости недоступны"
    heads = []
    for it in root.findall(".//item")[:n]:
        title = html.unescape(it.findtext("title") or "").strip()
        if " - " in title:
            title = title.rsplit(" - ", 1)[0].strip()
        if title:
            heads.append(title)
    return (". ".join(heads) + ".") if heads else "Новостей не нашлось."
