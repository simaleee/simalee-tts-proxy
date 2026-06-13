# Simalee TTS Proxy

Прокси-сервер на FastAPI, который превращает текст в речь голосом **Microsoft Edge Neural** (Светлана / Дмитрий, русский) и отдаёт **MP3-стрим**. Сделан под робота **Simalee_** на ESP32-S3 — обходит ограничение прямой потоковой загрузки 32 КБ через TLS на ESP32, потому что:

- ESP32 ходит на **обычный HTTP**, а не HTTPS.
- MP3 — в 5–10 раз компактнее WAV → весь короткий ответ помещается одним куском.

## Сервис защищён паролем

GET-параметр `key=Simalee00221922` обязателен. Без него прокси возвращает `403`.

## Деплой на Render (бесплатно)

1. Зайди на [render.com](https://render.com), залогинься через **GitHub**.
2. **New** → **Blueprint** → выбери репозиторий **`simaleee/simalee-tts-proxy`**.
3. Render прочитает `render.yaml` и сам всё развернёт.
4. Получишь URL вида `https://simalee-tts-proxy.onrender.com`. Это и есть твой TTS-сервер.

> Бесплатный тариф Render «засыпает» через 15 мин простоя — первый запрос после паузы будет ~30 сек. Дальше — мгновенно.

## API

```
GET /                                      → health check
GET /voices                                → список русских голосов
GET /tts?key=PASSWORD&text=привет          → audio/mpeg (MP3 24 кГц mono)
GET /tts?key=PASSWORD&text=...&voice=ru-RU-DmitryNeural   → мужской голос
```

Опционально: `rate=+0%` (скорость), `pitch=+0Hz` (тон).

## Локальный запуск

```bash
pip install -r requirements.txt
TTS_PASSWORD=Simalee00221922 uvicorn app:app --host 0.0.0.0 --port 8000
```

Тест: `http://localhost:8000/tts?key=Simalee00221922&text=привет`

## Лицензия

MIT. Использует [edge-tts](https://github.com/rany2/edge-tts).
