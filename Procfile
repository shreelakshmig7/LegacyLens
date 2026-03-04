# LegacyLens — Railway / Heroku process file
# Bind to 0.0.0.0 and use PORT so the proxy can reach the app (avoids 502 Bad Gateway).
web: sh -c 'uvicorn legacylens.api.main:app --host 0.0.0.0 --port ${PORT:-8000}'
