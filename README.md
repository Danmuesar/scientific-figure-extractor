# Scientific Figure Extractor

Upload a research PDF and extract figure images with their associated captions.

## Stack

- FastAPI backend
- Single-page HTML frontend served by FastAPI
- PyMuPDF for PDF parsing and cropping

## Local run

```powershell
python -m pip install -r requirements.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Then open `http://127.0.0.1:8000`.

## Render deploy

This repo includes `render.yaml`, so Render can deploy it as a Python web service.

Start command:

```powershell
uvicorn api:app --host 0.0.0.0 --port $PORT
```
