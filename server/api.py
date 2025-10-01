import logging
from pathlib import Path
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

# from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ------------ Добавляем в sys.path:
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -------------
from server.configure_logger import configure_file_rotating_logger
configure_file_rotating_logger()  # For logging long imports
from server.inference_queue import InferenceQueue
from server.interface import PredictionRequest, Entity

app = FastAPI(title="Neko Coders NER Prediction API", version="1.0.0", debug=True)
inq = InferenceQueue(maxsize=200, request_timeout_s=60.0)

# Монтируем статические файлы (для CSS/JS если нужно)
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

log = logging.getLogger("api")


@app.post("/api/predict", response_model=list[Entity])
async def predict_entities_endpoint(request: PredictionRequest):
    """
    Предсказание именованных сущностей в тексте
    """
    try:
        log.info(f"POST api/predict body='{request.model_dump_json()}'")
        start_time = time.perf_counter()  # TODO: move to decorator
        if not request.input or not request.input.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty")

        # Ограничение длины текста для безопасности
        if len(request.input) > 1000:
            raise HTTPException(status_code=400, detail="Input text too long")
        response = await inq.submit(request.input)

        duration = time.perf_counter() - start_time
        log.info(f"/api/predict executed in {duration:.3f}s")
        return response

    except HTTPException:
        # пробрасываем наши осмысленные ошибки дальше
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Главная страница с тестовым интерфейсом
    """
    with open(Path(__file__).parent / "index.html", "rt") as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/test")
async def test_endpoint():
    """
    Тестовый эндпоинт с примером работы
    """
    start_time = time.perf_counter()  # TODO: move to decorator
    test_text = "Я купил свежее молоко Простоквашино"
    try:
        predictions = await inq.submit(test_text)
    except HTTPException:
        # пробрасываем наши осмысленные ошибки дальше
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    duration = time.perf_counter() - start_time
    log.info(f"/test executed in {duration:.3f}s")
    return {
        "test_text": test_text,
        "predictions": predictions,
    }

# lifespan — корректный запуск/остановка воркера
@app.on_event("startup")
async def _startup():
    await inq.start()

@app.on_event("shutdown")
async def _shutdown():
    await inq.stop()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
