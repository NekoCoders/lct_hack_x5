import logging
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
    print(PROJECT_ROOT)
# -------------
from server.configure_logger import configure_file_rotating_logger
from server.inference_queue import InferenceQueue
from server.interface import PredictionRequest, Entity

app = FastAPI(title="Neko Coders NER Prediction API", version="1.0.0", debug=True)
inq = InferenceQueue(maxsize=200, request_timeout_s=60.0)

# Монтируем статические файлы (для CSS/JS если нужно)
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

configure_file_rotating_logger()
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
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NER Prediction API Test</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            textarea { width: 100%; height: 100px; margin: 10px 0; padding: 10px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
            .entity { display: inline-block; margin: 2px; padding: 2px 5px; border-radius: 3px; }
            .B-TYPE { background: #ffcccc; }
            .I-TYPE { background: #ffdddd; }
            .B-BRAND { background: #ccccff; }
            .I-BRAND { background: #ddddff; }
            .B-VOLUME { background: #ccffcc; }
            .I-VOLUME { background: #ddffdd; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 NER Prediction API Test</h1>
            <p>Введите текст для распознавания сущностей:</p>
            
            <textarea id="textInput" placeholder="Введите текст здесь...">Я купил свежее молоко Простоквашино и пастеризованное масло Чудо.</textarea>
            
            <button onclick="predictEntities()">🔍 Распознать сущности</button>
            
            <div id="result" class="result" style="display:none;">
                <h3>Результат:</h3>
                <div id="highlightedText"></div>
                <pre id="jsonResult"></pre>
            </div>

            <div style="margin-top: 30px;">
                <h3>Примеры для тестирования:</h3>
                <button onclick="loadExample(1)">Пример 1</button>
                <button onclick="loadExample(2)">Пример 2</button>
                <button onclick="loadExample(3)">Пример 3</button>
            </div>

            <script>
                const examples = {
                    1: "Я купил свежее молоко Простоквашино и сгущенное масло.",
                    2: "В магазине было пастеризованное мясо от Домик в деревне и чудо молоко.",
                    3: "Сгущенное молоко и свежее мясо - лучшие продукты."
                };

                function loadExample(num) {
                    document.getElementById('textInput').value = examples[num];
                }

                async function predictEntities() {
                    const text = document.getElementById('textInput').value;
                    const resultDiv = document.getElementById('result');
                    const highlightedDiv = document.getElementById('highlightedText');
                    const jsonPre = document.getElementById('jsonResult');

                    try {
                        const response = await fetch('/api/predict', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ input: text })
                        });

                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }

                        const data = await response.json();
                        
                        // Показываем JSON результат
                        jsonPre.textContent = JSON.stringify(data, null, 2);
                        
                        // Показываем текст с подсветкой сущностей
                        let highlightedText = text;
                        const entities = data.sort((a, b) => b.start_index - a.start_index);
                        
                        entities.forEach(entity => {
                            const entityText = text.substring(entity.start_index, entity.end_index);
                            const highlighted = `<span class="entity ${entity.entity}" title="${entity.entity}">${entityText}</span>`;
                            highlightedText = highlightedText.substring(0, entity.start_index) + 
                                            highlighted + 
                                            highlightedText.substring(entity.end_index);
                        });
                        
                        highlightedDiv.innerHTML = '<strong>Текст с сущностями:</strong><br>' + highlightedText;
                        resultDiv.style.display = 'block';

                    } catch (error) {
                        jsonPre.textContent = 'Ошибка: ' + error.message;
                        resultDiv.style.display = 'block';
                    }
                }

                // Автоматически запускаем при загрузке страницы
                window.onload = function() {
                    predictEntities();
                };
            </script>
        </div>
    </body>
    </html>
    """
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
