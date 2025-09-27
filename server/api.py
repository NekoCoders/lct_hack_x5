from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional
import re

from model_runner import infer_model

app = FastAPI(title="NER Prediction API", version="1.0.0", debug=True)

# Монтируем статические файлы (для CSS/JS если нужно)
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Модели данных
class PredictionRequest(BaseModel):
    input: str

class Entity(BaseModel):
    start_index: int
    end_index: int
    entity: str

# Простая логика для демонстрации (замените на вашу реальную модель)
def predict_entities(text: str) -> List[Entity]:
    """
    Простая демонстрационная логика распознавания сущностей
    В реальном приложении здесь будет ваша ML модель
    """

    model_out = infer_model(text)
    try:
        entities = [Entity(start_index=ent["start"], end_index=ent["end"], entity=ent["entity"]) for ent in model_out]
    except Exception as e:
        print(e)

    # entities = []
    
    # # Простые правила для демонстрации
    # patterns = {
    #     'B-TYPE': [r'\bсгущенное\b', r'\bсвежее\b', r'\bпастеризованное\b'],
    #     'I-TYPE': [r'\bмолоко\b', r'\bмясо\b', r'\bмасло\b'],
    #     'B-BRAND': [r'\bпростоквашино\b', r'\bдомик в деревне\b', r'\bчудо\b'],
    # }
    
    # text_lower = text.lower()
    
    # for entity_type, regex_list in patterns.items():
    #     for pattern in regex_list:
    #         for match in re.finditer(pattern, text_lower):
    #             entities.append(Entity(
    #                 start_index=match.start(),
    #                 end_index=match.end(),
    #                 entity=entity_type
    #             ))
    
    # Сортировка по начальному индексу
    entities.sort(key=lambda x: x.start_index)
    
    return entities

@app.post("/api/predict", response_model=List[Entity])
async def predict_entities_endpoint(request: PredictionRequest):
    """
    Предсказание именованных сущностей в тексте
    """
    try:
        if not request.input or not request.input.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty")
        
        # Ограничение длины текста для безопасности
        if len(request.input) > 1000:
            raise HTTPException(status_code=400, detail="Input text too long")
        
        entities = predict_entities(request.input)
        
        return entities
        
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
    test_text = "Я купил свежее молоко Простоквашино"
    entities = predict_entities(test_text)
    
    return {
        "test_text": test_text,
        "predictions": entities,
        "message": "Тестовый запрос выполнен успешно"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)