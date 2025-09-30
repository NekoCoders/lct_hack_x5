## Структура модулей:
- data - исходные данные
- model - подготовка модели и само распознавание сущностей
- validation - тестирование распознавания. Использует model
- server - API бекенд, использующий модуль model

## Запуск сервера
```
pip install -r requirements.txt
python server/api.py
```
### Запуск через docker
```
docker compose up
```