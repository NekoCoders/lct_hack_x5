
# Модели данных
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    input: str


class Entity(BaseModel):
    start_index: int
    end_index: int
    entity: str
