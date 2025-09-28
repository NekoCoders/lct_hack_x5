import asyncio
import logging
from typing import Any

from fastapi import HTTPException

from server.interface import Entity
from server.model_runner import infer_model, load_model

log = logging.getLogger(__name__)


def predict_entities(text: str) -> list[Entity]:
    """
    Простая демонстрационная логика распознавания сущностей
    В реальном приложении здесь будет ваша ML модель
    """

    model_out = infer_model(text)
    try:
        entities = [
            Entity(start_index=ent[0], end_index=ent[1], entity=ent[2])
            for ent in model_out
        ]
    except Exception as e:
        log.exception("Error while unpacking predictions:")
    return entities


class InferenceQueue:
    def __init__(self, maxsize: int = 100, request_timeout_s: float = 60.0):
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self.request_timeout_s = request_timeout_s
        self.worker_task: asyncio.Task | None = None
        self._last_qsize = 0
        self._stopping = asyncio.Event()

    async def submit(self, text: str) -> list[Entity]:
        """
        Кладём задачу в очередь и ждём будущий результат (с таймаутом).
        Если очередь переполнена — 503.
        """
        if self.queue.full():
            raise HTTPException(status_code=503, detail="Queue is full, try later")

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        job = {"text": text, "future": fut}

        await self.queue.put(job)
        self._log_qsize_if_changed()

        try:
            # Ждём результат от воркера
            return await asyncio.wait_for(fut, timeout=self.request_timeout_s)
        except asyncio.TimeoutError:
            # Сигнализируем воркеру, что можно проигнорировать (по идее, он сам доставит fut)
            if not fut.done():
                fut.set_exception(TimeoutError("Inference timed out"))
            raise HTTPException(status_code=504, detail="Inference timed out")

    async def _worker(self):
        """
        Единственный воркер: вынимает задачи по одной и синхронно гоняет модель.
        Сам инференс — синхронный; выносим в executor, чтобы не блокировать event loop.
        """
        load_model()  # прогреваем модель в воркере
        loop = asyncio.get_running_loop()

        while not self._stopping.is_set():
            try:
                job = await self.queue.get()
                self._log_qsize_if_changed()
            except asyncio.CancelledError:
                break

            text = job["text"]
            fut: asyncio.Future = job["future"]

            try:
                # Выполняем синхронный инференс в ThreadPool, чтобы не блокировать loop
                entities: list[Entity] = await loop.run_in_executor(None, predict_entities, text)
                if not fut.done():
                    fut.set_result(entities)
            except Exception as e:
                log.exception("Inference failed")
                if not fut.done():
                    fut.set_exception(e)
            finally:
                self.queue.task_done()

    async def start(self):
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._worker(), name="inference-worker")
            log.info("Inference worker started")

    async def stop(self):
        self._stopping.set()
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            log.info("Inference worker stopped")

    def _log_qsize_if_changed(self):
        qsize = self.queue.qsize()
        if qsize != self._last_qsize:
            log.info(f"Queue size changed: {self._last_qsize} -> {qsize}")
            self._last_qsize = qsize
