import asyncio
import logging
from typing import Any

from fastapi import HTTPException

from server.interface import Entity
from server.model_runner import infer_model, load_model

log = logging.getLogger(__name__)


class InferenceQueue:
    def __init__(
        self,
        maxsize: int = 100,
        request_timeout_s: float = 60.0,  # TODO: improve timeout and move to config?
        batch_size: int = 1,  # TURN OFF
        batch_wait_ms: int = 20,  # окно добора задач в батч, миллисекунды
        joiner: str = ". ",
    ):
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self.request_timeout_s = request_timeout_s
        self.worker_task: asyncio.Task | None = None
        self._last_qsize = 0
        self._stopping = asyncio.Event()
        self.batch_size = batch_size
        self.batch_wait_ms = batch_wait_ms
        self.joiner = joiner

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
            return await asyncio.wait_for(fut, timeout=self.request_timeout_s)
        except asyncio.TimeoutError:
            # Сигнализируем воркеру, что можно проигнорировать (по идее, он сам доставит fut)
            if not fut.done():
                fut.set_exception(TimeoutError("Inference timed out"))
            raise HTTPException(status_code=504, detail="Inference timed out")

    async def _worker(self):
        """
        Один воркер, но с микробатчингом: за короткое окно добираем до batch_size задач,
        склеиваем тексты через joiner и делаем один прогон модели.
        """
        load_model()  # прогреваем модель внутри воркера
        loop = asyncio.get_running_loop()

        while not self._stopping.is_set():
            try:
                # берём первую задачу (ждём, если пусто)
                first_job = await self.queue.get()
                self._log_qsize_if_changed()
            except asyncio.CancelledError:
                break

            # пытаемся добрать ещё задачи в батч
            jobs = [first_job]
            if self.batch_size > 1:
                end_time = loop.time() + (self.batch_wait_ms / 1000.0)
                # пока есть время окна и не заполнен батч — добираем без блокировки
                while len(jobs) < self.batch_size and loop.time() < end_time:
                    try:
                        # чуть-чуть ждём вторую/третью задачу в рамках окна
                        timeout_left = max(0.0, end_time - loop.time())
                        job = await asyncio.wait_for(self.queue.get(), timeout=timeout_left)
                        jobs.append(job)
                        self._log_qsize_if_changed()
                    except (asyncio.TimeoutError, asyncio.QueueEmpty):
                        break
            if len(jobs) > 1:
                log.info("%d requests will be processed in batch", len(jobs))
            # подготавливаем общий текст и смещения сегментов
            texts = [j["text"] for j in jobs]
            joiner = self.joiner
            # стартовые индексы каждого сегмента в общий_текст
            segment_starts = []
            offset = 0
            for t in texts:
                segment_starts.append(offset)
                offset += len(t) + len(joiner)
            # общий текст: последний joiner можно оставить — мы просто будем
            # отфильтровывать сущности по границам сегмента
            combined_text = joiner.join(texts)

            try:
                # один прогон синхронной модели в ThreadPool
                raw_entities = await loop.run_in_executor(None, infer_model, combined_text)

                # раскладываем сущности обратно по задачам
                results_per_job: list[list[Entity]] = [[] for _ in jobs]
                for seg_idx, (seg_start) in enumerate(segment_starts):
                    # границы сегмента в общем тексте
                    seg_end = seg_start + len(texts[seg_idx])

                    # берём только те сущности, которые лежат ПОЛНОСТЬЮ внутри сегмента
                    seg_entities = []
                    for start, end, label in raw_entities:
                        if start >= seg_start and end <= seg_end:
                            # корректируем смещение к локальному тексту
                            local_start = start - seg_start
                            local_end = end - seg_start
                            seg_entities.append(
                                Entity(start_index=local_start, end_index=local_end, entity=label)
                            )
                    results_per_job[seg_idx] = seg_entities

                # отдаём результаты во fiture
                for (job, ents) in zip(jobs, results_per_job):
                    fut: asyncio.Future = job["future"]
                    if not fut.done():
                        fut.set_result(ents)

            except Exception as e:
                log.exception("Inference failed (batch of %d)", len(jobs))
                for job in jobs:
                    fut: asyncio.Future = job["future"]
                    if not fut.done():
                        fut.set_exception(e)
            finally:
                # помечаем ВСЕ взятые задачи как завершённые
                for _ in jobs:
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
