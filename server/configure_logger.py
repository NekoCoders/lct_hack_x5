import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent / "logs"

def configure_file_rotating_logger():
    LOGS_DIR.mkdir(exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        str(LOGS_DIR / "app.log"),  # base file name
        when="H",  # hours
        interval=1,
        backupCount=96,
        encoding="utf-8"
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
