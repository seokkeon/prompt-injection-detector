import sys
import os
from loguru import logger


def setup_logger(log_level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
    )
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/detector.log", rotation="10 MB", retention="7 days", level="DEBUG")


setup_logger(os.getenv("LOG_LEVEL", "INFO"))
