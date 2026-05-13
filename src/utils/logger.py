import sys
import os
from loguru import logger


def setup_logger(log_level: str = "INFO") -> None:
    """Configure loguru logger."""
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
    )
    logger.add(
        "logs/detector.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )


# Initialize on import
os.makedirs("logs", exist_ok=True)
setup_logger(os.getenv("LOG_LEVEL", "INFO"))
