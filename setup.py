from setuptools import setup, find_packages

setup(
    name="prompt-injection-detector",
    version="1.0.0",
    description="Detect hidden prompt injections in images and emails",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pillow",
        "opencv-python",
        "numpy",
        "scipy",
        "pytesseract",
        "beautifulsoup4",
        "lxml",
        "fastapi",
        "uvicorn",
        "pydantic",
        "python-multipart",
        "python-dotenv",
        "loguru",
        "rich",
    ],
    extras_require={
        "ml": [
            "torch",
            "transformers",
            "sentence-transformers",
            "scikit-learn",
            "datasets",
        ],
        "ocr": ["easyocr"],
        "dev": ["pytest", "pytest-cov", "pytest-asyncio", "httpx"],
    },
)
