import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "extra": "ignore"}

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "huggingface.co/QuantFactory/finance-Llama3-8B-GGUF:Q5_K_M"

    newsapi_key: str = ""

    screener_email: str = ""
    screener_password: str = ""

    max_portfolio_var_pct: float = 5.0
    kelly_fraction: float = 0.25

    data_dir: Path = BASE_DIR / "data"
    cache_dir: Path = BASE_DIR / "data" / "cache"
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"

    max_tokens: int = 2048
    temperature: float = 0.3
    top_p: float = 0.9

    news_buffer_days: int = 5

    def ensure_dirs(self):
        for d in [self.data_dir, self.cache_dir, self.raw_data_dir, self.processed_data_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
