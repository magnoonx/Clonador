import os
from pathlib import Path

# Directorio base del backend
BACKEND_DIR = Path(__file__).resolve().parent.parent

# Directorios de datos y modelos locales dentro de la carpeta del proyecto (Portabilidad USB)
MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
CLIPS_DIR = DATA_DIR / "clips"

# Crear directorios si no existen
for directory in [MODELS_DIR, DATA_DIR, SAMPLES_DIR, CLIPS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SETTINGS_FILE = DATA_DIR / "settings.json"

# Forzar a HuggingFace, PyTorch y TTS a guardar caché y modelos EXCLUSIVAMENTE en el proyecto
os.environ["HF_HOME"] = str(MODELS_DIR / "huggingface")
os.environ["TTS_HOME"] = str(MODELS_DIR / "tts")
os.environ["TORCH_HOME"] = str(MODELS_DIR / "torch")
os.environ["XDG_CACHE_HOME"] = str(MODELS_DIR / "cache")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

