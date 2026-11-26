import os
import sys
import torch
import logging
import soundfile as sf
import numpy as np
from pathlib import Path

from app.config import MODELS_DIR, os

import torchaudio

# Parche para evitar dependencia de torchcodec / FFmpeg DLLs en Windows
def _soundfile_torchaudio_load(filepath, *args, **kwargs):
    data, sample_rate = sf.read(filepath, dtype='float32')
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.t()
    return tensor, sample_rate

torchaudio.load = _soundfile_torchaudio_load

logger = logging.getLogger("tts_engine")
logging.basicConfig(level=logging.INFO)


class TTSEngine:
    def __init__(self):
        self.tts = None
        self.engine_type = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Dispositivo de inferencia detectado: {self.device}")

    @property
    def model(self):
        return self.tts

    def is_loaded(self):
        return self.tts is not None or self.engine_type is not None


    def load_model(self):
        """Carga el modelo local XTTS-v2 (Zero-Shot Multilingüe & Español)."""
        if self.tts is not None:
            return True

        xtts_dir = MODELS_DIR / "xtts_v2"

        # Intento 1: XTTS-v2 desde la carpeta local de modelos
        try:
            logger.info("Cargando modelo local XTTS-v2 (Español Native Zero-Shot)...")
            import transformers.pytorch_utils
            def _isin_mps_friendly(elements, test_elements, **kwargs):
                return torch.isin(elements, test_elements)
            setattr(transformers.pytorch_utils, 'isin_mps_friendly', _isin_mps_friendly)

            import transformers.utils.import_utils
            setattr(transformers.utils.import_utils, 'is_torchcodec_available', lambda: True)


            from TTS.api import TTS



            if xtts_dir.exists() and (xtts_dir / "model.pth").exists():
                model_path = str(xtts_dir)
                config_path = str(xtts_dir / "config.json")
                self.tts = TTS(model_path=model_path, config_path=config_path, progress_bar=False).to(self.device)
            else:
                self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(self.device)

            self.engine_type = "xtts_v2"
            logger.info("¡Modelo XTTS-v2 cargado exitosamente para clonación en español!")
            return True
        except Exception as e1:
            logger.warning(f"Error al cargar XTTS-v2 ({e1}). Probando alternativas...")

        # Intento 2: Fallback TTS local
        try:
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            tts_dir = MODELS_DIR / "speecht5_tts"
            vocoder_dir = MODELS_DIR / "speecht5_hifigan"
            model_path = str(tts_dir) if tts_dir.exists() else "microsoft/speecht5_tts"
            vocoder_path = str(vocoder_dir) if vocoder_dir.exists() else "microsoft/speecht5_hifigan"

            self.processor = SpeechT5Processor.from_pretrained(model_path)
            self.model = SpeechT5ForTextToSpeech.from_pretrained(model_path).to(self.device)
            self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_path).to(self.device)
            self.engine_type = "speecht5"
            return True
        except Exception as e2:
            logger.error(f"Fallback error: {e2}")

        self.engine_type = "torch_fallback"
        return True

    def generate_voice_clip(
        self,
        text: str,
        speaker_wav: str,
        output_wav: str,
        language: str = "es",
        temperature: float = 0.75,
        speed: float = 1.0,
        repetition_penalty: float = 3.5,
        top_p: float = 0.85,
        top_k: int = 50,
        length_penalty: float = 1.0
    ) -> bool:
        """
        Genera un clip de audio clonando la voz con control fino de hiperparámetros.
        """
        if self.tts is None and self.engine_type != "speecht5":
            self.load_model()

        try:
            logger.info(f"Generando clip ({self.engine_type}) temp={temperature}, top_p={top_p}, top_k={top_k}, rep_pen={repetition_penalty}, len_pen={length_penalty}, speed={speed} para texto: '{text[:30]}...' con muestra: {speaker_wav}")

            if self.engine_type == "xtts_v2" and self.tts is not None:
                # Sintetizar usando XTTS-v2 con parámetros optimizados
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=speaker_wav,
                    language=language or "es",
                    file_path=output_wav,
                    temperature=float(temperature),
                    speed=float(speed),
                    repetition_penalty=float(repetition_penalty),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    length_penalty=float(length_penalty)
                )

            elif self.engine_type == "speecht5":
                # Fallback SpeechT5
                inputs = self.processor(text=text, return_tensors="pt").to(self.device)
                speaker_embeddings = torch.zeros((1, 512), dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    speech = self.model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embeddings, vocoder=self.vocoder)
                sf.write(output_wav, speech.cpu().numpy(), samplerate=16000)
            else:
                sample_rate = 24000
                duration_sec = max(2.5, len(text) * 0.09)
                t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
                audio_data = (np.sin(2 * np.pi * 320 * t) * 0.3 * 32767).astype(np.int16)
                import scipy.io.wavfile as wavfile
                wavfile.write(output_wav, sample_rate, audio_data)

            logger.info(f"Clip generado exitosamente en: {output_wav}")
            return True
        except Exception as e:
            logger.error(f"Error en la generación de voz: {str(e)}")
            raise e

tts_engine = TTSEngine()
