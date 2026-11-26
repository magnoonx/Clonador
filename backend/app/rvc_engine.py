import os
import sys
import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from app.config import MODELS_DIR

logger = logging.getLogger("rvc_engine")

RVC_MODELS_DIR = MODELS_DIR / "rvc"
RVC_MODELS_DIR.mkdir(parents=True, exist_ok=True)

class RVCEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hubert = None
        self.rmvpe = None
        self.current_model_name = None
        self.current_model_mtime = 0
        self.net_g = None
        self.sr = 40000

    def list_models(self) -> List[str]:
        """Lista los modelos RVC disponibles (.pth) ignorando los modelos base."""
        if not RVC_MODELS_DIR.exists():
            return []
        ignored = {"rmvpe.pt", "contentvec_base.pt", "hubert_base.pt"}
        models = []
        for p in RVC_MODELS_DIR.glob("*.pth"):
            if p.name.lower() not in ignored:
                models.append(p.stem)
        return sorted(models)

    def load_base_models(self):
        """Carga en memoria ContentVec y RMVPE si no están cargados."""
        if self.hubert is None:
            logger.info("Cargando ContentVec base...")
            from fairseq import checkpoint_utils
            import fairseq.tasks.hubert_pretraining
            import fairseq.models.hubert.hubert

            contentvec_path = RVC_MODELS_DIR / "contentvec_base.pt"
            if not contentvec_path.exists():
                raise FileNotFoundError(f"No se encontró {contentvec_path}")

            models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([str(contentvec_path)])
            self.hubert = models[0].to(self.device).eval()
            logger.info("ContentVec cargado exitosamente.")

        if self.rmvpe is None:
            logger.info("Cargando RMVPE...")
            from rvc_python.lib.rmvpe import RMVPE

            rmvpe_path = RVC_MODELS_DIR / "rmvpe.pt"
            if not rmvpe_path.exists():
                raise FileNotFoundError(f"No se encontró {rmvpe_path}")

            self.rmvpe = RMVPE(str(rmvpe_path), is_half=(self.device == "cuda"), device=self.device)
            logger.info("RMVPE cargado exitosamente.")

    def load_voice_model(self, model_name: str):
        """Carga los pesos de un modelo de voz RVC específico."""
        model_path = RVC_MODELS_DIR / f"{model_name}.pth"
        if not model_path.exists():
            # Buscar sin extensión si ya la incluye
            model_path = RVC_MODELS_DIR / model_name
            if not model_path.exists():
                raise FileNotFoundError(f"Modelo RVC '{model_name}' no encontrado en {RVC_MODELS_DIR}")

        mtime = model_path.stat().st_mtime
        if self.current_model_name == model_name and self.current_model_mtime == mtime and self.net_g is not None:
            return

        logger.info(f"Cargando modelo RVC: {model_path.name}...")
        ckpt = torch.load(str(model_path), map_location=self.device, weights_only=False)
        c = ckpt["config"]
        self.sr = ckpt.get("sr", 40000)

        from rvc_python.lib.infer_pack.models import SynthesizerTrnMs768NSFsid
        self.net_g = SynthesizerTrnMs768NSFsid(*c, is_half=(self.device == "cuda"))
        self.net_g.load_state_dict(ckpt["weight"], strict=False)
        self.net_g.to(self.device).eval()
        self.current_model_name = model_name
        self.current_model_mtime = mtime
        logger.info(f"Modelo RVC '{model_name}' listo para inferencia.")

    def convert_voice(
        self,
        input_audio_path: str,
        output_audio_path: str,
        model_name: str = "david",
        pitch_shift: int = 0
    ) -> bool:
        """
        Convierte la voz de un archivo de audio aplicando el modelo RVC especificado.
        """
        try:
            self.load_base_models()
            self.load_voice_model(model_name)

            logger.info(f"Aplicando RVC ({model_name}) con pitch_shift={pitch_shift} a {input_audio_path}...")
            data, sr = sf.read(input_audio_path)
            if data.ndim == 2:
                data = data.mean(axis=1)

            # Resample a 16kHz para extracción acústica
            if sr != 16000:
                import scipy.signal
                data_16k = scipy.signal.resample_poly(data, 16000, sr)
            else:
                data_16k = data

            # Extraer F0 con RMVPE
            f0 = self.rmvpe.infer_from_audio(data_16k, thred=0.03)

            # Ajuste de tono si pitch_shift != 0
            if pitch_shift != 0:
                f0 = f0 * pow(2, pitch_shift / 12.0)

            # Convertir F0 a escala mel cuantizada
            f0_mel_min = 1127 * np.log(1 + 50 / 700)
            f0_mel_max = 1127 * np.log(1 + 1100 / 700)
            f0_mel = 1127 * np.log(1 + f0 / 700)
            f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
            f0_mel[f0_mel <= 1] = 1
            f0_mel[f0_mel > 255] = 255
            f0_coarse = np.rint(f0_mel).astype(np.int64)

            # Extraer características ContentVec (layer 12)
            feats = torch.from_numpy(data_16k).float().view(1, -1).to(self.device)
            padding_mask = torch.BoolTensor(feats.shape).fill_(False).to(self.device)
            inputs = {"source": feats, "padding_mask": padding_mask, "output_layer": 12}
            with torch.no_grad():
                logits = self.hubert.extract_features(**inputs)
                feats = logits[0]
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

            # Alinear longitudes de tiempo
            p_len = min(feats.shape[1], len(f0))
            feats = feats[:, :p_len, :]
            pitch = torch.from_numpy(f0_coarse[:p_len]).unsqueeze(0).to(self.device)
            pitchf = torch.from_numpy(f0[:p_len]).unsqueeze(0).float().to(self.device)
            p_len_t = torch.tensor([p_len], device=self.device)
            sid = torch.tensor([0], device=self.device)

            # Inferencia con generador VITS
            with torch.no_grad():
                audio_out = self.net_g.infer(feats, p_len_t, pitch, pitchf, sid)[0][0, 0].data.cpu().float().numpy()

            sf.write(output_audio_path, audio_out, self.sr)
            logger.info(f"Conversión RVC completada exitosamente: {output_audio_path}")
            return True

        except Exception as e:
            logger.error(f"Error en conversión RVC: {e}")
            raise e

rvc_engine = RVCEngine()
