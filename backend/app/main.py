import os
import shutil
import uuid
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import json
from app.config import (
    BACKEND_DIR,
    MODELS_DIR,
    SAMPLES_DIR,
    CLIPS_DIR,
    SETTINGS_FILE,
)
from app.audio_utils import get_audio_duration, validate_audio_file
from app.tts_engine import tts_engine
from app.rvc_engine import rvc_engine
from app.quality_evaluator import quality_evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="API Local de Clonación de Voz",
    description="API portable en Python para grabación de voz, clonación Zero-Shot con XTTS-v2 y generación de clips.",
    version="1.0.0"
)

# Permitir CORS para desarrollo local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Esquemas Pydantic
class SettingsModel(BaseModel):
    language: Optional[str] = "es"
    output_format: Optional[str] = "wav"
    temperature: Optional[float] = 0.75
    top_p: Optional[float] = 0.85
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 3.5
    length_penalty: Optional[float] = 1.0
    speed: Optional[float] = 1.0
    use_rvc: Optional[bool] = False
    rvc_model: Optional[str] = "david"
    pitch_shift: Optional[int] = 0
    evaluate_quality: Optional[bool] = True
    auto_retry: Optional[bool] = False
    min_quality_threshold: Optional[int] = 75
    max_retry_attempts: Optional[int] = 2

class GenerateClipRequest(BaseModel):
    text: str
    sample_filename: Optional[str] = None
    language: Optional[str] = "es"
    output_format: Optional[str] = "wav"
    temperature: Optional[float] = 0.75
    top_p: Optional[float] = 0.85
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 3.5
    length_penalty: Optional[float] = 1.0
    speed: Optional[float] = 1.0
    use_rvc: Optional[bool] = False
    rvc_model: Optional[str] = None
    pitch_shift: Optional[int] = 0
    evaluate_quality: Optional[bool] = True
    auto_retry: Optional[bool] = False
    min_quality_threshold: Optional[int] = 75
    max_retry_attempts: Optional[int] = 2



# Configuración predeterminada de hiperparámetros y preferencias
DEFAULT_SETTINGS = {
    "language": "es",
    "output_format": "wav",
    "temperature": 0.75,
    "top_p": 0.85,
    "top_k": 50,
    "repetition_penalty": 3.5,
    "length_penalty": 1.0,
    "speed": 1.0,
    "use_rvc": False,
    "rvc_model": "david",
    "pitch_shift": 0,
    "evaluate_quality": True,
    "auto_retry": False,
    "min_quality_threshold": 75,
    "max_retry_attempts": 2
}

# Endpoints de API

@app.get("/api/settings")
def get_settings():
    """Retorna la configuración guardada del usuario o los valores por defecto."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                return {**DEFAULT_SETTINGS, **saved}
        except Exception as e:
            logger.warning(f"Error leyendo settings.json: {e}")
    return DEFAULT_SETTINGS


@app.post("/api/settings")
def save_settings(settings: SettingsModel):
    """Guarda los parámetros de usuario para persistencia entre sesiones."""
    try:
        data = settings.dict()
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return {"message": "Configuración guardada exitosamente.", "settings": data}
    except Exception as e:
        logger.error(f"Error guardando configuración: {e}")
        raise HTTPException(status_code=500, detail=f"Error guardando configuración: {str(e)}")


@app.get("/api/status")
def get_status():
    """Retorna el estado de la API, hardware (CUDA/CPU) y modelos descargados."""
    import torch
    
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    
    samples = list(SAMPLES_DIR.glob("*.wav"))
    clips = list(CLIPS_DIR.glob("*.wav"))
    
    model_downloaded = (MODELS_DIR / "xtts_v2").exists() or (MODELS_DIR / "speecht5_tts").exists()
    
    rvc_models = rvc_engine.list_models()

    return {
        "status": "online",
        "device": device_name,
        "cuda_available": cuda_available,
        "model_loaded": tts_engine.is_loaded(),
        "engine_type": tts_engine.engine_type,
        "model_downloaded": model_downloaded,
        "models_dir": str(MODELS_DIR),
        "total_samples": len(samples),
        "total_clips": len(clips),
        "rvc_models": rvc_models,
    }


@app.get("/api/rvc/models")
def get_rvc_models():
    """Retorna los modelos de voz RVC disponibles (.pth)."""
    return {"models": rvc_engine.list_models()}


@app.post("/api/models/download")
def download_model():
    """Descarga/Pre-inicializa el modelo local de clonación de voz en ./backend/models/."""
    try:
        success = tts_engine.load_model()
        if success:
            return {
                "message": "¡Modelo local cargado exitosamente en ./backend/models/!",
                "engine": tts_engine.engine_type
            }
        else:
            raise HTTPException(status_code=500, detail="No se pudo descargar/cargar el modelo.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/samples/upload")
async def upload_sample(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None)
):
    """Subir o guardar una grabación de muestra de voz (.wav, .m4a, .mp3, etc)."""
    allowed_exts = ('.wav', '.mp3', '.ogg', '.m4a', '.aac', '.flac', '.wma', '.3gp', '.caf', '.webm', '.weba')
    if not file.filename.lower().endswith(allowed_exts):
        raise HTTPException(status_code=400, detail="Formato de audio no soportado. Usa WAV, MP3, AAC, M4A u OGG.")


    filename_clean = Path(file.filename).stem
    safe_name = name.strip() if name and name.strip() else filename_clean
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in ("_", "-")).rstrip()
    
    save_path = SAMPLES_DIR / f"{safe_name}.wav"
    
    # Si ya existe, agregar un sufijo único
    if save_path.exists():
        unique_id = uuid.uuid4().hex[:6]
        save_path = SAMPLES_DIR / f"{safe_name}_{unique_id}.wav"

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Convertir a formato WAV PCM estándar 16kHz
        from app.audio_utils import convert_to_pcm_wav
        convert_to_pcm_wav(str(save_path))

        is_valid, msg = validate_audio_file(str(save_path))
        duration = get_audio_duration(str(save_path))

        
        return {
            "message": "Muestra de voz guardada exitosamente.",
            "filename": save_path.name,
            "duration_seconds": round(duration, 2),
            "validation": msg,
            "path": str(save_path)
        }
    except Exception as e:
        logger.error(f"Error al guardar la muestra de audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error guardando muestra: {str(e)}")

@app.get("/api/samples")
def list_samples():
    """Listar todas las muestras de voz almacenadas localmente."""
    samples = []
    for file in SAMPLES_DIR.glob("*.wav"):
        duration = get_audio_duration(str(file))
        samples.append({
            "filename": file.name,
            "size_bytes": file.stat().st_size,
            "duration_seconds": round(duration, 2),
            "modified_time": file.stat().st_mtime
        })
    return sorted(samples, key=lambda x: x["modified_time"], reverse=True)

@app.delete("/api/samples/{filename}")
def delete_sample(filename: str):
    """Eliminar una muestra de voz."""
    sample_path = SAMPLES_DIR / filename
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Muestra de voz no encontrada.")
    
    sample_path.unlink()
    return {"message": f"Muestra '{filename}' eliminada exitosamente."}

@app.post("/api/clone/generate")
def generate_clip(req: GenerateClipRequest):
    """Generar un clip de voz a partir de un texto y una muestra de audio con autoreintento y evaluación opcional."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="El texto para generar no puede estar vacío.")

    sample_file = req.sample_filename
    if not sample_file:
        all_samples = list(SAMPLES_DIR.glob("*.wav"))
        if all_samples:
            sample_file = all_samples[0].name
        else:
            raise HTTPException(status_code=400, detail="Debes grabar o subir al menos una muestra de audio.")

    sample_path = SAMPLES_DIR / sample_file
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"La muestra de voz '{sample_file}' no existe.")

    # Parámetros de autoreintento
    max_attempts = max(1, min(3, req.max_retry_attempts or 2)) if req.auto_retry else 1
    threshold = max(40, min(95, req.min_quality_threshold or 75))

    best_clip_path = None
    best_quality = None
    best_filename = None
    attempts_log = []

    base_temp = req.temperature if req.temperature is not None else 0.75
    base_top_p = req.top_p if req.top_p is not None else 0.85

    try:
        for attempt in range(1, max_attempts + 1):
            clip_id = uuid.uuid4().hex[:8]
            attempt_filename = f"clip_{clip_id}.wav"
            attempt_path = CLIPS_DIR / attempt_filename

            # Micro-ajuste tonal si es un reintento para evitar el mismo mínimo local
            current_temp = base_temp
            current_top_p = base_top_p
            if attempt > 1:
                jitter = 0.04 if (attempt % 2 == 0) else -0.04
                current_temp = max(0.25, min(1.15, base_temp + jitter))
                current_top_p = max(0.40, min(0.98, base_top_p + (jitter * 0.5)))
                logger.info(f"Reintento {attempt}/{max_attempts}: Explorando variante con temp={current_temp:.2f}, top_p={current_top_p:.2f}")

            # 1. Síntesis neuronal con TTS
            tts_engine.generate_voice_clip(
                text=req.text.strip(),
                speaker_wav=str(sample_path),
                output_wav=str(attempt_path),
                language=req.language or "es",
                temperature=current_temp,
                top_p=current_top_p,
                top_k=req.top_k if req.top_k is not None else 50,
                repetition_penalty=req.repetition_penalty if req.repetition_penalty is not None else 3.5,
                length_penalty=req.length_penalty if req.length_penalty is not None else 1.0,
                speed=req.speed if req.speed is not None else 1.0
            )

            # 2. Afinación de timbre con RVC si está activada
            if req.use_rvc and req.rvc_model:
                rvc_temp_path = CLIPS_DIR / f"rvc_{attempt_filename}"
                logger.info(f"Aplicando transformación RVC ({req.rvc_model}) en intento {attempt}...")
                rvc_engine.convert_voice(
                    input_audio_path=str(attempt_path),
                    output_audio_path=str(rvc_temp_path),
                    model_name=req.rvc_model,
                    pitch_shift=req.pitch_shift or 0
                )
                if rvc_temp_path.exists():
                    attempt_path.unlink(missing_ok=True)
                    rvc_temp_path.replace(attempt_path)

            # 3. Evaluación de calidad automática por IA
            eval_res = None
            if req.evaluate_quality:
                eval_res = quality_evaluator.evaluate_clip(
                    output_wav=str(attempt_path),
                    speaker_wav=str(sample_path),
                    text=req.text.strip(),
                    language=req.language or "es",
                    tts_engine=tts_engine
                )
                eval_res["attempt_number"] = attempt
                eval_res["total_attempts"] = max_attempts
                eval_res["threshold_set"] = threshold
                eval_res["met_threshold"] = eval_res["overall_score"] >= threshold
                attempts_log.append({
                    "attempt": attempt,
                    "score": eval_res["overall_score"],
                    "rating": eval_res["rating"],
                    "similarity": eval_res["speaker_similarity"],
                    "clarity": eval_res["acoustic_clarity"]
                })
                logger.info(f"Intento {attempt}/{max_attempts} -> Calificación: {eval_res['overall_score']}% ({eval_res['rating']})")

            current_score = eval_res["overall_score"] if eval_res else 100

            # Determinar si este intento supera al mejor previo
            if best_quality is None or current_score > best_quality.get("overall_score", 0):
                if best_clip_path and best_clip_path.exists() and best_clip_path != attempt_path:
                    best_clip_path.unlink(missing_ok=True)
                best_clip_path = attempt_path
                best_quality = eval_res
                best_filename = attempt_filename
            else:
                # Descartar archivo inferior
                attempt_path.unlink(missing_ok=True)

            # Condición de éxito: umbral superado o autoreintento no requerido
            if not req.auto_retry or (eval_res and eval_res["overall_score"] >= threshold):
                logger.info(f"Clip finalizado exitosamente (score {current_score}% >= umbral {threshold}%).")
                break
            else:
                logger.info(f"Calidad {current_score}% por debajo de {threshold}%. Procediendo a reintento...")

        if not best_clip_path or not best_clip_path.exists():
            raise RuntimeError("No se pudo generar un clip de audio válido.")

        # Añadir historial de intentos al reporte de calidad
        if best_quality:
            best_quality["attempts_history"] = attempts_log
            best_quality["attempts_count"] = len(attempts_log)

        # 4. Exportar al formato deseado (wav, mp3, aac, ogg)
        from app.audio_utils import export_audio_format
        target_fmt = (req.output_format or "wav").lower().strip()
        final_path_str = export_audio_format(str(best_clip_path), target_format=target_fmt)
        final_filename = Path(final_path_str).name
        duration = get_audio_duration(final_path_str)

        # 5. Guardar metadatos de calidad del clip
        meta_file = CLIPS_DIR / f"{Path(final_filename).stem}.meta.json"
        try:
            with open(meta_file, "w", encoding="utf-8") as mf:
                json.dump({
                    "clip_filename": final_filename,
                    "duration_seconds": round(duration, 2),
                    "created_at": Path(final_path_str).stat().st_mtime,
                    "quality": best_quality,
                    "text_prompt": req.text[:100],
                    "sample_used": sample_file
                }, mf, indent=2, ensure_ascii=False)
        except Exception as e_meta:
            logger.warning(f"No se pudieron persistir los metadatos del clip: {e_meta}")

        return {
            "message": "Clip de voz generado exitosamente.",
            "clip_filename": final_filename,
            "duration_seconds": round(duration, 2),
            "download_url": f"/api/clips/download/{final_filename}",
            "quality": best_quality
        }

    except Exception as e:
        logger.error(f"Error generando clip: {e}")
        raise HTTPException(status_code=500, detail=f"Error en la clonación: {str(e)}")


@app.get("/api/clips")
def list_clips():
    """Listar todos los clips generados con sus metadatos de calidad."""
    clips = []
    for file in CLIPS_DIR.glob("*.*"):
        if file.suffix.lower() in ('.wav', '.mp3', '.aac', '.ogg', '.m4a', '.flac'):
            duration = get_audio_duration(str(file))
            
            # Cargar metadatos de calidad si existen
            quality_data = None
            meta_file = CLIPS_DIR / f"{file.stem}.meta.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as mf:
                        m = json.load(mf)
                        quality_data = m.get("quality")
                except Exception:
                    pass

            clips.append({
                "filename": file.name,
                "size_bytes": file.stat().st_size,
                "duration_seconds": round(duration, 2),
                "modified_time": file.stat().st_mtime,
                "download_url": f"/api/clips/download/{file.name}",
                "quality": quality_data
            })
    return sorted(clips, key=lambda x: x["modified_time"], reverse=True)


@app.get("/api/clips/{filename}/quality")
def evaluate_existing_clip(filename: str, sample_filename: Optional[str] = None):
    """Evalúa la calidad de un clip existente en la biblioteca a petición."""
    clip_path = CLIPS_DIR / filename
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip no encontrado.")

    speaker_path = None
    if sample_filename:
        s_path = SAMPLES_DIR / sample_filename
        if s_path.exists():
            speaker_path = str(s_path)
    else:
        samples = sorted(list(SAMPLES_DIR.glob("*.wav")), key=lambda p: p.stat().st_mtime, reverse=True)
        if samples:
            speaker_path = str(samples[0])

    res = quality_evaluator.evaluate_clip(
        output_wav=str(clip_path),
        speaker_wav=speaker_path,
        tts_engine=tts_engine
    )

    # Actualizar o guardar en .meta.json
    meta_file = CLIPS_DIR / f"{clip_path.stem}.meta.json"
    try:
        meta_data = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as mf:
                meta_data = json.load(mf)
        meta_data["quality"] = res
        with open(meta_file, "w", encoding="utf-8") as mf:
            json.dump(meta_data, mf, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Error actualizando meta de calidad: {e}")

    return res


@app.get("/api/clips/download/{filename}")
def download_clip(filename: str):
    """Descargar o reproducir un clip generado o una muestra de voz."""
    clip_path = CLIPS_DIR / filename
    sample_path = SAMPLES_DIR / filename
    
    target_path = clip_path if clip_path.exists() else (sample_path if sample_path.exists() else None)
    
    if not target_path:
        raise HTTPException(status_code=404, detail="Archivo de audio no encontrado.")
        
    ext = target_path.suffix.lower()
    media_type_map = {
        ".mp3": "audio/mpeg",
        ".aac": "audio/aac",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".wav": "audio/wav",
        ".flac": "audio/flac"
    }
    media_type = media_type_map.get(ext, "audio/wav")
    return FileResponse(path=target_path, media_type=media_type, filename=filename)


@app.delete("/api/clips/{filename}")
def delete_clip(filename: str):
    """Eliminar un clip generado y sus metadatos asociados."""
    clip_path = CLIPS_DIR / filename
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip de audio no encontrado.")
    
    clip_path.unlink()
    # Eliminar archivo meta asociado si existe
    (CLIPS_DIR / f"{clip_path.stem}.meta.json").unlink(missing_ok=True)
    return {"message": f"Clip '{filename}' eliminado."}

# Servir archivos estáticos del Frontend
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
