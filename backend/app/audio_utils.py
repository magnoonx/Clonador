import os
import wave
import struct
import math
from pathlib import Path
from typing import Tuple

def convert_to_pcm_wav(file_path: str) -> bool:
    """Convierte cualquier archivo de audio (WebM, OGG, MP3, AAC) a formato PCM WAV estándar 16kHz mono."""
    path = Path(file_path)
    if not path.exists():
        return False
        
    temp_path = path.with_suffix(".tmp.wav")
    
    # Intento 1: usar pydub + imageio_ffmpeg
    try:
        import imageio_ffmpeg
        from pydub import AudioSegment
        AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
        
        audio = AudioSegment.from_file(str(path))
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(str(temp_path), format="wav")
        
        temp_path.replace(path)
        return True
    except Exception:
        if temp_path.exists():
            temp_path.unlink()

    # Intento 2: usar soundfile
    try:
        import soundfile as sf
        data, sr = sf.read(str(path))
        sf.write(str(path), data, 16000, subtype='PCM_16')
        return True
    except Exception:
        pass
        
    return False

def export_audio_format(input_wav_path: str, target_format: str = "wav") -> str:
    """Exporta un archivo WAV resultante al formato deseado (wav, mp3, aac, ogg, flac)."""
    target_fmt = target_format.lower().strip()
    if target_fmt not in ("mp3", "aac", "ogg", "m4a", "flac"):
        return input_wav_path
        
    path = Path(input_wav_path)
    output_file = path.with_suffix(f".{target_fmt}")
    
    try:
        import imageio_ffmpeg
        from pydub import AudioSegment
        AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
        
        sound = AudioSegment.from_file(str(path))
        fmt_param = "adts" if target_fmt == "aac" else target_fmt
        sound.export(str(output_file), format=fmt_param)
        
        if output_file.exists() and output_file.stat().st_size > 0:
            path.unlink()
            return str(output_file)
    except Exception as e:
        print(f"Error exportando audio a {target_fmt}: {e}")
        
    return input_wav_path

def get_audio_duration(file_path: str) -> float:
    """Calcula la duración en segundos de un archivo de audio."""
    try:
        with wave.open(file_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        try:
            import soundfile as sf
            info = sf.info(file_path)
            return info.duration
        except Exception:
            return 0.0

def validate_audio_file(file_path: str) -> Tuple[bool, str]:
    """Verifica si el archivo de audio existe y es válido."""
    path = Path(file_path)
    if not path.exists():
        return False, "El archivo de audio no existe."
    
    if path.stat().st_size == 0:
        return False, "El archivo de audio está vacío."
        
    duration = get_audio_duration(file_path)
    if duration < 1.0:
        return False, f"El audio es demasiado corto ({duration:.1f}s). Se recomiendan entre 3 y 10 segundos."
        
    return True, f"Audio válido ({duration:.1f}s)."
