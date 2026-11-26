import os
import sys
from pathlib import Path

# Configurar encoding UTF-8 en consola de Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import app.config

def download_local_models():
    print("=" * 65)
    print(" [>] DESCARGA AUTOMATICA DE MODELO ESPAÑOL XTTS-v2 (PORTABLE USB)")
    print("=" * 65)
    print(f" [*] Guardando modelos en: {app.config.MODELS_DIR}\n")
    
    try:
        from huggingface_hub import snapshot_download
        
        target_dir = app.config.MODELS_DIR / "xtts_v2"
        if target_dir.exists() and len(list(target_dir.rglob("*"))) > 3:
            print(f" [OK] El modelo XTTS-v2 (Español) ya existe en: {target_dir}")
            return True
            
        print(" [1/1] Descargando 'coqui/XTTS-v2' (SOTA Español Zero-Shot) desde HuggingFace...")
        print(" Por favor espera unos momentos mientras se descargan los pesos...\n")
        
        snapshot_download(
            repo_id="coqui/XTTS-v2",
            local_dir=str(target_dir)
        )
        
        print("\n" + "=" * 65)
        print(" [OK] MODELO MULTILINGÜE Y ESPAÑOL XTTS-v2 DESCARGADO CON EXITO.")
        print("=" * 65 + "\n")
        return True
        
    except Exception as e:
        print(f"\n [!] Error durante la descarga automatica: {e}\n")
        return False

if __name__ == "__main__":
    download_local_models()
