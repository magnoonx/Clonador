import os
import sys
import socket
import webbrowser
from pathlib import Path

# Configurar encoding UTF-8 en consola de Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Agregar backend al path de python
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

# Cargar la configuración portable antes de importar uvicorn/torch
import app.config

def get_local_ip() -> str:
    """Obtiene la IP local de la máquina en la red (LAN)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.254.254.254', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

if __name__ == "__main__":
    import uvicorn
    
    port = 8000
    lan_ip = get_local_ip()
    
    print("=" * 65)
    print(" [>] INICIANDO ESTUDIO DE CLONACION DE VOZ (PORTABLE USB)")
    print("=" * 65)
    print(f" [*] Modelos locales guardados en: {app.config.MODELS_DIR}")
    print(f" [*] Muestras y clips guardados en: {app.config.DATA_DIR}")
    print(f" [OK] Acceso local (esta PC):       http://127.0.0.1:{port}")
    if lan_ip != "127.0.0.1":
        print(f" [OK] Acceso en red local (LAN):     http://{lan_ip}:{port}")
    print("=" * 65)
    
    # Abrir navegador en la PC local
    webbrowser.open(f"http://127.0.0.1:{port}")
    
    # host="0.0.0.0" permite acceso tanto local como desde otros dispositivos en la red
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False, reload_excludes=["backend/data/*", "backend/models/*"])

