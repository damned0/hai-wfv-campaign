# hai_common - wspólny pakiet dla wszystkich instancji HAI
# (EPV, DEV, LAB, LIV, TST, RES)
# Dokumentuje ścieżki i zależności między instancjami.

import os
from pathlib import Path

HAI_ROOT = Path(os.environ.get("HAI_ROOT", "/root/ProjektHAI"))
