import os
from pathlib import Path


def detect_instance() -> str:
    """Detekcja nazwy instancji na podstawie katalogu lub HAI_INSTANCE."""
    env = os.environ.get("HAI_INSTANCE")
    if env:
        return env
    cwd = Path.cwd()
    # nowa struktura: HAI-NT/EPV → HAI_EPV
    if cwd.parent.name == "HAI-NT":
        return "HAI_" + cwd.name
    # HAI-NL (z minusem!)
    if cwd.name == "HAI-NL":
        return "HAI_NL"
    if cwd.parent.name == "HAI-NL":
        return "HAI_NL"
    # legacy: HAI_* w nazwie
    if cwd.name.startswith("HAI_"):
        return cwd.name
    f = Path(__file__).resolve()
    for p in f.parents:
        if p.name.startswith("HAI_"):
            return p.name
    return "HAI_UNKNOWN"


def instance_port(instance: str = None) -> int:
    """Zwraca domyślny port dla instancji."""
    ports = {
        "HAI_EPV": 5010,
        "HAI_DEV": 5015,
        "HAI_LAB": 5020,
        "HAI_LIV": 5025,
        "HAI_SNP": 5030,
        "HAI_TST": 5030,   # legacy alias, 2026-08-01 SNP zastapil TST
        "HAI_NL":  5005,
    }
    return ports.get(instance or detect_instance(), 5010)


INSTANCE = detect_instance()
