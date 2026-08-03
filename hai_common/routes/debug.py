# ===========================================
# HAI_EPV Engine ver.10 Final — routes/debug.py
# Created by Hauzer | Coded & produced by Claude Sonnet 5
# ===========================================
# Funkcje: /debug/syntax (py_compile sprawdzenie skladni), /system/stats,
# /vm/processes, /vm/host — diagnostyka systemowa/VM dla dashboardu.
# ===========================================
from fastapi import APIRouter
import py_compile
from pathlib import Path

router = APIRouter()

@router.get("/debug/syntax")
async def check_syntax():
    base = Path(__file__).resolve().parent.parent
    results = {}
    for folder in ["routes", "core"]:
        for py_file in (base / folder).glob("*.py"):
            try:
                py_compile.compile(str(py_file), doraise=True)
                results[str(py_file)] = "OK"
            except py_compile.PyCompileError as e:
                results[str(py_file)] = f"ERROR: {e}"
    return results


@router.get("/system/stats")
async def system_stats():
    import psutil, time, os
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    net = psutil.net_io_counters()

    # load average (1/5/15 min) - front czyta v.load[0/1/2]
    try:
        load = list(os.getloadavg())
    except Exception:
        load = [0.0, 0.0, 0.0]

    # aktywne polaczenia sieciowe
    try:
        net_active = len([c for c in psutil.net_connections() if c.status == 'ESTABLISHED'])
    except Exception:
        net_active = 0

    temps = {}
    try:
        t = psutil.sensors_temperatures()
        if t:
            for name, entries in t.items():
                if entries:
                    temps[name] = round(entries[0].current, 1)
    except Exception:
        pass

    return {
        "cpu_pct":      round(cpu, 1),
        # front czyta v.cores (nie cpu_cores)
        "cores":        psutil.cpu_count(),
        "cpu_cores":    psutil.cpu_count(),  # zostawiam dla kompatybilnosci
        "mem_total_gb": round(mem.total / 1e9, 1),
        "mem_used_gb":  round(mem.used  / 1e9, 1),
        "mem_pct":      mem.percent,
        "disk_total_gb":round(disk.total / 1e9, 1),
        "disk_used_gb": round(disk.used  / 1e9, 1),
        "disk_pct":     round(disk.percent, 1),
        # front czyta v.net_sent / v.net_recv w BAJTACH (fmtBytes)
        "net_sent":     net.bytes_sent,
        "net_recv":     net.bytes_recv,
        "net_active":   net_active,
        # front czyta v.uptime_s (sekundy, fmtUptime)
        "uptime_s":     int(time.time() - psutil.boot_time()),
        # front czyta v.load[0/1/2]
        "load":         [round(x, 2) for x in load],
        "temps":        temps,
    }


@router.get("/vm/processes")
async def vm_processes():
    """Top procesy wg pamieci - front loadProcesses() czeka [{name, pid, mem_bytes}]."""
    import psutil
    procs = []
    for p in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
        try:
            info = p.info
            mem = info.get('memory_info')
            procs.append({
                'pid':       info.get('pid'),
                'name':      info.get('name') or '?',
                'mem_bytes': mem.rss if mem else 0,
                'cpu_pct':   info.get('cpu_percent') or 0.0,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    # top 10 wg pamieci
    procs.sort(key=lambda x: x['mem_bytes'], reverse=True)
    return procs[:10]


@router.get("/vm/host")
async def vm_host():
    """Info o hoscie - front loadHostInfo() czeka {hostname, kernel, python}."""
    import platform, socket
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = '?'
    return {
        "hostname": hostname,
        "kernel":   f"{platform.system()} {platform.release()}",
        "python":   platform.python_version(),
    }
