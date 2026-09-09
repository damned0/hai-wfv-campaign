# ===========================================
# Bramki KONTRARIAŃSKIE (warstwa abstencji w backtesterze/live)
# ===========================================
# Odkrycia (2026-07-24, ict_edge_test_v2 + of_decisive_test): sygnaly sa
# KONTRARIAŃSKIE (faduj przebicie/wypelnienie/CVD):
#   - of_cvd: wysokie skumulowane CVD -> nizszy zwrot (24-48h). Wymaga of_* (BTC+pula).
#   - BOS (break of structure): bull BOS -> down (3-6h, rho -0.09). Z SAMYCH swiec.
#   - FVG mitygacja: bull FVG wypelniany -> down (24-48h, rho -0.089). Z SAMYCH swiec.
# Gate = ABSTENCJA: wetuje kierunek modelu ZGODNY z (fadeowanym) sygnalem.
#   bull sygnal (expect DOWN) -> veto LONG; bear sygnal (expect UP) -> veto SHORT.
# Env: HAI_CVD_GATE / HAI_BOS_GATE / HAI_FVG_GATE (=1). Backtester wola gdy ktorykolwiek.
import os
import numpy as np
import pandas as pd
from pathlib import Path

_WH = Path(os.environ.get("HAI_WH", "/root/ProjektHAI/data_warehouse"))
CVD_HIGH = float(os.environ.get("HAI_CVD_HIGH", "0.60"))
CVD_LOW = float(os.environ.get("HAI_CVD_LOW", "0.40"))
CVD_WIN = int(os.environ.get("HAI_CVD_WIN", "168"))
BOS_K = int(os.environ.get("HAI_BOS_K", "6"))    # okno aktywnosci BOS (h)
FVG_K = int(os.environ.get("HAI_FVG_K", "24"))   # okno aktywnosci FVG (h)
PIVOT_N = 3


def of_cvd_pct(candles_1h, symbol):
    """percentyl of_cvd (rolling CVD_WIN) wyrownany do candles; None gdy brak of_*."""
    stem = symbol.split("/")[0].split(":")[0]
    p = _WH / "orderflow" / "binance" / f"{stem}.parquet"
    if not p.exists():
        return None
    of = pd.read_parquet(p)
    if "of_cvd" not in of.columns:
        return None
    of["ts"] = pd.to_datetime(of["timestamp"]).dt.floor("h").astype("datetime64[ns]")
    of = of.sort_values("ts").reset_index(drop=True)
    of["cvd_pct"] = of["of_cvd"].rolling(CVD_WIN, min_periods=24).rank(pct=True)
    lut = dict(zip(of["ts"].astype("int64").values, of["cvd_pct"].values))
    ts_ns = (pd.to_datetime([c["timestamp"] for c in candles_1h], unit="ms")
             .floor("h").astype("datetime64[ns]").astype("int64"))
    return np.array([lut.get(int(k), np.nan) for k in ts_ns])


def _struct_bias(candles_1h):
    """Zwraca (bos_expect, fvg_expect): kontrariańska EKSPEKTACJA kierunku per swieca
    (+1 = expect UP / fade shorts, -1 = expect DOWN / fade longs, 0 = brak),
    z oknem aktywnosci BOS_K / FVG_K. Liczone z samych OHLC."""
    h = np.array([c["high"] for c in candles_1h], dtype=float)
    l = np.array([c["low"] for c in candles_1h], dtype=float)
    c = np.array([c["close"] for c in candles_1h], dtype=float)
    n = len(c)
    N = PIVOT_N
    piv_at = [[] for _ in range(n)]
    for i in range(N, n - N):
        if h[i] == max(h[i-N:i+N+1]):
            if i+N < n: piv_at[i+N].append((h[i], 'H'))
        if l[i] == min(l[i-N:i+N+1]):
            if i+N < n: piv_at[i+N].append((l[i], 'L'))
    # BOS/CHoCH state machine -> bos_dir (nominalny kierunek przebicia)
    bias = 0; key_high = np.nan; key_low = np.nan
    bos_raw = np.zeros(n)
    for i in range(n):
        for price, typ in piv_at[i]:
            if typ == 'H': key_high = price
            else: key_low = price
        if not np.isnan(key_high) and c[i] > key_high:
            bos_raw[i] = 1 if bias >= 0 else 0  # bull BOS (kontynuacja); CHoCH ignorujemy (slaby)
            if bias == 0: bias = 1
            elif bias < 0: bias = 1
            key_high = np.nan
        if not np.isnan(key_low) and c[i] < key_low:
            bos_raw[i] = -1 if bias <= 0 else 0
            if bias == 0: bias = -1
            elif bias > 0: bias = -1
            key_low = np.nan
    # FVG mitygacja -> fvg_raw (kierunek luki mitygowanej)
    zones = []; fvg_raw = np.zeros(n)
    for i in range(n):
        if i >= 2:
            if l[i] > h[i-2]: zones.append([l[i], h[i-2], 1, i])
            elif h[i] < l[i-2]: zones.append([l[i-2], h[i], -1, i])
        still = []; fired = 0
        for z in zones:
            top, bot, dr, born = z
            if i - born < 1: still.append(z); continue
            if dr == 1:
                if l[i] <= top and l[i] >= bot:
                    if not fired: fvg_raw[i] = 1; fired = 1
                    still.append(z)
                elif l[i] >= bot: still.append(z)
            else:
                if h[i] >= bot and h[i] <= top:
                    if not fired: fvg_raw[i] = -1; fired = 1
                    still.append(z)
                elif h[i] <= top: still.append(z)
        zones = still[-200:]
    # rozciagnij zdarzenia na okno aktywnosci + KONTRARIAN (odwroc znak: nominalny bull -> expect DOWN)
    def spread(raw, K):
        out = np.zeros(n)
        for i in range(n):
            if raw[i] != 0:
                out[i:min(i+K, n)] = -raw[i]  # kontrarian: bull sygnal -> expect DOWN (-1)
        return out
    return spread(bos_raw, BOS_K), spread(fvg_raw, FVG_K)


def apply_gate(batch_action, candles_1h, symbol):
    """Wetuje (0) kierunki modelu zgodne z fadeowanym sygnalem. Suma bramek wg env.
    Zwraca (batch_action, n_veto). Brak sygnalu/danych -> bez zmian dla tej bramki."""
    ba = np.asarray(batch_action)
    n = len(ba)
    cvd_on = os.environ.get("HAI_CVD_GATE") == "1"
    bos_on = os.environ.get("HAI_BOS_GATE") == "1"
    fvg_on = os.environ.get("HAI_FVG_GATE") == "1"
    # expect_down maski (gdzie veto LONG) i expect_up (gdzie veto SHORT)
    veto_long = np.zeros(n, bool)
    veto_short = np.zeros(n, bool)
    if cvd_on:
        cvdp = of_cvd_pct(candles_1h, symbol)
        if cvdp is not None:
            v = ~np.isnan(cvdp)
            veto_long |= v & (cvdp > CVD_HIGH)   # wysokie CVD -> expect down -> veto LONG
            veto_short |= v & (cvdp < CVD_LOW)
    if bos_on or fvg_on:
        bos_e, fvg_e = _struct_bias(candles_1h)
        if bos_on:
            veto_long |= (bos_e < 0)   # expect down
            veto_short |= (bos_e > 0)
        if fvg_on:
            veto_long |= (fvg_e < 0)
            veto_short |= (fvg_e > 0)
    veto = (ba == 1) & veto_long | (ba == -1) & veto_short
    nv = int(np.sum(veto))
    if nv:
        ba = ba.copy(); ba[veto] = 0
    return ba, nv
