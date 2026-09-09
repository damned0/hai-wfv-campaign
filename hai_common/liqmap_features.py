# ===========================================
# Cechy mapy likwidacji — WSPOLNE zrodlo dla treningu i backtestu (parzystosc!)
# ===========================================
# Wynik weryfikacji (2026-07-22, genflow_liqmap_deep na 2 latach):
#   - liq_imbalance: REZIMOWY, odrzucony (z 0.492->0.103 na 2 latach).
#   - dist_below_liq / dist_above_liq: STABILNE przez 2 lata (z=0.21-0.27),
#     dokladaja drzewom +0.101 precyzji_LONG PONAD cechy cenowe (honest hold-out).
#
# METODA (zamrozona, jak Coinglass estymuje free): nowe OI (delta dodatnia) otwiera
# pozycje ~przy biezacej cenie; ls_ratio dzieli je na long/short; przy dzwigni L
# pozycja likwiduje sie ~1/L od wejscia (long w dol, short w gore). Akumulujemy
# "paliwo" per poziom, usuwamy gdy cena przez nie przejdzie (zlikwidowane) lub po
# decay. Cechy = dystans (w ATR) do najblizszego klastra ponizej/powyzej ceny.
#
# ANTY-LEAKAGE: ledger budowany INKREMENTALNIE tylko z przeszlosci (OI/cena <= t).
# Wartosc w wierszu i zalezy wylacznie od danych do i wlacznie.
#
# PARZYSTOSC: ml_trainer (dataset) i backtester (feat_src) MUSZA wolac te sama
# funkcje na tych samych danych, inaczej powtorka buga derywatow (brak cechy ->
# feat_src.get(f, zeros) wstawia zera -> scaler robi z nich OOD -> smieciowy routing).
# ===========================================
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Parametry ZAMROZONE (identyczne jak w zweryfikowanym genflow_liqmap_deep.py)
LEVERAGES = [(25, 0.25), (50, 0.35), (100, 0.40)]  # (dzwignia, waga udzialu) - retail high-lev
DECAY_H = 24 * 14        # paliwo starzeje sie po ~14 dniach
# Default gdy brak mapy (przed pokryciem ls_ratio/OI albo pusty ledger): duzy
# dystans = "zaden klaster nie jest blisko" = neutralne (NIE zero, ktore znaczy
# "klaster dokladnie na cenie" = przeciwna semantyka i OOD dla scalera).
DIST_DEFAULT = 30.0      # ATR; ~poza zasiegiem realnych klastrow (obserwowane mu 8-19)

_WH = Path(os.environ.get("HAI_WH", "/root/ProjektHAI/data_warehouse"))


def _atr(h, l, c, n=14):
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.concatenate([[h[0] - l[0]], tr])
    return pd.Series(tr).rolling(n, min_periods=1).mean().values


def compute_liq_dist(ts, close, high, low, oi_ts, oi_val, ls_ts, ls_val):
    """Zwraca (dist_below_liq, dist_above_liq) w ATR, wyrownane do ts (1h grid).

    ts/close/high/low: tablice 1h (posortowane rosnaco po ts).
    oi_ts/oi_val: szereg open_interest (dowolna granularnosc; reindex+ffill na ts).
    ls_ts/ls_val: szereg ls_ratio (long/short; reindex+ffill na ts).
    Gdy brak pokrycia OI/ls w danym wierszu -> ledger nie rosnie, dist = DIST_DEFAULT.
    """
    # ujednolic dtype czasu na ns (merge_asof wymaga zgodnych typow ms/us/ns)
    _ts = pd.to_datetime(pd.Series(ts)).astype("datetime64[ns]")
    n = len(_ts)
    c = np.asarray(close, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    a = _atr(h, l, c)

    # reindex OI i ls na siatke ts (backward ffill: tylko przeszle wartosci)
    grid = pd.DataFrame({"ts": _ts.values})
    oi_df = pd.DataFrame({"ts": pd.to_datetime(pd.Series(oi_ts)).astype("datetime64[ns]").values,
                          "oi": np.asarray(oi_val, dtype=np.float64)}).sort_values("ts")
    ls_df = pd.DataFrame({"ts": pd.to_datetime(pd.Series(ls_ts)).astype("datetime64[ns]").values,
                          "ls": np.asarray(ls_val, dtype=np.float64)}).sort_values("ts")
    g = pd.merge_asof(grid, oi_df, on="ts", direction="backward")
    g = pd.merge_asof(g, ls_df, on="ts", direction="backward")
    oiv = g["oi"].values
    lsr = g["ls"].values

    dist_b = np.full(n, DIST_DEFAULT, dtype=np.float64)
    dist_a = np.full(n, DIST_DEFAULT, dtype=np.float64)

    ledger = []  # (liq_price, mass, side, born_i)
    oi_prev = np.nan
    for i in range(n):
        oi_i = oiv[i]
        ls_i = lsr[i]
        # nowe OI (delta dodatnia) tylko gdy mamy biezacy i poprzedni odczyt OI+ls
        if np.isfinite(oi_i) and np.isfinite(oi_prev) and np.isfinite(ls_i):
            d_oi = max(0.0, oi_i - oi_prev)
            if d_oi > 0:
                long_frac = ls_i / (1.0 + ls_i)
                for L, w in LEVERAGES:
                    ledger.append((c[i] * (1 - 1.0 / L), d_oi * long_frac * w, 'long', i))
                    ledger.append((c[i] * (1 + 1.0 / L), d_oi * (1 - long_frac) * w, 'short', i))
        if np.isfinite(oi_i):
            oi_prev = oi_i
        # usun zlikwidowane (cena przeszla przez poziom) i przestarzale
        if ledger:
            ledger = [(p, m, s, b) for (p, m, s, b) in ledger
                      if (i - b) < DECAY_H
                      and not (s == 'long' and l[i] <= p)
                      and not (s == 'short' and h[i] >= p)]
        if ledger and a[i] > 0:
            below = [c[i] - p for (p, m, s, b) in ledger if p < c[i]]
            above = [p - c[i] for (p, m, s, b) in ledger if p > c[i]]
            if below:
                dist_b[i] = min(min(below) / a[i], DIST_DEFAULT)
            if above:
                dist_a[i] = min(min(above) / a[i], DIST_DEFAULT)
    return dist_b, dist_a


def compute_liq_dist_from_warehouse(df_1h, symbol, oi_df=None):
    """Wygodny wrapper: liczy dist_* dla ramki 1h (kol. timestamp/close/high/low).

    ls_ratio z GLEBOKIEGO zrodla (ls_ratio_deep/, Binance metrics 2 lata, corr 0.996
    z Coinalyze) — patrz backfill_binance_metrics.py. OI: przekazane oi_df albo
    dzienne z open_interest/ (jak w zweryfikowanym tescie). Brak plikow -> DIST_DEFAULT.
    """
    ts = pd.to_datetime(df_1h["timestamp"]).values
    close = df_1h["close"].values
    high = df_1h["high"].values
    low = df_1h["low"].values

    # OI (dzienne, jak w walidacji)
    if oi_df is None:
        oip = _WH / "derivatives" / "open_interest" / f"{symbol}.parquet"
        if oip.exists():
            oi_df = pd.read_parquet(oip)
        else:
            return (np.full(len(ts), DIST_DEFAULT), np.full(len(ts), DIST_DEFAULT))
    oi_ts = pd.to_datetime(oi_df["timestamp"]).values
    oi_col = "close" if "close" in oi_df.columns else oi_df.columns[-1]
    oi_val = oi_df[oi_col].values

    # ls_ratio GLEBOKI (fallback do Coinalyze 3-mies jesli brak deep)
    lsp = _WH / "derivatives" / "ls_ratio_deep" / f"{symbol}.parquet"
    if not lsp.exists():
        lsp = _WH / "derivatives" / "ls_ratio" / f"{symbol}.parquet"
    if not lsp.exists():
        return (np.full(len(ts), DIST_DEFAULT), np.full(len(ts), DIST_DEFAULT))
    ls_df = pd.read_parquet(lsp).dropna()
    ls_ts = pd.to_datetime(ls_df["timestamp"]).values
    ls_val = ls_df["ls_ratio"].values

    return compute_liq_dist(ts, close, high, low, oi_ts, oi_val, ls_ts, ls_val)
