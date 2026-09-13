#!/usr/bin/env python3
"""Dwa testy dla cech 4h/1d (cechy_tf.py):

1. PARYTET Z PRODUKCJA — dla losowych godzin buduje szereg dokladnie jak engine
   (zamkniete swiece tf + swieca w toku z biezaca cena) i liczy go PRAWDZIWYMI
   funkcjami silnika. Wynik cechy_tf.tf_w_toku musi sie zgadzac.
2. PRZECIEK — cecha nie moze mocniej korelowac z przyszla cena niz z przeszla.
   Ten test w 30 sekund zlapal przeciek, ktorego dwa miesiace audytow kodu nie widzialy.
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, "/root/ProjektHAI/hai_common")
from hai_common.cechy_tf import tf_w_toku, rsi_sma, MIN_4H, MIN_1D
from hai_common.strategies.ai_strategy import AIStrategy
from hai_common.strategies.base import BaseStrategy
from hai_common.features import _trend_jak_trening
from scipy.stats import spearmanr

class _S(BaseStrategy):
    def analyze(self,*a,**k): pass
    def get_parameters(self): return {}
    def set_parameters(self,p): pass
S = _S()
W = "/root/ProjektHAI/data_warehouse/ohlcv/binance"

def wczytaj(sym, tf):
    d = pd.read_parquet(f"{W}/{tf}/{sym}.parquet")[["timestamp", "close"]]
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.drop_duplicates("timestamp").sort_values("timestamp")
    return d.timestamp.values.astype("datetime64[ns]").astype(np.int64), d.close.values.astype(float)

blad_max = 0.0; zle = 0; n_test = 0
rng = np.random.default_rng(3)
for sym in ("BTC", "ETH", "SOL", "DOGE", "LINK", "XLM"):
    t1, c1 = wczytaj(sym, "1h")
    for tf, mn in (("4h", MIN_4H), ("1d", MIN_1D)):
        tt, ct = wczytaj(sym, tf)
        rsi, tr = tf_w_toku(c1, t1, ct, tt, mn)
        for i in rng.choice(np.arange(3000, len(c1)), 150, replace=False):
            b = np.searchsorted(tt, t1[i], side="right") - 1
            seria = list(ct[max(0, b - 299):b]) + [c1[i]]    # jak engine: limit 300
            if len(seria) >= mn:
                r_live = AIStrategy.calculate_rsi(None, seria, 14); t_live = _trend_jak_trening(seria)
            else:
                r_live, t_live = 50.0, 0.0
            blad_max = max(blad_max, abs(r_live - rsi[i])); n_test += 1
            if abs(r_live - rsi[i]) > 1e-6 or t_live != tr[i]:
                zle += 1
    rs = rsi_sma(c1, 14)
    for i in rng.choice(np.arange(20, len(c1)), 300, replace=False):
        rl = AIStrategy.calculate_rsi(None, list(c1[max(0, i - 299):i + 1]), 14)
        blad_1h = abs(rl - rs[i]); n_test += 1
        blad_max = max(blad_max, blad_1h)
        if blad_1h > 1e-6: zle += 1
print(f"[1] PARYTET: {n_test} porownan z kodem produkcji | max roznica RSI {blad_max:.2e} | niezgodnych {zle}")
ok1 = zle == 0

# [2] PRZECIEK: na wejsciach walidatora porownaj korelacje cechy z przeszloscia i przyszloscia
import sqlite3
c = sqlite3.connect("file:/root/ProjektHAI/HAI-NL/data/wyniki_kampanii_20260824/MAG-cvd_Z0_odniesienie.db?mode=ro", uri=True)
tx = pd.read_sql("SELECT symbol, open_ts FROM wfv_trade_log WHERE model_config='MAG-cvd'", c)
tx["s"] = tx.symbol.str.split("/").str[0].str.replace("USDT", "")
tx["t"] = pd.to_datetime(tx.open_ts, unit="ms").values.astype("datetime64[ns]").astype(np.int64)
R = []
for sym, g in tx.groupby("s"):
    try:
        t1, c1 = wczytaj(sym, "1h"); t4, c4 = wczytaj(sym, "4h"); td, cd = wczytaj(sym, "1d")
    except Exception:
        continue
    r4, tr4 = tf_w_toku(c1, t1, c4, t4, MIN_4H); rd, trd = tf_w_toku(c1, t1, cd, td, MIN_1D)
    pos = {v: k for k, v in enumerate(t1)}
    for t in g.t:
        i = pos.get(t)
        if i is None or i < 12 or i + 12 >= len(c1): continue
        R.append(dict(rsi_4h=r4[i], trend_4h=tr4[i], rsi_1d=rd[i], trend_1d=trd[i],
                      past3=c1[i]/c1[i-3]-1, fut3=c1[i+3]/c1[i]-1, past12=c1[i]/c1[i-12]-1, fut12=c1[i+12]/c1[i]-1))
d = pd.DataFrame(R)
print(f"[2] PRZECIEK: {len(d)} wejsc walidatora")
print(f"    {'cecha':<10}{'przeszl.3h':>11}{'PRZYSZL.3h':>11}{'przeszl.12h':>12}{'PRZYSZL.12h':>12}   werdykt")
ok2 = True
for k in ("rsi_4h", "trend_4h", "rsi_1d", "trend_1d"):
    a3, f3 = spearmanr(d[k], d.past3)[0], spearmanr(d[k], d.fut3)[0]
    a12, f12 = spearmanr(d[k], d.past12)[0], spearmanr(d[k], d.fut12)[0]
    zly = max(f3, f12) > 0.10 and max(f3, f12) > max(a3, a12)
    ok2 &= not zly
    print(f"    {k:<10}{a3:>+11.3f}{f3:>+11.3f}{a12:>+12.3f}{f12:>+12.3f}   {'PRZECIEK' if zly else 'ok'}")
print(f"\nWYNIK: parytet {'OK' if ok1 else 'BLAD'} | przeciek {'brak' if ok2 else 'WYKRYTY'}")
sys.exit(0 if (ok1 and ok2) else 1)
