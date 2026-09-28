#!/usr/bin/env python3
"""CO NASZE CECHY W OGOLE UMIEJA — kierunek vs wielkosc ruchu (2026-09-28).

Pytanie: skoro modele nie przewiduja kierunku (AUC 0,51, 11 ksztaltow 0/11, 7 celow 0/7),
to czy ich cechy umieja cokolwiek innego?

Katalog cech sugerowal: te same cechy maja znikoma moc przy KIERUNKU (najlepsza 0,061),
ale duza przy WIELKOSCI ruchu (atr_pct 0,348, bb_bandwidth_pct 0,261). To bylaby zupelnie
inna zdolnosc — i inna uzytecznosc.

Mierzymy trzy rzeczy dla kazdej cechy produkcyjnej:
  KIERUNEK    IC przekrojowe wobec zwrotu minus srednia rynku
  WIELKOSC    IC wobec |zwrot| (bezwzglednej wielkosci ruchu)
  WIELKOSC+   czy przewiduje wielkosc PONAD sama zmiennosc (reszta po regresji na atr_pct)

Ostatnia kolumna jest kluczowa: przewidywanie wielkosci ze zmiennosci jest prawie tautologia.
Pytanie brzmi, czy cokolwiek dodaje ponad nia.

Probkowanie co 6 h (bez nakladania), 2021-08..2026-09.
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get("HAI_ROOT", "/root/ProjektHAI")
HZ = 6

CECHY_PROD = [
    "atr_pct", "bb_bandwidth_pct", "rsi", "momentum", "price_position_bb",
    "ema_mid_r", "ema_slow_r", "adx_14", "volume_ratio", "volume_zscore",
    "x_roc_24", "x_roc_168", "x_rsi_7", "r_atr_zscore_20", "r_range_compression",
    "btc_corr_24h", "btc_beta_72h", "rank_mom_24h", "sr_dist_pct", "fib_dist_pct",
    "e_ichimoku_cloud_thickness",
]


def licz_cechy(C, H, L, V):
    D = {}
    D["atr_pct"] = (H.rolling(14).max() - L.rolling(14).min()) / C * 100
    sr = C.rolling(20).mean(); od = C.rolling(20).std()
    D["bb_bandwidth_pct"] = (4 * od) / sr * 100
    D["price_position_bb"] = (C - (sr - 2 * od)) / (4 * od).replace(0, np.nan)
    d = C.diff()
    zy = d.clip(lower=0).rolling(14).mean(); st = (-d.clip(upper=0)).rolling(14).mean()
    D["rsi"] = 100 - 100 / (1 + zy / st.replace(0, np.nan))
    zy7 = d.clip(lower=0).rolling(7).mean(); st7 = (-d.clip(upper=0)).rolling(7).mean()
    D["x_rsi_7"] = 100 - 100 / (1 + zy7 / st7.replace(0, np.nan))
    D["momentum"] = (C / C.shift(10) - 1) * 100
    D["ema_mid_r"] = (C / C.ewm(span=21, adjust=False).mean() - 1) * 100
    D["ema_slow_r"] = (C / C.ewm(span=55, adjust=False).mean() - 1) * 100
    D["x_roc_24"] = (C / C.shift(24) - 1) * 100
    D["x_roc_168"] = (C / C.shift(168) - 1) * 100
    D["volume_ratio"] = V / V.rolling(24).mean().replace(0, np.nan)
    D["volume_zscore"] = (V - V.rolling(168).mean()) / V.rolling(168).std()
    D["r_atr_zscore_20"] = (D["atr_pct"] - D["atr_pct"].rolling(20).mean()) / D["atr_pct"].rolling(20).std()
    zak6 = H.rolling(6).max() - L.rolling(6).min()
    zak48 = H.rolling(48).max() - L.rolling(48).min()
    D["r_range_compression"] = zak6 / zak48.replace(0, np.nan)
    # Korelacja i beta z SUM KROCZACYCH. rolling().corr(Series) na ramce ze 134 kolumnami
    # liczy sie kolumna po kolumnie i trwa kwadranse — zle dobrana metoda (zlapane 28.09).
    r = C.pct_change()
    rb = r["BTC"] if "BTC" in r.columns else r.mean(axis=1)
    for okno, nazwa in ((24, "btc_corr_24h"), (72, "btc_beta_72h")):
        sx = r.rolling(okno).sum()
        sy = rb.rolling(okno).sum()
        sxy = r.mul(rb, axis=0).rolling(okno).sum()
        syy = (rb ** 2).rolling(okno).sum()
        kow = sxy.div(okno) - sx.div(okno).mul(sy.div(okno), axis=0)
        war_y = syy.div(okno) - (sy.div(okno)) ** 2
        if nazwa == "btc_beta_72h":
            D[nazwa] = kow.div(war_y.replace(0, np.nan), axis=0)
        else:
            sxx = (r ** 2).rolling(okno).sum()
            war_x = sxx.div(okno) - (sx.div(okno)) ** 2
            mian = np.sqrt(war_x).mul(np.sqrt(war_y), axis=0)
            D[nazwa] = kow / mian.replace(0, np.nan)
    D["rank_mom_24h"] = ((C / C.shift(24) - 1)).rank(axis=1, pct=True)
    D["sr_dist_pct"] = (C - L.rolling(48).min()) / C * 100
    D["fib_dist_pct"] = (H.rolling(48).max() - C) / (H.rolling(48).max() - L.rolling(48).min()).replace(0, np.nan)
    ich = (H.rolling(26).max() + L.rolling(26).min()) / 2 - (H.rolling(52).max() + L.rolling(52).min()) / 2
    D["e_ichimoku_cloud_thickness"] = ich / C * 100
    D["adx_14"] = (C.diff().abs().rolling(14).mean() / C * 100)
    return D


def ic(X, Y, idx):
    Xa, Ya = X.reindex(idx), Y.reindex(idx)
    w = []
    for t in idx:
        x, y = Xa.loc[t], Ya.loc[t]
        m = x.notna() & y.notna()
        if m.sum() > 20 and x[m].nunique() > 5:
            w.append(x[m].corr(y[m], method="spearman"))
    w = pd.Series(w, index=idx[:len(w)]).dropna()
    return w.mean() if len(w) else np.nan


def main():
    C, H, L, V = {}, {}, {}, {}
    for p in sorted(glob.glob(f"{ROOT}/data_warehouse/ohlcv/binance/1h/*.parquet")):
        s = os.path.basename(p)[:-8]
        o = pd.read_parquet(p, columns=["timestamp", "high", "low", "close", "volume"])
        o["timestamp"] = pd.to_datetime(o.timestamp).dt.tz_localize(None)
        o = o.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        C[s], H[s], L[s], V[s] = o.close, o.high, o.low, o.volume
    C = pd.DataFrame(C).sort_index()
    H = pd.DataFrame(H).reindex(C.index); L = pd.DataFrame(L).reindex(C.index)
    V = pd.DataFrame(V).reindex(C.index)
    C = C[C.index >= "2021-08-01"]
    H, L, V = H.reindex(C.index), L.reindex(C.index), V.reindex(C.index)
    D = licz_cechy(C, H, L, V)
    R = (C.shift(-HZ) / C - 1) * 100
    RS = R.sub(R.mean(axis=1), axis=0)
    A = R.abs()
    idx = C.index[::HZ]
    print(f"monet {C.shape[1]} | okresow {len(idx):,} | horyzont {HZ} h\n", flush=True)

    # reszta wielkosci po zmiennosci: |zwrot| minus to, co tlumaczy atr_pct
    atr = D["atr_pct"]
    rank_atr = atr.rank(axis=1, pct=True)
    rank_A = A.rank(axis=1, pct=True)
    RESZTA = rank_A - rank_atr        # ile ruchu PONAD to, co sugeruje zmiennosc

    print("=" * 96)
    print("CO UMIEJA CECHY PRODUKCYJNE")
    print("=" * 96)
    print(f"  {'cecha':<28} {'KIERUNEK':>10} {'WIELKOSC':>10} {'WIELKOSC ponad zmiennosc':>26}")
    W = []
    for nm in CECHY_PROD:
        if nm not in D:
            continue
        X = D[nm]
        kier = ic(X, RS, idx)
        wiel = ic(X, A, idx)
        wiel_p = ic(X, RESZTA, idx)
        W.append((nm, kier, wiel, wiel_p))
    for nm, k, w, wp in sorted(W, key=lambda x: -abs(x[2])):
        print(f"  {nm:<28} {k:>+10.4f} {w:>+10.4f} {wp:>+26.4f}")

    kk = np.array([abs(x[1]) for x in W])
    ww = np.array([abs(x[2]) for x in W])
    wwp = np.array([abs(x[3]) for x in W])
    print(f"\n  SREDNIA |IC|:  kierunek {kk.mean():.4f}  |  wielkosc {ww.mean():.4f}  "
          f"|  wielkosc ponad zmiennosc {wwp.mean():.4f}")
    print(f"  NAJLEPSZA:     kierunek {kk.max():.4f}  |  wielkosc {ww.max():.4f}  "
          f"|  ponad zmiennosc {wwp.max():.4f}")
    print(f"\n  ILORAZ: cechy sa {ww.mean()/max(kk.mean(),1e-9):.1f}x mocniejsze przy WIELKOSCI niz przy KIERUNKU")


if __name__ == "__main__":
    main()
