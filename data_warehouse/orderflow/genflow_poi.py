#!/usr/bin/env python3
# ===========================================
# gen.Flow rev.2 — reakcja order-flow przy HTF POI (BTC)
# ===========================================
# Wg konsensusu bytow (DeepSeek+Grok, 2026-07-22): edge order-flow jest
# WARUNKOWY przy poziomach S/R, nie bezwarunkowa predykcja kierunku (tamto
# potwierdzono jako slepy zaulek: z-delta of_*<0.16). Tu modelujemy:
# "gdy cena dochodzi do HTF POI -> absorpcja(odwrocenie) czy agresja(przebicie)".
#
# ANTY-LEAKAGE (Grok, najwieksza pulapka):
#  - POI TYLKO z przeszlych, POTWIERDZONYCH piwotow (pivot t-k widoczny dopiero
#    t-k+N; touch w t uzywa poziomow potwierdzonych PRZED t).
#  - cechy tylko z danych <= t (touch). Etykieta z ceny PO t (target, OK).
#  - walk-forward po EVENTACH (chronologicznie), MALO cech (~8) na ~10^2-3 eventow.
#
# 1. piwoty swing N=3 -> strefy (klaster ATR*0.25), aktywne przy >=2 touchach.
# 2. touch event: cena wchodzi w aktywna strefe.
# 3. cechy OF wzgledem poziomu (CVD podejscia, delta/absorpcja w strefie, dystans).
# 4. etykieta: odbicie (>0.5 ATR w strone przeciwna do podejscia) vs przebicie, okno 12h.
# 5. drzewo (HistGB) walk-forward -> precyzja odbicia.
# ===========================================
import sys
from pathlib import Path
import numpy as np
import pandas as pd

WH = Path("/root/ProjektHAI/data_warehouse")
PIVOT_N = 3            # fraktal: high>3 przed i 3 po
ZONE_ATR = 0.25       # szerokosc klastrowania stref (x ATR)
MIN_TOUCHES = 2        # strefa aktywna przy >=2 historycznych dotknieciach
REACT_WINDOW = 12      # okno reakcji (h) po dotknieciu
REACT_ATR = 0.5        # prog odbicia/przebicia (x ATR)


def atr(h, l, c, n=14):
    tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
    tr = np.concatenate([[h[0]-l[0]], tr])
    return pd.Series(tr).rolling(n, min_periods=1).mean().values


def find_pivots(h, l, N):
    """Zwraca (idx, cena, typ) potwierdzone piwoty. Piwot w i potwierdzony w i+N."""
    piv = []
    for i in range(N, len(h)-N):
        if h[i] == max(h[i-N:i+N+1]):
            piv.append((i, h[i], "H", i+N))   # potwierdzony w i+N
        if l[i] == min(l[i-N:i+N+1]):
            piv.append((i, l[i], "L", i+N))
    return piv


def main():
    o = pd.read_parquet(WH/"ohlcv/binance/1h/BTC.parquet")
    o["timestamp"] = pd.to_datetime(o["timestamp"]); o = o.sort_values("timestamp").reset_index(drop=True)
    of = pd.read_parquet(WH/"orderflow/binance/BTC.parquet")
    of["timestamp"] = pd.to_datetime(of["timestamp"])
    of_cols = [x for x in of.columns if x.startswith("of_")]
    # ogranicz do okna gdzie mamy order-flow
    of_start = of["timestamp"].min()
    o = o[o["timestamp"] >= of_start - pd.Timedelta(days=30)].reset_index(drop=True)  # +30d na piwoty
    m = o.merge(of[["timestamp"]+of_cols], on="timestamp", how="left")
    for cc in of_cols: m[cc] = m[cc].fillna(0.0)
    c,h,l = m["close"].values, m["high"].values, m["low"].values
    a = atr(h,l,c); n = len(m)
    print(f"BTC 1h w oknie OF: {n} swiec ({str(m['timestamp'].min())[:10]} -> {str(m['timestamp'].max())[:10]})")

    piv = find_pivots(h, l, PIVOT_N)
    print(f"piwotow (potwierdzonych): {len(piv)}")

    # events: dla kazdej swiecy sprawdz czy dotyka aktywnej strefy (poziom z >=2
    # piwotow potwierdzonych PRZED ta swieca, w promieniu ZONE_ATR*atr)
    events = []
    for i in range(PIVOT_N+50, n-REACT_WINDOW):
        band = ZONE_ATR * a[i]
        # aktywne poziomy = ceny piwotow potwierdzonych przed i
        active = [pc for (pi,pc,pt,pconf) in piv if pconf < i]
        if not active: continue
        active = np.array(active)
        # czy cena swiecy i wchodzi w skupisko >=MIN_TOUCHES piwotow?
        lo, hi = l[i], h[i]
        # poziom = mediana piwotow w zasiegu ceny swiecy
        near = active[(active >= lo-band) & (active <= hi+band)]
        if len(near) < MIN_TOUCHES: continue
        level = float(np.median(near))
        approached_from = "below" if c[i-1] < level else "above"
        # cechy OF wzgledem poziomu (dane <= i)
        cvd_3h = m["of_delta"].values[max(0,i-3):i+1].sum()   # agresja podejscia
        delta_touch = m["of_delta"].values[i]
        absorp_touch = m["of_absorption"].values[i]
        big_touch = m["of_big_delta"].values[i]
        dist_atr = abs(c[i]-level)/a[i] if a[i]>0 else 0
        retest = int((np.abs(near-level) < band).sum())
        # etykieta: odbicie w strone PRZECIWNA do podejscia > REACT_ATR*atr w oknie
        fut_h = h[i+1:i+1+REACT_WINDOW]; fut_l = l[i+1:i+1+REACT_WINDOW]
        if approached_from == "below":   # podejscie do oporu z dolu -> odbicie = spadek
            bounce = (fut_l.min() <= level - REACT_ATR*a[i])
            breakout = (fut_h.max() >= level + REACT_ATR*a[i])
        else:                            # podejscie do wsparcia z gory -> odbicie = wzrost
            bounce = (fut_h.max() >= level + REACT_ATR*a[i])
            breakout = (fut_l.min() <= level - REACT_ATR*a[i])
        if not (bounce or breakout): continue  # chop -> odrzuc
        label = 1 if (bounce and not breakout) else (0 if breakout and not bounce else (1 if bounce else 0))
        events.append({"ts": m["timestamp"].values[i], "from": approached_from,
                       "cvd_3h": cvd_3h, "delta_touch": delta_touch, "absorp_touch": absorp_touch,
                       "big_touch": big_touch, "dist_atr": dist_atr, "retest": retest,
                       "label": label})
    ev = pd.DataFrame(events)
    print(f"\neventow przy POI: {len(ev)} | odbicia={int((ev['label']==1).sum())} przebicia={int((ev['label']==0).sum())}")
    if len(ev) < 50:
        print("ZA MALO eventow na sensowny test"); return

    # z-delta cech OF: odbicie vs przebicie
    feats = ["cvd_3h","delta_touch","absorp_touch","big_touch","dist_atr","retest"]
    print("\n=== z-delta cech przy POI: ODBICIE vs PRZEBICIE ===")
    B = ev[ev["label"]==1]; K = ev[ev["label"]==0]
    for f in feats:
        s = ev[f].std() or 1
        z = abs(B[f].mean()-K[f].mean())/s
        print(f"  {f:<14} z={z:.3f}  (odbicie μ={B[f].mean():+.3f} przebicie μ={K[f].mean():+.3f})" + (" <<<" if z>=0.2 else ""))

    # drzewo walk-forward po eventach
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import precision_score
    ev = ev.sort_values("ts").reset_index(drop=True)
    X = ev[feats].values; y = ev["label"].values
    split = int(len(ev)*0.7)
    base = y[split:].mean()  # baza (frakcja odbic w tescie)
    mdl = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.05,
                                         min_samples_leaf=15, random_state=42)
    mdl.fit(X[:split], y[:split])
    p = mdl.predict(X[split:])
    prec = precision_score(y[split:], p, zero_division=0)
    acc = (p==y[split:]).mean()
    print(f"\n=== DRZEWO walk-forward (train {split}, test {len(ev)-split}) ===")
    print(f"  baza odbic w tescie: {base:.3f} | acc={acc:.3f} | precyzja_ODBICIE={prec:.3f}")
    print(f"  >>> edge nad baza: {prec-base:+.3f} ({'JEST' if prec-base>0.05 else 'BRAK/slaby'})")

if __name__ == "__main__":
    main()
