#!/usr/bin/env python3
# ===========================================
# gen.Flow — krok 1+2: merge of_* + tani test drzewny (BTC-only)
# ===========================================
# Pytanie: czy order-flow (CVD/delta/absorpcja) niesie sygnal kierunkowy na
# 48h PONAD to, co daja cechy cenowe? Tani test PRZED neuralem.
# 1. BTC OHLCV 1h + etykieta 48h (triple-barrier TP4/SL1 ATR, jak ml_trainer).
# 2. Merge of_* (order-flow) po timestamp.
# 3. z-delta of_* miedzy wynikiem LONG a SHORT (czy agresja przewiduje kierunek).
# 4. Baseline RF: cechy cenowe SAME vs +of_* -> czy precyzja rosnie.
# ===========================================
import sys
from pathlib import Path
import numpy as np
import pandas as pd

WH = Path("/root/ProjektHAI/data_warehouse")
LOOKAHEAD, TP_MULT, SL_MULT = 48, 4.0, 1.0

def atr(h, l, c, n=14):
    tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
    tr = np.concatenate([[h[0]-l[0]], tr])
    return pd.Series(tr).rolling(n, min_periods=1).mean().values

def triple_barrier(c, h, l, a, i, n):
    tpL, slL = c[i]+TP_MULT*a[i], c[i]-SL_MULT*a[i]
    tpS, slS = c[i]-TP_MULT*a[i], c[i]+SL_MULT*a[i]
    lw=sw=False; ld=sd=False; first=None
    for j in range(i+1, min(i+1+LOOKAHEAD, n)):
        if not ld:
            if h[j]>=tpL: lw=True; ld=True; first=first or 'L'
            elif l[j]<=slL: ld=True
        if not sd:
            if l[j]<=tpS: sw=True; sd=True; first=first or 'S'
            elif h[j]>=slS: sd=True
        if ld and sd: break
    if lw and sw: return 1 if first=='L' else 2
    if lw: return 1
    if sw: return 2
    return 0

def main():
    # 1. BTC OHLCV 1h
    o = pd.read_parquet(WH/"ohlcv/binance/1h/BTC.parquet")
    o["timestamp"] = pd.to_datetime(o["timestamp"])
    o = o.sort_values("timestamp").reset_index(drop=True)
    c,h,l = o["close"].values, o["high"].values, o["low"].values
    a = atr(h,l,c)
    n = len(o)
    print(f"BTC OHLCV 1h: {n} swiec")

    # 2. etykieta 48h
    labels = np.zeros(n, dtype=int)
    for i in range(n-LOOKAHEAD):
        labels[i] = triple_barrier(c,h,l,a,i,n)
    o["label"] = labels
    o["ret_fwd"] = c  # placeholder
    dist = pd.Series(labels).value_counts().to_dict()
    print(f"rozklad etykiet 48h: NEUTRAL={dist.get(0,0)} LONG={dist.get(1,0)} SHORT={dist.get(2,0)}")

    # 3. merge of_*
    of = pd.read_parquet(WH/"orderflow/binance/BTC.parquet")
    of["timestamp"] = pd.to_datetime(of["timestamp"])
    of_cols = [x for x in of.columns if x.startswith("of_")]
    df = o.merge(of[["timestamp"]+of_cols], on="timestamp", how="inner")
    print(f"po merge z order-flow: {len(df)} swiec (pokrycie {100*len(df)/n:.0f}% OHLCV)")
    df = df[df["label"]!=0].copy()  # tylko kierunkowe wyniki
    print(f"swiec kierunkowych (LONG/SHORT): {len(df)}")

    # z-delta: LONG(1) vs SHORT(2) — czy of_* rozdziela kierunek
    print("\n=== z-delta of_* : LONG-wynik vs SHORT-wynik (im wyzej tym lepiej przewiduje kierunek) ===")
    L = df[df["label"]==1]; S = df[df["label"]==2]
    scored=[]
    for f in of_cols:
        wl,sl = L[f].values, S[f].values
        if len(wl)<20 or len(sl)<20: continue
        std = df[f].std() or 1
        z = abs(wl.mean()-sl.mean())/std
        scored.append((f,z, wl.mean(), sl.mean()))
    for f,z,lm,sm in sorted(scored,key=lambda x:-x[1]):
        flag = " <<< SYGNAL" if z>=0.2 else ""
        print(f"  {f:<20} z={z:.3f}  (LONG μ={lm:+.3f} SHORT μ={sm:+.3f}){flag}")

    # 4. baseline RF: cenowe SAME vs +of_*
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_score
    # proste cechy cenowe
    df2 = df.copy()
    df2["rsi_proxy"] = pd.Series(c).pct_change(14).reindex(df2.index).fillna(0).values[:len(df2)] if False else 0
    price_feats = []
    # policz kilka cech cenowych na df (z OHLCV)
    o["ret1"] = o["close"].pct_change().fillna(0)
    o["ret24"] = o["close"].pct_change(24).fillna(0)
    o["atr_pct"] = a/c
    o["hl_range"] = (h-l)/c
    pf = ["ret1","ret24","atr_pct","hl_range"]
    dfx = o.merge(of[["timestamp"]+of_cols], on="timestamp", how="inner")
    dfx = dfx[dfx["label"]!=0].dropna(subset=pf+of_cols)
    y = (dfx["label"]==1).astype(int).values  # LONG vs SHORT (binarne)
    split = int(len(dfx)*0.7)
    def evalset(cols, name):
        X = dfx[cols].values
        m = RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_leaf=20,
                                   random_state=42, n_jobs=-1, class_weight="balanced")
        m.fit(X[:split], y[:split])
        p = m.predict(X[split:])
        prec = precision_score(y[split:], p, zero_division=0)
        acc = (p==y[split:]).mean()
        print(f"  {name:<24} acc={acc:.3f} precision_LONG={prec:.3f} (n_test={len(y)-split})")
        return prec
    print("\n=== BASELINE RF (LONG vs SHORT, holdout 30%) ===")
    pp = evalset(pf, "cenowe SAME")
    po = evalset(of_cols, "order-flow SAME")
    pb = evalset(pf+of_cols, "cenowe + order-flow")
    print(f"\n  >>> order-flow dodaje: precyzja {pp:.3f} -> {pb:.3f} ({'+' if pb>pp else ''}{pb-pp:+.3f})")
    print(f"  >>> order-flow SAMO: {po:.3f} (vs cenowe {pp:.3f})")

if __name__ == "__main__":
    main()
