#!/usr/bin/env python3
# ===========================================
# gen.Flow — estymator mapy likwidacji (DARMOWY, z OI+cena+ls_ratio)
# ===========================================
# Metoda jak Coinglass (free): nie mamy heatmapy, ale mozemy ja ESTYMOWAC.
# Nowe OI (delta) otwiera pozycje ~przy biezacej cenie; ls_ratio dzieli je na
# long/short; przy dzwigni L pozycja likwiduje sie ~1/L od wejscia (long w dol,
# short w gore). Akumulujemy "paliwo" likwidacji per poziom, usuwamy gdy cena
# przez nie przejdzie (zlikwidowane). To daje gdzie SIEDZI plynnosc longow/shortow.
#
# HIPOTEZA ZAMROZONA (przed patrzeniem na wynik, ICT/SMT: cena dazy do plynnosci):
#   "W oknie 48h cena czesciej siega WIEKSZEGO klastra likwidacji (magnes).
#    liq_imbalance = (masa_powyzej - masa_ponizej)/(suma) przewiduje kierunek:
#    dodatnia (wiecej short-liq powyzej) -> cena ciagnie W GORE (LONG-wynik)."
# Test: z-delta liq_imbalance i dist_to_cluster miedzy LONG a SHORT wynikiem 48h.
# ANTY-LEAKAGE: mapa liczona TYLKO z przeszlosci (OI/cena <= t), etykieta z ceny PO t.
# Hold-out: ostatnie 30% eventow (chronologicznie) NIE tykane do decyzji.
# ===========================================
import numpy as np
import pandas as pd
from pathlib import Path

WH = Path("/root/ProjektHAI/data_warehouse")
LEVERAGES = [(25, 0.25), (50, 0.35), (100, 0.40)]  # (dzwignia, waga udzialu) - retail high-lev
DECAY_H = 24 * 14         # paliwo "starzeje sie" po ~14 dniach (pozycje zamykane)
LOOKAHEAD = 48
TP_MULT, SL_MULT = 4.0, 1.0

def atr(h,l,c,n=14):
    tr=np.maximum(h[1:]-l[1:],np.maximum(abs(h[1:]-c[:-1]),abs(l[1:]-c[:-1])))
    return pd.Series(np.concatenate([[h[0]-l[0]],tr])).rolling(n,min_periods=1).mean().values

def tb_label(c,h,l,a,i,n):
    tpL,slL=c[i]+TP_MULT*a[i],c[i]-SL_MULT*a[i]; tpS,slS=c[i]-TP_MULT*a[i],c[i]+SL_MULT*a[i]
    lw=sw=ld=sd=False; first=None
    for j in range(i+1,min(i+1+LOOKAHEAD,n)):
        if not ld:
            if h[j]>=tpL: lw=True;ld=True;first=first or 'L'
            elif l[j]<=slL: ld=True
        if not sd:
            if l[j]<=tpS: sw=True;sd=True;first=first or 'S'
            elif h[j]>=slS: sd=True
        if ld and sd: break
    if lw and sw: return 1 if first=='L' else 2
    return 1 if lw else (2 if sw else 0)

def main():
    o=pd.read_parquet(WH/"ohlcv/binance/1h/BTC.parquet"); o["timestamp"]=pd.to_datetime(o["timestamp"])
    o=o.sort_values("timestamp").reset_index(drop=True)
    oi=pd.read_parquet(WH/"derivatives/open_interest/BTC.parquet"); oi["timestamp"]=pd.to_datetime(oi["timestamp"])
    ls=pd.read_parquet(WH/"derivatives/ls_ratio/BTC.parquet"); ls["timestamp"]=pd.to_datetime(ls["timestamp"])
    # okno: gdzie mamy ls_ratio (split long/short)
    start=ls["timestamp"].min()
    o=o[o["timestamp"]>=start].reset_index(drop=True)
    # merge OI (dzienne->ffill godzinowo) + ls_ratio (godzinowe)
    o=o.merge(oi[["timestamp","close"]].rename(columns={"close":"oi"}),on="timestamp",how="left")
    o["oi"]=o["oi"].ffill().bfill()
    o=o.merge(ls[["timestamp","ls_ratio"]],on="timestamp",how="left"); o["ls_ratio"]=o["ls_ratio"].ffill().bfill()
    c,h,l=o["close"].values,o["high"].values,o["low"].values
    oiv=o["oi"].values; lsr=o["ls_ratio"].values; a=atr(h,l,c); n=len(o)
    print(f"okno: {n} swiec ({str(o['timestamp'].min())[:10]} -> {str(o['timestamp'].max())[:10]})")

    # ledger likwidacji: lista (liq_price, mass, side, born_i). Budowany INKREMENTALNIE.
    ledger=[]
    feats=np.zeros((n,3))  # dist_below_atr, dist_above_atr, imbalance
    oi_prev=oiv[0]
    for i in range(n):
        # nowe OI (delta dodatnia = nowe pozycje) przy cenie c[i]
        d_oi=max(0.0, oiv[i]-oi_prev); oi_prev=oiv[i]
        if d_oi>0:
            long_frac=lsr[i]/(1+lsr[i])  # ls_ratio = long/short -> frac long
            for L,w in LEVERAGES:
                # longi likwiduja sie PONIZEJ c*(1-1/L); shorty POWYZEJ c*(1+1/L)
                ledger.append((c[i]*(1-1/L), d_oi*long_frac*w, 'long', i))
                ledger.append((c[i]*(1+1/L), d_oi*(1-long_frac)*w, 'short', i))
        # usun zlikwidowane (cena przeszla przez poziom) i przestarzale
        ledger=[(p,m,s,b) for (p,m,s,b) in ledger
                if (i-b)<DECAY_H and not (s=='long' and l[i]<=p) and not (s=='short' and h[i]>=p)]
        if ledger:
            below=[(p,m) for (p,m,s,b) in ledger if p<c[i]]
            above=[(p,m) for (p,m,s,b) in ledger if p>c[i]]
            mb=sum(m for _,m in below); ma=sum(m for _,m in above)
            nb=min((c[i]-p for p,_ in below), default=c[i])  # dystans do najblizszego ponizej
            na=min((p-c[i] for p,_ in above), default=c[i])
            feats[i,0]=nb/a[i] if a[i]>0 else 0
            feats[i,1]=na/a[i] if a[i]>0 else 0
            feats[i,2]=(ma-mb)/(ma+mb) if (ma+mb)>0 else 0  # imbalance: + = wiecej paliwa powyzej

    # etykieta 48h + test
    lab=np.array([tb_label(c,h,l,a,i,n) if i<n-LOOKAHEAD else 0 for i in range(n)])
    df=pd.DataFrame({"ts":o["timestamp"],"dist_below":feats[:,0],"dist_above":feats[:,1],
                     "imbalance":feats[:,2],"label":lab})
    df=df[(df["label"]!=0) & (df["imbalance"]!=0)].reset_index(drop=True)
    print(f"eventow kierunkowych z mapa: {len(df)} | LONG={int((df.label==1).sum())} SHORT={int((df.label==2).sum())}")

    print("\n=== z-delta: LONG-wynik vs SHORT-wynik (hipoteza: imbalance przewiduje kierunek) ===")
    L=df[df.label==1]; S=df[df.label==2]
    for f in ["imbalance","dist_below","dist_above"]:
        sd=df[f].std() or 1; z=abs(L[f].mean()-S[f].mean())/sd
        print(f"  {f:<12} z={z:.3f}  (LONG μ={L[f].mean():+.3f} SHORT μ={S[f].mean():+.3f})"+(" <<<" if z>=0.2 else ""))

    # kierunek imbalance: dodatnia -> hipoteza mowi LONG-wynik
    hi=df[df.imbalance>df.imbalance.quantile(0.7)]  # duzo paliwa powyzej
    lo=df[df.imbalance<df.imbalance.quantile(0.3)]  # duzo ponizej
    print(f"\n  imbalance WYSOKA (paliwo powyzej): LONG-wynik {100*(hi.label==1).mean():.1f}% (n={len(hi)})")
    print(f"  imbalance NISKA  (paliwo ponizej): LONG-wynik {100*(lo.label==1).mean():.1f}% (n={len(lo)})")
    print(f"  >>> spread kierunkowy: {100*((hi.label==1).mean()-(lo.label==1).mean()):+.1f} pkt (hipoteza: dodatni)")

    # drzewo z HOLD-OUT (ostatnie 30% nietkniete)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import precision_score
    df=df.sort_values("ts").reset_index(drop=True)
    X=df[["imbalance","dist_below","dist_above"]].values; y=(df.label==1).astype(int).values
    sp=int(len(df)*0.7); base=y[sp:].mean()
    m=HistGradientBoostingClassifier(max_depth=3,max_iter=120,learning_rate=0.05,min_samples_leaf=20,random_state=42)
    m.fit(X[:sp],y[:sp]); p=m.predict(X[sp:])
    print(f"\n=== DRZEWO hold-out (train {sp}, test {len(df)-sp}) ===")
    print(f"  baza LONG: {base:.3f} | precyzja_LONG={precision_score(y[sp:],p,zero_division=0):.3f} | acc={(p==y[sp:]).mean():.3f}")

if __name__=="__main__":
    main()
