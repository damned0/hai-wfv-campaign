#!/usr/bin/env python3
# ===========================================
# Decydujacy test edge: FVG / BOS / CHoCH (struktura ICT) — mierz PRZED budowa
# ===========================================
# Ta sama dyscyplina co of_decisive_test: czy sygnaly strukturalne przewiduja
# kierunek forward-return na wielu horyzontach i przezywaja block-permutation.
# Detektory uproszczone ale wierne definicjom:
#  - FVG: 3-swiecowa luka (bullish: low[i] > high[i-2]; bearish: high[i] < low[i-2])
#  - swing piwoty (fraktal N=3) -> last_swing_high/low
#  - BOS: close przebija ostatni swing W KIERUNKU trendu (kontynuacja)
#  - CHoCH: close przebija ostatni swing PRZECIW trendowi (pierwsze odwrocenie)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

WH = Path("/root/ProjektHAI/data_warehouse")
o = pd.read_parquet(WH/"ohlcv/binance/1h/BTC.parquet"); o["timestamp"]=pd.to_datetime(o["timestamp"])
o = o.sort_values("timestamp").reset_index(drop=True)
c=o["close"].values.astype(float); h=o["high"].values.astype(float); l=o["low"].values.astype(float)
n=len(o)
print(f"BTC 1h: {n} swiec ({str(o['timestamp'].min())[:10]}->{str(o['timestamp'].max())[:10]})")

# --- FVG bias (signed): netto FVG w ostatnich 12 barach ---
fvg = np.zeros(n)
for i in range(2, n):
    if l[i] > h[i-2]: fvg[i] = 1     # bullish FVG (luka w gore)
    elif h[i] < l[i-2]: fvg[i] = -1  # bearish FVG
fvg_bias = pd.Series(fvg).rolling(12, min_periods=1).sum().values  # netto ostatnie 12

# --- swing piwoty (potwierdzone: pivot w i widoczny w i+N) ---
N=3
swing_hi=np.full(n,np.nan); swing_lo=np.full(n,np.nan)
for i in range(N, n-N):
    if h[i]==max(h[i-N:i+N+1]): swing_hi[i+N]=h[i]  # potwierdzony w i+N
    if l[i]==min(l[i-N:i+N+1]): swing_lo[i+N]=l[i]
# ostatni potwierdzony swing high/low (ffill)
last_sh=pd.Series(swing_hi).ffill().values
last_sl=pd.Series(swing_lo).ffill().values

# --- trend struktury: HH/HL (up) vs LH/LL (down) po ostatnich swingach ---
# uproszczenie: kierunek = znak nachylenia last_sh w oknie 48h
sh_slope = pd.Series(last_sh).diff(24).values
trend_struct = np.sign(np.nan_to_num(sh_slope))  # +1 struktura rosnaca

# --- BOS / CHoCH ---
bos=np.zeros(n); choch=np.zeros(n)
for i in range(1,n):
    if np.isnan(last_sh[i]) or np.isnan(last_sl[i]): continue
    bull_break = c[i]>last_sh[i] and c[i-1]<=last_sh[i]  # swieze przebicie w gore
    bear_break = c[i]<last_sl[i] and c[i-1]>=last_sl[i]
    if bull_break:
        if trend_struct[i]>=0: bos[i]=1        # przebicie zgodne z trendem = BOS
        else: choch[i]=1                        # przebicie przeciw trendowi = CHoCH (reversal up)
    if bear_break:
        if trend_struct[i]<=0: bos[i]=-1
        else: choch[i]=-1

def fwd(H):
    r=np.full(n,np.nan); r[:n-H]=np.log(c[H:]/c[:n-H]); return r

def blockp(x,y,rho,valid,B=24,NP=800):
    yv=y.copy(); L=len(yv); nb=int(np.ceil(L/B)); rng=np.random.default_rng(7); cnt=0; xv=x[valid]
    for _ in range(NP):
        order=rng.permutation(nb)
        yp=np.concatenate([yv[b*B:(b+1)*B] for b in order])[:L]
        if abs(spearmanr(xv,yp[valid]).statistic)>=abs(rho): cnt+=1
    return (cnt+1)/(NP+1)

HZ=[6,12,24,48]
signals={"fvg_bias":fvg_bias, "bos":bos, "choch":choch}
print("\n=== monotonicznosc sygnal -> forward-return (Spearman rho, p block-perm) ===")
print(f"{'sygnal':<10}{'H':>4}{'n_nonzero':>10}{'rho':>8}{'p':>8}  edge?")
hits=[]
for name,sig in signals.items():
    nz=int((sig!=0).sum())
    for H in HZ:
        y=fwd(H); valid=~np.isnan(y)&~np.isnan(sig)&(sig!=0)  # tylko gdzie sygnal aktywny
        if valid.sum()<100: continue
        rho=spearmanr(sig[valid],y[valid]).statistic
        if abs(rho)<0.04:
            print(f"{name:<10}{H:>4}{int(valid.sum()):>10}{rho:>8.3f}{'--':>8}"); continue
        p=blockp(sig,y,rho,valid)
        edge=p<0.10
        print(f"{name:<10}{H:>4}{int(valid.sum()):>10}{rho:>8.3f}{p:>8.3f}"+("  <<<" if edge else ""))
        if edge: hits.append((name,H,rho,p))

print("\n=== WERDYKT ===")
if hits:
    print(f"ZNALEZIONO {len(hits)} przezywajacych block-perm:")
    for name,H,rho,p in sorted(hits,key=lambda z:z[3]):
        print(f"  {name} @ {H}h: rho={rho:+.3f} p={p:.3f}")
    print(">>> ktoras struktura ICT ma edge -> warto wpinac jako ceche (z dyscyplina)")
else:
    print("BRAK sygnalu ICT przezywajacego block-perm.")
    print(">>> FVG/BOS/CHoCH w tej prostej formie NIE daja kierunkowego edge na BTC 1h.")
