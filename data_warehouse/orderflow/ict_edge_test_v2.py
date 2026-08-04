#!/usr/bin/env python3
# ===========================================
# Wierne detektory ICT/SMC + decydujacy test edge (v2)
# ===========================================
# Silnik struktury (state machine): chronione swingi, BOS=przebicie zgodne z bias,
# CHoCH=przebicie przeciwne (zmiana bias). FVG z MITYGACJA: sygnal gdy cena wraca
# do niewypelnionej luki (reakcja), nie samo powstanie luki.
# Test: przy zdarzeniu sygnalu -> forward-return; z-delta bull vs bear + Spearman
# na kierunkowej etykiecie + block-permutation. Raportuje ZNAK (momentum vs contrarian).
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

WH = Path("/root/ProjektHAI/data_warehouse")
o = pd.read_parquet(WH/"ohlcv/binance/1h/BTC.parquet"); o["timestamp"]=pd.to_datetime(o["timestamp"])
o=o.sort_values("timestamp").reset_index(drop=True)
c=o["close"].values.astype(float); h=o["high"].values.astype(float); l=o["low"].values.astype(float)
n=len(o)
print(f"BTC 1h: {n} swiec ({str(o['timestamp'].min())[:10]}->{str(o['timestamp'].max())[:10]})")

# --- 1. potwierdzone swingi (fraktal N, potwierdzony w i+N) ---
N=3
piv=[]  # (idx_potwierdzenia, cena, typ 'H'/'L')
for i in range(N, n-N):
    if h[i]==max(h[i-N:i+N+1]): piv.append((i+N, h[i], 'H'))
    if l[i]==min(l[i-N:i+N+1]): piv.append((i+N, l[i], 'L'))
piv.sort()
# mapy: dla kazdego indeksu czasu -> lista nowo potwierdzonych piwotow
piv_at=[[] for _ in range(n)]
for conf,price,typ in piv:
    if conf<n: piv_at[conf].append((price,typ))

# --- 2. state machine struktury (BOS/CHoCH) ---
bias=0            # +1 bull, -1 bear, 0 neutral
key_high=np.nan   # swing high do przebicia (BOS w bull / CHoCH w bear)
key_low=np.nan    # swing low do przebicia (BOS w bear / CHoCH w bull)
bos=np.zeros(n); choch=np.zeros(n)
last_H=np.nan; last_L=np.nan
for i in range(n):
    # zaktualizuj swingi nowo potwierdzone w i
    for price,typ in piv_at[i]:
        if typ=='H': last_H=price; key_high=price
        else: last_L=price; key_low=price
    # sprawdz przebicia na close[i]
    if not np.isnan(key_high) and c[i]>key_high:
        if bias>=0:
            bos[i]=1                      # bull BOS (kontynuacja)
        else:
            choch[i]=1; bias=1            # CHoCH bear->bull (reversal up)
        key_high=np.nan                   # zuzyte az nowy swing high
        if bias==0: bias=1
    if not np.isnan(key_low) and c[i]<key_low:
        if bias<=0:
            bos[i]=-1                     # bear BOS
        else:
            choch[i]=-1; bias=-1          # CHoCH bull->bear (reversal down)
        key_low=np.nan
        if bias==0: bias=-1

# --- 3. FVG z mitygacja ---
# aktywne luki: (top, bottom, dir, born). Sygnal gdy cena wchodzi w luke.
fvg_zones=[]
fvg_mit=np.zeros(n)  # +1 bull FVG mitygowany (oczekuj gora), -1 bear
for i in range(n):
    # nowe FVG na i (3-swiecowe, i-2..i)
    if i>=2:
        if l[i]>h[i-2]:  fvg_zones.append([l[i], h[i-2], 1, i])   # bull gap [bottom=h[i-2], top=l[i]]
        elif h[i]<l[i-2]: fvg_zones.append([l[i-2], h[i], -1, i]) # bear gap [bottom=h[i], top=l[i-2]]
    # mitygacja: cena wchodzi w niewypelniona luke (nie ta swiezo utworzona)
    still=[]
    fired=0
    for z in fvg_zones:
        top,bot,dr,born=z
        if i-born<1: still.append(z); continue
        if dr==1:   # bull FVG: mityguje gdy low wchodzi w [bot,top]
            if l[i]<=top and l[i]>=bot:
                if not fired: fvg_mit[i]=1; fired=1
                still.append(z)              # zostaje az pelne wypelnienie
            elif l[i]<bot:
                pass                          # w pelni wypelniona -> usun
            else: still.append(z)
        else:       # bear FVG: mityguje gdy high wchodzi w [bot,top]
            if h[i]>=bot and h[i]<=top:
                if not fired: fvg_mit[i]=-1; fired=1
                still.append(z)
            elif h[i]>top: pass
            else: still.append(z)
    fvg_zones=still[-200:]  # cap

# --- 4. test edge: z-delta bull vs bear + Spearman + block-perm ---
def fwd(H):
    r=np.full(n,np.nan); r[:n-H]=np.log(c[H:]/c[:n-H]); return r
def blockp(x,y,rho,valid,B=24,NP=800):
    yv=y.copy(); L=len(yv); nb=int(np.ceil(L/B)); rng=np.random.default_rng(11); cnt=0; xv=x[valid]
    for _ in range(NP):
        order=rng.permutation(nb)
        yp=np.concatenate([yv[b*B:(b+1)*B] for b in order])[:L]
        if abs(spearmanr(xv,yp[valid]).statistic)>=abs(rho): cnt+=1
    return (cnt+1)/(NP+1)

HZ=[3,6,12,24,48]
sigs={"BOS":bos,"CHoCH":choch,"FVG_mit":fvg_mit}
for name,sig in sigs.items():
    nb=int((sig>0).sum()); ns=int((sig<0).sum())
    print(f"\n=== {name}: bull={nb} bear={ns} zdarzen ===")
    print(f"{'H':>4}{'rho':>8}{'p':>8}  bull_ret  bear_ret  znak")
    for H in HZ:
        y=fwd(H); valid=~np.isnan(y)&(sig!=0)
        if valid.sum()<80: continue
        rho=spearmanr(sig[valid],y[valid]).statistic
        br=np.nanmean(y[(sig>0)]); sr=np.nanmean(y[(sig<0)])
        znak="momentum" if rho>0 else "KONTRARIAN"
        p=blockp(sig,y,rho,valid) if abs(rho)>=0.03 else 1.0
        tag="  <<<" if p<0.10 and abs(rho)>=0.03 else ""
        print(f"{H:>4}{rho:>8.3f}{p:>8.3f}  {br:+.4f}  {sr:+.4f}  {znak}{tag}")
