#!/usr/bin/env python3
"""Kontrola zbioru przed wnioskami (2026-09-14): czy w danych nie ma bubla.

Sprawdza zbior przygotowany (cechy + wyniki transakcji r_*) i swiece z magazynu:
 1. ciaglosc: duplikaty godzin, dziury w czasie, liczba wierszy per symbol;
 2. ceny: skoki |zwrot 1h| > 40% (zmiana mnoznika / blad danych), zamrozone ceny (bez zmiany >= 24 h);
 3. cechy: udzial NaN i nieskonczonosci, cechy stale, wartosci skrajne;
 4. koszty: ATR bliski zera (koszt w ATR rosnie do nieskonczonosci);
 5. wyniki transakcji r_*: udzial celu / stopu / zamkniecia po czasie per geometria;
 6. przeciek: korelacja cechy z PRZYSZLYM zwrotem 24 h vs z PRZESZLYM (jak tools/skan_przecieku_pelny.py).
"""
import argparse, os, sys, re, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poszukiwania as PZ

ap = argparse.ArgumentParser(); ap.add_argument("--przygotowany", required=True); a = ap.parse_args()
wynik_kol = re.compile(r"^r_(\d+\.\d+)_(\d+\.\d+)_(\d+)_([LS])$")
import pyarrow.parquet as pq
kol = pq.read_schema(a.przygotowany).names
cechy = [k for k in kol if not wynik_kol.match(k) and not k.startswith(("_", "label_", "trade_", "cel_", "__"))
         and k not in ("timestamp", "symbol", "close")]
rk = [k for k in kol if wynik_kol.match(k)]
Z = pd.read_parquet(a.przygotowany, columns=["symbol", "_t", "_koszt_atr", "atr_pct"] + [c for c in cechy if c != "atr_pct"] + rk)
print(f"ZBIOR: {len(Z)} wierszy, {Z.symbol.nunique()} symboli, {len(cechy)} cech, {len(rk)} kolumn wynikow, "
      f"{Z._t.min()} -> {Z._t.max()}")
print(f"  makro wyciete obecne w zbiorze: {[c for c in PZ.MAKRO_USUNIETE if c in cechy]} (stary zbior — narzedzia je filtruja)")

# 1. ciaglosc
dup = Z.duplicated(["symbol", "_t"]).sum()
g = Z.sort_values(["symbol", "_t"]).groupby("symbol")["_t"]
dz = g.diff().dt.total_seconds().div(3600)
print(f"\n1. CIAGLOSC: duplikatow (symbol, godzina): {dup} | dziur > 3 h: {(dz > 3).sum()} "
      f"| najwieksza dziura: {dz.max():.0f} h | wierszy per symbol min/mediana: {g.size().min()}/{int(g.size().median())}")
zl = dz[dz > 72].groupby(Z.sort_values(['symbol', '_t'])['symbol']).size().sort_values(ascending=False).head(5)
if len(zl): print("   symbole z dziurami > 3 dni:", zl.to_dict())

# 2. ceny ze swiec magazynu
skoki, zamr = {}, {}
for s in Z.symbol.unique():
    o = pd.read_parquet(f"{PZ.ROOT}/data_warehouse/ohlcv/binance/1h/{s}.parquet", columns=["timestamp", "close"])
    c = o.close.values; r = np.abs(np.diff(c) / c[:-1])
    n = int((r > 0.4).sum())
    if n: skoki[s] = n
    stale = (np.diff(c) == 0).astype(int)
    najdl, bieg = 0, 0
    for x in stale:
        bieg = bieg + 1 if x else 0; najdl = max(najdl, bieg)
    if najdl >= 24: zamr[s] = najdl
print(f"\n2. CENY: symboli ze skokiem |1h| > 40%: {len(skoki)} {dict(list(skoki.items())[:8])}")
print(f"   symboli z zamrozona cena >= 24 h: {len(zamr)} {dict(sorted(zamr.items(), key=lambda x: -x[1])[:8])}")

# 3. cechy
pr = Z.sample(min(500_000, len(Z)), random_state=1)
tab = []
for c in cechy:
    x = pd.to_numeric(pr[c], errors="coerce")
    inf = np.isinf(x).mean(); x = x.replace([np.inf, -np.inf], np.nan)
    med, iqr = x.median(), x.quantile(0.75) - x.quantile(0.25)
    skr = ((x - med).abs() > 50 * (iqr if iqr > 0 else 1)).mean()
    tab.append((c, x.isna().mean(), inf, x.nunique(), skr))
T = pd.DataFrame(tab, columns=["cecha", "nan", "inf", "unikalnych", "skrajne"])
print(f"\n3. CECHY: z NaN > 20%: {list(T[T.nan > 0.2].cecha)} | z nieskonczonoscia: {list(T[T.inf > 0].cecha)}")
print(f"   stale (<= 2 wartosci): {list(T[T.unikalnych <= 2].cecha)} | ze skrajnymi (>50 IQR) > 0.5%: "
      f"{list(T[T.skrajne > 0.005].cecha)}")

# 4. koszty
print(f"\n4. KOSZTY: ATR% mediana {Z.atr_pct.median():.3f}, 1% najnizszych < {Z.atr_pct.quantile(0.01):.3f} | "
      f"wierszy z ATR% < 0.05 (koszt obciety): {(Z.atr_pct < 0.05).mean():.2%} | koszt w ATR: mediana "
      f"{Z._koszt_atr.median():.3f}, 99. percentyl {Z._koszt_atr.quantile(0.99):.3f}")

# 5. wyniki transakcji
print("\n5. WYNIKI r_*: (cel / stop / po czasie) — sanity: wiekszy cel = rzadziej trafiony")
for k in rk:
    tp, sl, hz, st = wynik_kol.match(k).groups(); tp, sl = float(tp), float(sl)
    v = Z[k].dropna()
    print(f"   {k:18s} cel {np.isclose(v, tp).mean():.1%} | stop {np.isclose(v, -sl).mean():.1%} | "
          f"po czasie {(~np.isclose(v, tp) & ~np.isclose(v, -sl)).mean():.1%} | srednio {v.mean():+.3f} ATR")

# 6. przeciek: korelacja z przyszloscia vs przeszloscia (zwrot 24 h)
czesci = []
for s in pr.symbol.unique()[:40]:
    o = pd.read_parquet(f"{PZ.ROOT}/data_warehouse/ohlcv/binance/1h/{s}.parquet", columns=["timestamp", "close"])
    o["_t"] = pd.to_datetime(o.timestamp).astype("datetime64[ns]"); o = o.drop_duplicates("_t").set_index("_t").sort_index()
    c = o.close
    d = pd.DataFrame({"symbol": s, "_t": c.index, "przysz": c.reindex(c.index + pd.Timedelta(hours=24)).values / c.values - 1,
                      "przesz": c.values / c.reindex(c.index - pd.Timedelta(hours=24)).values - 1})
    czesci.append(d)
F = pd.concat(czesci); pr2 = pr.copy(); pr2["_t"] = pr2["_t"].astype("datetime64[ns]")
M = pr2.merge(F, on=["symbol", "_t"], how="inner")
podejrzane = []
for c in cechy:
    x = pd.to_numeric(M[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.nunique() < 3: continue
    kp = x.corr(M.przysz, method="spearman"); kw = x.corr(M.przesz, method="spearman")
    if abs(kp) > 0.08 and abs(kp) > abs(kw):
        podejrzane.append((c, round(kp, 3), round(kw, 3)))
print(f"\n6. PRZECIEK (Spearman z przyszlym vs przeszlym zwrotem 24 h, {len(M)} wierszy): "
      f"{'brak podejrzanych cech' if not podejrzane else podejrzane}")
