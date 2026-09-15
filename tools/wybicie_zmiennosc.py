#!/usr/bin/env python3
"""Wybicie z bramka zmiennosci (2026-09-15, konsultacja: 5/6 modeli) — kierunek wybiera rynek, model tylko "kiedy".

Cechy dobrze przewiduja WIELKOSC ruchu, slabo kierunek. Tu model (LGB, WFV 12x45, uczony przed oknem - 7 dni embarga)
przewiduje ekspansje: (max high - min low w nastepnych 12 h) / ATR. Gdy prognoza w gornych --gorne (10%) wzgledem
prognoz z poprzednich 7 dni (przyczynowo): zlecenia stop OCO — kupno nad max(high 12 h) + 0,1 ATR, sprzedaz pod
min(low 12 h) - 0,1 ATR, wazne 6 h; pierwsze wypelnione anuluje drugie. Po wejsciu: stop 0,8 ATR, cel 1,8 ATR, max 12 h.
Ostroznie: wejscie stop +0,05% poslizgu ponad koszt 0,22%; swieca z obiema stronami OCO albo z celem i stopem = strata;
w swiecy wejscia liczy sie tylko zamkniecie (jej high/low bylo czesciowo przed wejsciem) — poprawka 15.09 22:00.
Kontrola: ten sam mechanizm BEZ bramki (co --co-ile h na kazdym coinie) — czy bramka cos daje.
"""
import argparse, os, sys, time, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--zbior", required=True); ap.add_argument("--katalog", required=True)
ap.add_argument("--koniec", default="2026-08-14 00:00:00"); ap.add_argument("--okna", type=int, default=12)
ap.add_argument("--dni", type=int, default=45); ap.add_argument("--wiersze", type=int, default=800_000)
ap.add_argument("--gorne", type=float, default=0.10); ap.add_argument("--co-ile", type=int, default=12)
a = ap.parse_args()
ROOT = os.environ.get("HAI_ROOT", "/root/ProjektHAI")
sys.path.insert(0, ROOT if os.path.exists(f"{ROOT}/hai_common/ml_trainer.py") else f"{ROOT}/hai_common")
os.makedirs(a.katalog, exist_ok=True)
KOSZT, POSL = 0.22, 0.05
OKNO_Z, WAZNE, SL, TP, MAXH, BUF = 12, 6, 0.8, 1.8, 12, 0.1
MARTWE = ("WAVES", "OCEAN", "FTM", "OMG", "MKR", "TON", "ICX", "STORJ")
MAKRO = ("gold_chg", "oil_wti_chg", "sp500_chg", "vix_chg", "us10y_chg", "dxy_chg", "fear_greed", "btc_dominance_chg")
koniec = pd.Timestamp(a.koniec); starty = [koniec - pd.Timedelta(days=a.dni * k) for k in range(a.okna, 0, -1)]
t0 = time.time()
import hai_common.ml_trainer as mt
if not os.path.exists(a.zbior):
    mt.build_dataset().to_parquet(a.zbior, index=False)
Z = pd.read_parquet(a.zbior); Z = Z[~Z.symbol.isin(MARTWE)].copy()
Z["_t"] = pd.to_datetime(Z.timestamp).dt.tz_localize(None).astype("datetime64[ns]")
SW = {}
cz = []
for s, g in Z.groupby("symbol"):
    o = pd.read_parquet(f"{ROOT}/data_warehouse/ohlcv/binance/1h/{s}.parquet", columns=["timestamp", "high", "low", "close"])
    o["timestamp"] = pd.to_datetime(o.timestamp).dt.tz_localize(None).astype("datetime64[ns]")
    o = o.drop_duplicates("timestamp").set_index("timestamp"); o = o.reindex(pd.date_range(o.index[0], o.index[-1], freq="h"))
    SW[s] = o
    h, l = o.high, o.low
    fut = (h[::-1].rolling(OKNO_Z, min_periods=OKNO_Z).max()[::-1].shift(-1) - l[::-1].rolling(OKNO_Z, min_periods=OKNO_Z).min()[::-1].shift(-1))
    g = g.assign(close_=o.close.reindex(g._t.values).values, fut_zakres=fut.reindex(g._t.values).values)
    cz.append(g)
Z = pd.concat(cz, ignore_index=True)
Z["atr_abs"] = Z.atr_pct / 100 * Z.close_
Z["y"] = np.log1p((Z.fut_zakres / Z.atr_abs).clip(0, 50))
Z = Z[Z.y.notna() & (Z.atr_abs > 0)].reset_index(drop=True)
cechy = [c for c in Z.columns if c not in ("symbol", "timestamp", "_t", "y", "fut_zakres", "close_", "atr_abs", "close")
         and not c.startswith("label") and c not in MAKRO and pd.api.types.is_numeric_dtype(Z[c]) and Z[c].nunique() > 5]
X = Z[cechy].astype(np.float32).replace([np.inf, -np.inf], np.nan)
print(f"zbior {len(Z)} wierszy, {Z.symbol.nunique()} symboli, {len(cechy)} cech ({time.time() - t0:.0f} s)", flush=True)

import lightgbm as lgb
P = []
for nr, st in enumerate(starty, 1):
    kon = st + pd.Timedelta(days=a.dni)
    tr = np.where(Z._t < st - pd.Timedelta(days=7))[0]
    if len(tr) > a.wiersze: tr = np.sort(np.random.default_rng(nr).choice(tr, a.wiersze, replace=False))
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=2000, subsample=0.7, subsample_freq=1,
                          colsample_bytree=0.6, reg_lambda=10.0, verbose=-1, n_jobs=4).fit(X.iloc[tr], Z.y.values[tr])
    te = np.where((Z._t >= st - pd.Timedelta(days=7)) & (Z._t < kon))[0]     # 7 dni rozbiegu progu
    Q = Z.iloc[te][["symbol", "_t", "atr_abs", "y"]].copy(); Q["pred"] = m.predict(X.iloc[te]); Q["okno"] = nr; Q["w_oknie"] = Q._t >= st
    from scipy.stats import spearmanr
    w = Q[Q.w_oknie]; print(f"okno {nr}: korelacja prognozy ekspansji z faktyczna {spearmanr(w.pred, w.y)[0]:+.3f} ({time.time() - t0:.0f} s)", flush=True)
    P.append(Q)
P = pd.concat(P, ignore_index=True).sort_values("_t")
# prog przyczynowy: kwantyl (1 - gorne) prognoz z poprzednich 7 dni (wszystkie coiny)
dni = P._t.dt.normalize(); ud = np.sort(dni.unique()); kw = {}
for i, d in enumerate(ud):
    if i < 7: continue
    okno_ = P.pred.values[(dni.values >= ud[i - 7]) & (dni.values < d)]
    if len(okno_) > 1000: kw[d] = np.quantile(okno_, 1 - a.gorne)
P["prog"] = dni.map(kw).values
P = P[P.w_oknie & P.prog.notna()]


def symuluj(sygnaly):
    """sygnaly: DataFrame symbol, _t, atr_abs -> transakcje OCO"""
    tx = []
    for s, g in sygnaly.groupby("symbol"):
        o = SW[s]; H, L, C = o.high.values, o.low.values, o.close.values
        idx = pd.Series(np.arange(len(o)), index=o.index)
        wolny_od = -1
        for t, atr, okn in zip(g._t.values, g.atr_abs.values, g.okno.values):
            i = int(idx[t])
            if i <= wolny_od or i < OKNO_Z: continue
            gora = H[i - OKNO_Z + 1:i + 1].max() + BUF * atr; dol = L[i - OKNO_Z + 1:i + 1].min() - BUF * atr
            wej = None
            for j in range(i + 1, min(i + 1 + WAZNE, len(C))):
                ug, ud_ = H[j] >= gora, L[j] <= dol
                if ug and ud_: wej = ("oba", j); break
                if ug: wej = (1, j); break
                if ud_: wej = (-1, j); break
            if wej is None: continue
            if wej[0] == "oba":                                            # ostroznie: strata stopu
                tx.append((s, t, 0, -SL * atr / C[i] * 100 - KOSZT - POSL, okn)); wolny_od = wej[1]; continue
            zn, j = wej; cena = gora if zn == 1 else dol
            cel, stop = cena + zn * TP * atr, cena - zn * SL * atr
            wynik = None
            for k in range(j, min(j + MAXH, len(C))):
                if k == j:                          # swieca wejscia: jej high/low bylo czesciowo PRZED wejsciem (cena przebijala
                    trc = False                     # poziom) — liczy sie tylko zamkniecie ponizej stopu (ostroznie: strata)
                    trs = (C[k] <= stop) if zn == 1 else (C[k] >= stop)
                else:
                    trc = (H[k] >= cel) if zn == 1 else (L[k] <= cel); trs = (L[k] <= stop) if zn == 1 else (H[k] >= stop)
                if trs: wynik, wyj = -SL * atr, k; break
                if trc: wynik, wyj = TP * atr, k; break
            if wynik is None:
                wyj = min(j + MAXH - 1, len(C) - 1); wynik = (C[wyj] - cena) * zn
            tx.append((s, t, zn, wynik / cena * 100 - KOSZT - POSL, okn)); wolny_od = wyj
    return pd.DataFrame(tx, columns=["symbol", "t", "strona", "netto", "okno"])


rng = np.random.default_rng(0)
wyn = []
ndni = P._t.dt.normalize().nunique()
for nazwa, sy in (("bramka zmiennosci (top %.0f%%)" % (100 * a.gorne), P[P.pred >= P.prog]),
                  ("bez bramki (co %d h)" % a.co_ile, P[P._t.dt.hour % a.co_ile == 0])):
    T = symuluj(sy)
    d = T.groupby(pd.to_datetime(T.t).dt.normalize()).netto.mean()
    bs = [d.values[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    po = T.groupby("okno").netto.mean()
    r = dict(wariant=nazwa, tx_dzien=len(T) / ndni, wr=(T.netto > 0).mean(), netto_pct=T.netto.mean(), long=T[T.strona == 1].netto.mean(),
             short=T[T.strona == -1].netto.mean(), obie_strony_w_swiecy=(T.strona == 0).mean(), srednia_dzienna=d.mean(),
             boot5=np.percentile(bs, 5), okna_plus=f"{int((po > 0).sum())}/{len(po)}", polowa1=T[T.okno <= 6].netto.mean(),
             polowa2=T[T.okno > 6].netto.mean())
    wyn.append(r); T.to_parquet(f"{a.katalog}/transakcje_{'bramka' if 'bramka' in nazwa else 'bez'}.parquet", index=False)
W = pd.DataFrame(wyn); pd.set_option("display.width", 250)
print("\n=== WYBICIE OCO: stop 0,8 ATR, cel 1,8 ATR, max 12 h, koszt 0,22% + 0,05% poslizgu (99 coinow, 12 okien) ===")
print(W.round(3).to_string(index=False)); W.to_csv(f"{a.katalog}/wybicie_wyniki.csv", index=False)
