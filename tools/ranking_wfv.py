#!/usr/bin/env python3
"""WFV rankingu przekrojowego (2026-09-15): zamiast "czy coin trafi TP" -> "ktory coin bedzie lepszy od reszty".

Konsultacja 15.09 (6 modeli): wyzszy prog hybrydy nie podnosi WR, bo cechy mowia "ile sie ruszy", nie "w ktora strone";
dominuje beta rynku. Tu cel i cechy sa PRZEKROJOWE (w obrebie godziny), wiec wspolny ruch rynku sie znosi.
  cel     : zwrot log 6 h (zamkniecie t -> t+6) minus srednia wszystkich coinow w tej godzinie -> percentyl w godzinie
  cechy   : 97 cech ml_trainer -> percentyl w godzinie (+ surowe atr_pct, hour: kontekst)
  model   : LightGBM regresja na percentylu celu (uczenie: wszystko przed oknem - 7 dni embarga, max --wiersze)
  handel  : co --co-ile h (domyslnie 6: 00/06/12/18 UTC) long top-k i short bottom-k prognozy, trzymanie 6 h,
            koszt 0,22% ceny na noge; k z --k (liczba transakcji/dzien = 2 * k * 24 / co_ile)
  ocena   : IC (Spearman prognoza vs cel w godzinie), WR i netto na noge, para L-S, srednia DZIENNA + bootstrap po dniach,
            kontrola: losowy wybor k coinow (to samo, bez modelu).
Tryb GH: --tylko-okno N -> wyniki okna do plikow; skladanie: --polacz (bez zbioru).
"""
import argparse, glob, json, os, sys, time, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--zbior", required=True); ap.add_argument("--katalog", required=True)
ap.add_argument("--koniec", default="2026-08-14 00:00:00"); ap.add_argument("--okna", type=int, default=12)
ap.add_argument("--dni", type=int, default=45); ap.add_argument("--embargo", type=int, default=7)
ap.add_argument("--wiersze", type=int, default=800_000); ap.add_argument("--tylko-okno", type=int, default=0)
ap.add_argument("--polacz", action="store_true"); ap.add_argument("--co-ile", type=int, default=6)
ap.add_argument("--k", default="1,2,3,5"); ap.add_argument("--koszt", type=float, default=0.22)
ap.add_argument("--horyzont", type=int, default=6, help="h trzymania i celu; co-ile domyslnie = horyzont")
ap.add_argument("--uczen", action="store_true", help="nauczyciel z podgladem przyszlosci -> uczen na cechach uczciwych (+ model zwykly do porownania)")
ap.add_argument("--kol", default="pred", help="--polacz: ktora kolumne prognozy oceniac (pred / pred_zwykly)")
a = ap.parse_args()
if a.co_ile == 6 and a.horyzont != 6: a.co_ile = min(a.horyzont, 24)
ROOT = os.environ.get("HAI_ROOT", "/root/ProjektHAI")
sys.path.insert(0, ROOT if os.path.exists(f"{ROOT}/hai_common/ml_trainer.py") else f"{ROOT}/hai_common")   # repo GH: pakiet w korzeniu
os.makedirs(a.katalog, exist_ok=True)
H = a.horyzont
MARTWE = ("WAVES", "OCEAN", "FTM", "OMG", "MKR", "TON", "ICX", "STORJ")
MAKRO = ("gold_chg", "oil_wti_chg", "sp500_chg", "vix_chg", "us10y_chg", "dxy_chg", "fear_greed", "btc_dominance_chg")
koniec = pd.Timestamp(a.koniec)
starty = [koniec - pd.Timedelta(days=a.dni * k) for k in range(a.okna, 0, -1)]


def zbuduj():
    import hai_common.ml_trainer as mt
    if not os.path.exists(a.zbior):
        mt.build_dataset().to_parquet(a.zbior, index=False)
    Z = pd.read_parquet(a.zbior)
    Z = Z[~Z.symbol.isin(MARTWE)].copy()
    Z["_t"] = pd.to_datetime(Z.timestamp).dt.tz_localize(None).astype("datetime64[ns]")
    cz = []
    for s, g in Z.groupby("symbol"):                      # cel: zwrot 6 h z zamkniec 1h (przyszlosc TYLKO w celu)
        o = pd.read_parquet(f"{ROOT}/data_warehouse/ohlcv/binance/1h/{s}.parquet", columns=["timestamp", "close"])
        o["timestamp"] = pd.to_datetime(o.timestamp).dt.tz_localize(None).astype("datetime64[ns]")
        c = o.drop_duplicates("timestamp").set_index("timestamp").close
        c = c.reindex(pd.date_range(c.index[0], c.index[-1], freq="h"))
        fw = np.log(c.shift(-H) / c)
        g = g.assign(fwd=fw.reindex(g._t.values).values)
        if a.uczen:                                        # PODGLAD PRZYSZLOSCI — wylacznie cechy nauczyciela
            r1 = np.log(c / c.shift(1))
            g = g.assign(p_fwd_1h=r1.shift(-1).reindex(g._t.values).values,
                         p_vol_fut=r1.abs().rolling(H).sum().shift(-H).reindex(g._t.values).values)
        cz.append(g)
    Z = pd.concat(cz, ignore_index=True)
    Z = Z[Z.fwd.notna()]
    if a.uczen:
        b = Z[Z.symbol == "BTC"].set_index("_t").fwd
        Z["p_btc_fut"] = Z._t.map(b).values
    Z["fwd_res"] = Z.fwd - Z.groupby("_t").fwd.transform("mean")
    Z["y"] = Z.groupby("_t").fwd_res.rank(pct=True)
    Z["n_godz"] = Z.groupby("_t").symbol.transform("size")
    Z = Z[Z.n_godz >= 20]                                  # godziny z malym przekrojem odpadaja
    cechy = [c for c in Z.columns if c not in ("symbol", "timestamp", "_t", "fwd", "fwd_res", "y", "n_godz", "close")
             and not c.startswith("p_")
             and not c.startswith("label") and c not in MAKRO and pd.api.types.is_numeric_dtype(Z[c]) and Z[c].nunique() > 5]
    R = Z[["symbol", "_t", "fwd", "fwd_res", "y", "atr_pct"]].copy()
    X = Z[cechy].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    Xr = X.groupby(Z._t.values).rank(pct=True).astype(np.float32)    # percentyl w godzinie
    Xr.columns = [f"pr_{c}" for c in cechy]
    Xr["atr_pct_surowe"] = X["atr_pct"].values; Xr["godzina"] = Z._t.dt.hour.values.astype(np.float32)
    if a.uczen:
        R = R.join(Z[["p_fwd_1h", "p_vol_fut", "p_btc_fut"]].astype(np.float32))
    return R.reset_index(drop=True), Xr.reset_index(drop=True)


def okno(nr, R, X):
    import lightgbm as lgb
    st = starty[nr - 1]; kon = st + pd.Timedelta(days=a.dni)
    tr = np.where(R._t < st - pd.Timedelta(days=a.embargo))[0]
    if len(tr) > a.wiersze:
        tr = np.sort(np.random.default_rng(nr).choice(tr, a.wiersze, replace=False))
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=2000, subsample=0.7,
                          subsample_freq=1, colsample_bytree=0.6, reg_lambda=10.0, verbose=-1,
                          n_jobs=int(os.environ.get("ENS_NJ", "4")))
    m.fit(X.iloc[tr], R.y.values[tr])
    te = np.where((R._t >= st) & (R._t < kon))[0]
    P = R.iloc[te][["symbol", "_t", "fwd", "fwd_res", "y", "atr_pct"]].copy(); P["pred"] = m.predict(X.iloc[te]); P["okno"] = nr
    if a.uczen:
        # NAUCZYCIEL: cechy uczciwe + podglad (p_*), 5 blokow czasowych w TRENINGU; ocena kazdego bloku z modelu, ktory
        # go nie widzial (bez przeuczenia miekkich etykiet). UCZEN: tylko cechy uczciwe, cel = ocena nauczyciela.
        # Walidacja: okno WFV, prawdziwy cel — nauczyciel i podglad nie dotykaja okna.
        Pp = R[["p_fwd_1h", "p_vol_fut", "p_btc_fut"]].values
        XT = np.hstack([X.iloc[tr].values, Pp[tr]])
        blok = np.minimum((np.argsort(np.argsort(R._t.values[tr])) * 5) // len(tr), 4)
        miekkie = np.zeros(len(tr))
        for b in range(5):
            n = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=2000, subsample=0.7,
                                  subsample_freq=1, colsample_bytree=0.6, reg_lambda=10.0, verbose=-1, n_jobs=int(os.environ.get("ENS_NJ", "4")))
            n.fit(XT[blok != b], R.y.values[tr][blok != b]); miekkie[blok == b] = n.predict(XT[blok == b])
        from scipy.stats import spearmanr
        print(f"  nauczyciel (poza blokiem): korelacja z celem {spearmanr(miekkie, R.y.values[tr])[0]:+.3f}", flush=True)
        u = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=2000, subsample=0.7,
                              subsample_freq=1, colsample_bytree=0.6, reg_lambda=10.0, verbose=-1, n_jobs=int(os.environ.get("ENS_NJ", "4")))
        u.fit(X.iloc[tr], miekkie)
        P["pred_zwykly"] = P["pred"]; P["pred"] = u.predict(X.iloc[te])
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    return P, imp


def ocen(P):
    ks = [int(k) for k in a.k.split(",")]
    P = P.reset_index(drop=True).copy(); P["dzien"] = P._t.dt.normalize()
    P["pred"] = P[a.kol]
    ic = P.groupby("_t").apply(lambda g: g.pred.corr(g.y, method="spearman"))
    print(f"\nIC (Spearman prognoza vs cel w godzinie): srednio {ic.mean():+.4f}, t-stat {ic.mean() / ic.std() * np.sqrt(len(ic)):+.1f}, "
          f"godzin z IC>0 {(ic > 0).mean():.1%}, per okno: " + " ".join(f"{v:+.3f}" for v in ic.groupby(P.groupby('_t').okno.first()).mean().values))
    Q = P[P._t.dt.hour % a.co_ile == 0]
    wyn = []
    rng = np.random.default_rng(0)
    for k in ks:
        for tryb in ("model", "model+bramka_zm", "losowo"):
            if tryb == "model+bramka_zm":                      # tylko coiny z gornej 1/3 zmiennosci (atr_pct) w godzinie
                Qb = Q[Q.atr_pct >= Q.groupby("_t").atr_pct.transform(lambda x: x.quantile(2 / 3))]
                r = Qb.groupby("_t").pred.rank(ascending=False, method="first").reindex(Q.index).fillna(1e9)
                r2 = Qb.groupby("_t").pred.rank(ascending=True, method="first").reindex(Q.index).fillna(1e9)
            elif tryb == "model":
                r = Q.groupby("_t").pred.rank(ascending=False, method="first"); r2 = Q.groupby("_t").pred.rank(ascending=True, method="first")
            else:
                los = pd.Series(rng.random(len(Q)), index=Q.index)
                r = los.groupby(Q._t).rank(ascending=False, method="first"); r2 = los.groupby(Q._t).rank(ascending=True, method="first")
            L = Q[r <= k]; S = Q[r2 <= k]
            netL = (np.exp(L.fwd) - 1) * 100 - a.koszt; netS = (1 - np.exp(S.fwd)) * 100 - a.koszt
            nogi = pd.concat([netL.rename("n"), netS.rename("n")]); dz = pd.concat([L.dzien, S.dzien])
            d = nogi.groupby(dz.values).mean()
            bs = [d.values[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
            po = nogi.groupby(pd.concat([L.okno, S.okno]).values).mean()
            wyn.append(dict(k=k, tryb=tryb, tx_dzien=len(nogi) / P.dzien.nunique(), wr_nog=float((nogi > 0).mean()),
                            wr_long=float((netL > 0).mean()), wr_short=float((netS > 0).mean()), netto_noga_pct=float(nogi.mean()),
                            long_pct=float(netL.mean()), short_pct=float(netS.mean()), srednia_dzienna=float(d.mean()),
                            boot5=float(np.percentile(bs, 5)), okna_plus=f"{int((po > 0).sum())}/{len(po)}",
                            polowa1=float(nogi[pd.concat([L.okno, S.okno]).values <= a.okna // 2].mean()),
                            polowa2=float(nogi[pd.concat([L.okno, S.okno]).values > a.okna // 2].mean())))
    W = pd.DataFrame(wyn)
    pd.set_option("display.width", 250)
    print(W.round(3).to_string(index=False))
    return W, ic


if a.polacz:
    P = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{a.katalog}/**/prognozy_okno_*.parquet", recursive=True))], ignore_index=True)
    print(f"polaczono okna: {sorted(P.okno.unique())}, wierszy {len(P)}")
    W, ic = ocen(P); W.to_csv(f"{a.katalog}/ranking_wyniki.csv", index=False)
    sys.exit(0)
t0 = time.time()
R, X = zbuduj()
print(f"zbior: {len(R)} wierszy, {R.symbol.nunique()} symboli, {X.shape[1]} cech, godziny {R._t.min()} .. {R._t.max()} ({time.time() - t0:.0f} s)", flush=True)
nry = [a.tylko_okno] if a.tylko_okno else list(range(1, a.okna + 1))
for nr in nry:
    P, imp = okno(nr, R, X)
    P.to_parquet(f"{a.katalog}/prognozy_okno_{nr:02d}.parquet", index=False)
    print(f"okno {nr} ({starty[nr - 1].date()}): {len(P)} wierszy, top cechy: {', '.join(imp.index[:8])} ({time.time() - t0:.0f} s)", flush=True)
if not a.tylko_okno:
    W, ic = ocen(pd.concat([pd.read_parquet(f"{a.katalog}/prognozy_okno_{n:02d}.parquet") for n in nry]))
    W.to_csv(f"{a.katalog}/ranking_wyniki.csv", index=False)
