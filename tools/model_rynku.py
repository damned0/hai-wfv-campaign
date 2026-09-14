#!/usr/bin/env python3
"""Model RYNKU: osobny garnitur cech rynkowych -> kierunek calego rynku altow (2026-09-14).

Pomysl uzytkownika: cechy wspolne dla rynku (makro-krypto, BTC, pora dnia) nie nadaja sie do
WYBORU COINA (model zgaduje wtedy dzien), ale moga nadawac sie do osobnej warstwy: czy to czas
na longi czy na shorty. Potem polaczenie z modelem coina (cechy tylko coina).

Jeden wiersz = jedna godzina (a nie coin x godzina). Cechy:
- rynkowe wprost: fear_greed, btc_trend_1h/4h/1d, btc_rsi_4h, btc_dominance_chg, pora dnia;
- zagregowane po coinach w tej godzinie: srednia i rozrzut funding/RSI/momentum/ATR/wolumenu/CVD,
  szerokosc rynku (odsetek coinow nad EMA srednia, z dodatnim momentum).
Etykieta: rownowazony zwrot rynku altow w nastepnych H godzin (srednia po coinach), > 0 = wzrost.
Uczciwosc: podzial kalendarzowy (ten sam co wszedzie: ciecie 2024-10-31 + 7 dni), ocena po dniach.
Wynik: CSV z prawdopodobienstwem wzrostu rynku per godzina (do laczenia z transakcjami modeli coina).
"""
import argparse, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poszukiwania as PZ

ap = argparse.ArgumentParser()
ap.add_argument("--przygotowany", required=True); ap.add_argument("--wyniki", required=True)
ap.add_argument("--horyzonty", default="24,48")
a = ap.parse_args()
WPROST = ["fear_greed", "btc_trend_1h", "btc_trend_4h", "btc_trend_1d", "btc_rsi_4h", "btc_dominance_chg",
          "hour_sin", "hour_cos"]
AGREG = ["funding_rate", "funding_change_24h", "oi_change_24h", "rsi", "rsi_4h", "momentum", "x_roc_24",
         "ema_mid_r", "atr_pct", "volume_ratio", "cvd_z24", "taker_buy_ratio", "ls_ratio", "btc_corr_24h"]


def main():
    import pyarrow.parquet as pq
    dost = set(pq.read_schema(a.przygotowany).names)
    wp = [c for c in WPROST if c in dost]; ag = [c for c in AGREG if c in dost]
    Z = pd.read_parquet(a.przygotowany, columns=["symbol", "_t"] + wp + ag)
    # przyszly zwrot coina po H godzinach ze swiec magazynu, przez znacznik czasu (dziury w danych)
    hz = [int(h) for h in a.horyzonty.split(",")]
    czesci = []
    for s_ in Z.symbol.unique():
        o = pd.read_parquet(f"{PZ.ROOT}/data_warehouse/ohlcv/binance/1h/{s_}.parquet", columns=["timestamp", "close"])
        o["_t"] = pd.to_datetime(o.timestamp).astype("datetime64[ns]"); o = o.drop_duplicates("_t").set_index("_t").sort_index()
        c = o["close"]; d = pd.DataFrame({"symbol": s_}, index=c.index)
        for h in hz:
            d[f"_z{h}"] = c.reindex(c.index + pd.Timedelta(hours=h)).values / c.values - 1
        czesci.append(d.reset_index())
    Fz = pd.concat(czesci, ignore_index=True)
    Z["_t"] = Z["_t"].astype("datetime64[ns]")
    Z = Z.merge(Fz, on=["symbol", "_t"], how="left")
    g = Z.groupby("_t")
    R = g[wp].first()
    for c in ag:
        R[f"sr_{c}"] = g[c].mean(); R[f"roz_{c}"] = g[c].std()
    R["szerokosc_ema"] = g["ema_mid_r"].apply(lambda s: (s > 0).mean()) if "ema_mid_r" in ag else np.nan
    R["szerokosc_mom"] = g["momentum"].apply(lambda s: (s > 0).mean()) if "momentum" in ag else np.nan
    R["coinow"] = g["symbol"].count()
    for h in hz:
        R[f"rynek_{h}h"] = g[f"_z{h}"].mean()
    R = R[R.coinow >= 20].copy()
    cechy = [c for c in R.columns if not c.startswith("rynek_") and c != "coinow"]
    D = pd.Timestamp(os.environ.get("POSZ_CIECIE_STALE", "2024-10-31 15:00:00"))
    tr = R[R.index < D]; te = R[R.index >= D + pd.Timedelta(days=7)]
    print(f"model rynku: {len(cechy)} cech, trening {len(tr)} h ({tr.index.normalize().nunique()} dni), "
          f"test {len(te)} h ({te.index.normalize().nunique()} dni)", flush=True)
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    out = pd.DataFrame(index=R.index)
    for h in hz:
        y_tr = (tr[f"rynek_{h}h"] > 0).astype(int); m_tr = tr[f"rynek_{h}h"].notna()
        y_te = (te[f"rynek_{h}h"] > 0).astype(int); m_te = te[f"rynek_{h}h"].notna()
        Xtr = tr.loc[m_tr, cechy].fillna(0); Xte = te.loc[m_te, cechy].fillna(0)
        # Mocna regularyzacja: niezaleznych obserwacji jest ~dni, nie godzin
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.02, num_leaves=7, min_child_samples=400,
                               subsample=0.7, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, verbose=-1)
        m.fit(Xtr, y_tr[m_tr])
        p = m.predict_proba(Xte)[:, 1]
        sc = StandardScaler().fit(Xtr); lr = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(Xtr), y_tr[m_tr])
        pl = lr.predict_proba(sc.transform(Xte))[:, 1]
        # AUC z przedzialem z bootstrapu po DNIACH
        kod, ud = pd.factorize(te.loc[m_te].index.normalize()); rng = np.random.default_rng(3)
        idx = [np.where(kod == k)[0] for k in range(len(ud))]; yv = y_te[m_te].values
        bs = []
        for _ in range(300):
            ii = np.concatenate([idx[k] for k in rng.integers(0, len(ud), len(ud))])
            if len(np.unique(yv[ii])) == 2:
                bs.append(roc_auc_score(yv[ii], p[ii]))
        zw = te.loc[m_te, f"rynek_{h}h"].values
        gora, dol = p >= np.quantile(p, 0.8), p <= np.quantile(p, 0.2)
        print(f"\n=== rynek {h}h: wzrostow w tescie {yv.mean():.1%} | AUC LGB {roc_auc_score(yv, p):.3f} "
              f"(5-95%: {np.percentile(bs, 5):.3f}-{np.percentile(bs, 95):.3f}) | AUC logit {roc_auc_score(yv, pl):.3f}")
        print(f"    gorne 20% prognoz: sredni zwrot rynku {zw[gora].mean() * 100:+.2f}%, wzrost w {yv[gora].mean():.0%} | "
              f"dolne 20%: {zw[dol].mean() * 100:+.2f}%, wzrost w {yv[dol].mean():.0%} | wszystkie: {zw.mean() * 100:+.2f}%")
        kw = pd.Series((p >= 0.5) == yv, index=te.loc[m_te].index).groupby(te.loc[m_te].index.to_period("Q")).mean()
        print("    trafnosc kierunku per kwartal:", " ".join(f"{q}:{v:.0%}" for q, v in kw.items()))
        imp = pd.Series(m.feature_importances_, index=cechy).sort_values(ascending=False)
        print("    najwazniejsze cechy:", ", ".join(f"{k}({v})" for k, v in imp.head(8).items()))
        out.loc[te.loc[m_te].index, f"p_wzrost_{h}h"] = p
        out.loc[te.loc[m_te].index, f"rynek_{h}h"] = zw
    out.dropna(how="all").to_csv(a.wyniki)
    print(f"\nzapisano prognozy rynku per godzina: {a.wyniki}")


if __name__ == "__main__":
    main()
