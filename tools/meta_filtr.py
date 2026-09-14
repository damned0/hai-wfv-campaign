#!/usr/bin/env python3
"""Meta-filtr i analiza stopow na transakcjach z WFV kierunkowego (2026-09-14).

Uzytkownik: (1) filtr, ktory uczy sie, KIEDY dany model sie myli (meta-labeling, takze sieci),
(2) jakie i ile cech dominuje w pozycjach zamknietych STOPEM (SL).

Wejscie: katalog z tools/wfv_kierunkowe.py (transakcje_<model>.pkl z silnika + sygnaly_<model>.parquet
z pewnoscia modelu i numerem okna) + zbior przygotowany (cechy z chwili wejscia, dolaczane po
symbolu i godzinie — te same wartosci, ktore widzial model).

Uczciwosc: meta-model uczony na oknach 1..K (--okna-treningu), oceniany na pozniejszych oknach.
Cechy: tylko cechy coina + pewnosc modelu + strona (bez cech rynkowych — meta-model tez zgadywalby dzien).
"""
import argparse, os, sys, re, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poszukiwania as PZ

ap = argparse.ArgumentParser()
ap.add_argument("--katalog", required=True); ap.add_argument("--przygotowany", required=True)
ap.add_argument("--okna-treningu", type=int, default=8)
a = ap.parse_args()
RYNKOWE = {"fear_greed", "btc_trend_1h", "btc_trend_4h", "btc_trend_1d", "btc_rsi_4h", "btc_dominance_chg",
           "hour_sin", "hour_cos", "day_of_week", "x_weekend"}


def auc(x, y):
    m = ~np.isnan(x)
    x, y = x[m], y[m]
    n1, n0 = y.sum(), (~y).sum()
    if n1 < 5 or n0 < 5:
        return np.nan
    r = pd.Series(x).rank().values
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    import pyarrow.parquet as pq
    wynik_kol = re.compile(r"^r_\d+\.\d+_\d+\.\d+_\d+_[LS]$")
    kol = [k for k in pq.read_schema(a.przygotowany).names if not wynik_kol.match(k)
           and not k.startswith(("_", "label_", "trade_", "cel_", "__")) and k not in ("timestamp", "symbol", "close")
           and k not in PZ.MAKRO_USUNIETE and k not in RYNKOWE]
    for plik in sorted(glob.glob(f"{a.katalog}/transakcje_*.pkl")):
        model = os.path.basename(plik).replace("transakcje_", "").replace(".pkl", "")
        T = pd.read_pickle(plik)
        if not len(T):
            continue
        S = pd.read_parquet(f"{a.katalog}/sygnaly_{model}.parquet")
        T["coin"] = T.symbol.str.split("/").str[0]
        T = T.merge(S[["symbol", "ts_ms", "p", "okno"]].rename(columns={"symbol": "coin", "ts_ms": "open_ts"}),
                    on=["coin", "open_ts"], how="left")
        F = pd.read_parquet(a.przygotowany, columns=["symbol", "_t"] + kol)
        F["open_ts"] = F["_t"].astype("datetime64[ms]").astype("int64")
        T = T.merge(F.drop(columns="_t").rename(columns={"symbol": "coin"}), on=["coin", "open_ts"], how="left")
        cechy = [c for c in kol if c in T.columns and pd.api.types.is_numeric_dtype(T[c]) and T[c].notna().mean() > 0.5]
        T["wygrana"] = T.pnl_pct > 0; T["sl"] = T.result.astype(str).str.startswith("SL")
        print(f"\n################ {model.upper()}: {len(T)} transakcji z silnika | WR {T.wygrana.mean():.1%} | "
              f"SL {T.sl.mean():.1%} | sr {T.pnl_pct.mean():+.3f}% | z pewnoscia: {T.p.notna().mean():.0%}, z cechami: {T[cechy[0]].notna().mean():.0%}")

        # --- 1. co dominuje w pozycjach zamknietych stopem
        for strona in ("LONG", "SHORT"):
            D = T[T.side == strona]
            if len(D) < 60:
                continue
            y = D.sl.values
            tab = []
            for c in cechy:
                x = D[c].astype(float).replace([np.inf, -np.inf], np.nan).values
                u = auc(x, y)
                if np.isnan(u):
                    continue
                q = np.nanquantile(x, [1 / 3, 2 / 3])
                ter = [y[(x <= q[0])].mean(), y[(x > q[0]) & (x <= q[1])].mean(), y[x > q[1]].mean()]
                tab.append((c, u, ter))
            tab.sort(key=lambda t: -abs(t[1] - 0.5))
            ile = sum(1 for t in tab if abs(t[1] - 0.5) >= 0.05)
            print(f"\n  STOPY — {strona}: {len(D)} poz., SL {y.mean():.1%} | cech realnie mowiacych o SL (|AUC-0.5|>=0.05): "
                  f"{ile} z {len(tab)}")
            print("   cecha                          AUC   kierunek          SL: niska / srednia / wysoka")
            for c, u, ter in tab[:12]:
                kier = "wyzej = wiecej SL" if u > 0.5 else "nizej = wiecej SL"
                print(f"   {c:30s} {u:.3f} {kier:17s} {ter[0]:.0%} / {ter[1]:.0%} / {ter[2]:.0%}")

        # --- 2. meta-model: czy TA transakcja wygra (uczony na wczesnych oknach, test na pozniejszych)
        M = T[T.okno.notna()].copy()
        tr, te = M[M.okno <= a.okna_treningu], M[M.okno > a.okna_treningu]
        if len(tr) < 200 or len(te) < 100:
            print(f"\n  META: za malo transakcji (trening {len(tr)}, test {len(te)})"); continue
        X_c = cechy + ["p"]
        def X(d):
            x = d[X_c].astype(float).replace([np.inf, -np.inf], np.nan)
            x["strona_long"] = (d.side == "LONG").astype(float)
            return x
        Xtr, Xte = X(tr), X(te); ytr, yte = tr.wygrana.values, te.wygrana.values
        import lightgbm as lgb
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        modele = {
            "boosting": lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1),
            "siec (MLP)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=400,
                                                      early_stopping=True, random_state=1)),
            "logistyczna": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        }
        print(f"\n  META-FILTR (trening okna 1-{a.okna_treningu}: {len(tr)} tx, WR {ytr.mean():.1%} | "
              f"test okna {a.okna_treningu + 1}+: {len(te)} tx, WR {yte.mean():.1%}, sr {te.pnl_pct.mean():+.3f}%)")
        for nm, mm in modele.items():
            mm.fit(Xtr, ytr)
            pr = mm.predict_proba(Xte)[:, 1]
            u = auc(pr, yte)
            wiersz = [f"   {nm:12s} AUC {u:.3f}"]
            for zost in (0.5, 0.3):
                k = pr >= np.quantile(pr, 1 - zost)
                pp = te.pnl_pct.values[k]
                zysk, strata = pp[pp > 0].sum(), -pp[pp < 0].sum()
                wiersz.append(f"| najlepsze {int(zost * 100)}%: WR {yte[k].mean():.1%}, sr {pp.mean():+.3f}%, "
                              f"PF {zysk / strata if strata else float('inf'):.2f}, SL {te.sl.values[k].mean():.0%}")
            print(" ".join(wiersz))
        pp = te.pnl_pct.values; zysk, strata = pp[pp > 0].sum(), -pp[pp < 0].sum()
        print(f"   bez filtra   WR {yte.mean():.1%}, sr {pp.mean():+.3f}%, PF {zysk / strata if strata else float('inf'):.2f}, SL {te.sl.mean():.0%}")


if __name__ == "__main__":
    main()
