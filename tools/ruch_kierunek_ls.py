#!/usr/bin/env python3
"""Szeregowanie cech wg: RUCH / KIERUNEK / strona LONG / strona SHORT (2026-09-28).

IC przekrojowe jest SYMETRYCZNE — dodatnie IC znaczy "wysoka wartosc cechy -> wyzszy zwrot",
co obsluguje obie strony jednoczesnie. Ale asymetria MOZE istniec: cecha moze dobrze
wskazywac gorny ogon i wcale nie wskazywac dolnego.

Tu rozdzielamy:
  RUCH        IC wobec |zwrot|                      — jak mocno sie ruszy
  RUCH+       IC wobec |zwrot| ponad zmiennosc      — czy dodaje cokolwiek ponad ATR
  KIERUNEK    IC wobec zwrotu minus srednia rynku   — w ktora strone
  LONG        srednia reszty w GORNYM tercylu       — ile zarabia dluga noga
  SHORT       MINUS srednia reszty w DOLNYM tercylu — ile zarabia krotka noga
  ASYMETRIA   |LONG| - |SHORT|                      — ktora strona nosi wartosc

Kontrola: dla kazdej nogi losowy koszyk DOPASOWANY po (godzina x tercyl zmiennosci).
Bez tego cecha korelujaca ze zmiennoscia wyglada na sygnal po obu stronach.
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get("HAI_ROOT", "/root/ProjektHAI")
HZ = 6
rng = np.random.default_rng(0)


def licz(C, H, L, V):
    D = {}
    D["atr_pct"] = (H.rolling(14).max() - L.rolling(14).min()) / C * 100
    sr = C.rolling(20).mean(); od = C.rolling(20).std()
    D["bb_bandwidth_pct"] = (4 * od) / sr * 100
    D["price_position_bb"] = (C - (sr - 2 * od)) / (4 * od).replace(0, np.nan)
    d = C.diff()
    zy = d.clip(lower=0).rolling(14).mean(); st = (-d.clip(upper=0)).rolling(14).mean()
    D["rsi"] = 100 - 100 / (1 + zy / st.replace(0, np.nan))
    D["momentum"] = (C / C.shift(10) - 1) * 100
    D["ema_mid_r"] = (C / C.ewm(span=21, adjust=False).mean() - 1) * 100
    D["ema_slow_r"] = (C / C.ewm(span=55, adjust=False).mean() - 1) * 100
    D["x_roc_24"] = (C / C.shift(24) - 1) * 100
    D["x_roc_168"] = (C / C.shift(168) - 1) * 100
    D["volume_zscore"] = (V - V.rolling(168).mean()) / V.rolling(168).std()
    D["r_atr_zscore_20"] = (D["atr_pct"] - D["atr_pct"].rolling(20).mean()) / D["atr_pct"].rolling(20).std()
    zak6 = H.rolling(6).max() - L.rolling(6).min()
    zak48 = H.rolling(48).max() - L.rolling(48).min()
    D["r_range_compression"] = zak6 / zak48.replace(0, np.nan)
    D["rank_mom_24h"] = (C / C.shift(24) - 1).rank(axis=1, pct=True)
    D["sr_dist_pct"] = (C - L.rolling(48).min()) / C * 100
    ich = (H.rolling(26).max() + L.rolling(26).min()) / 2 - (H.rolling(52).max() + L.rolling(52).min()) / 2
    D["e_ichimoku_cloud_thickness"] = ich / C * 100
    return D


def main():
    C, H, L, V = {}, {}, {}, {}
    for p in sorted(glob.glob(f"{ROOT}/data_warehouse/ohlcv/binance/1h/*.parquet")):
        s = os.path.basename(p)[:-8]
        o = pd.read_parquet(p, columns=["timestamp", "high", "low", "close", "volume"])
        o["timestamp"] = pd.to_datetime(o.timestamp).dt.tz_localize(None)
        o = o.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        C[s], H[s], L[s], V[s] = o.close, o.high, o.low, o.volume
    C = pd.DataFrame(C).sort_index()
    H = pd.DataFrame(H).reindex(C.index); L = pd.DataFrame(L).reindex(C.index)
    V = pd.DataFrame(V).reindex(C.index)
    C = C[C.index >= "2021-08-01"]
    H, L, V = H.reindex(C.index), L.reindex(C.index), V.reindex(C.index)
    D = licz(C, H, L, V)
    R = (C.shift(-HZ) / C - 1) * 100
    RS = R.sub(R.mean(axis=1), axis=0)
    A = R.abs()
    ZM = C.pct_change().rolling(72).std() * 100
    idx = C.index[::HZ]
    kol = list(C.columns)
    print(f"monet {len(kol)} | okresow {len(idx):,} | horyzont {HZ} h\n", flush=True)

    # dluga tabela raz, potem maski per cecha
    T = np.repeat(idx.values, len(kol))
    rs = RS.reindex(idx).to_numpy().ravel()
    aa = A.reindex(idx).to_numpy().ravel()
    zm = ZM.reindex(idx).to_numpy().ravel()
    baza = pd.DataFrame({"t": T, "rs": rs, "aa": aa, "zm": zm})
    ok = baza[["rs", "aa", "zm"]].notna().all(axis=1).to_numpy()
    baza = baza[ok].reset_index(drop=True)
    baza["kub"] = baza.groupby("t").zm.transform(
        lambda z: pd.qcut(z.rank(method="first"), 3, labels=False, duplicates="drop"))
    baza = baza.dropna(subset=["kub"]).reset_index(drop=True)
    kt, _ = pd.factorize(baza.t)
    kom, _ = pd.factorize(pd.Series(kt * 3 + baza.kub.to_numpy().astype(int)))
    kolp = np.argsort(kom, kind="stable"); lic = np.bincount(kom)
    stt = np.concatenate([[0], np.cumsum(lic)[:-1]])
    okr, _ = pd.factorize(pd.to_datetime(baza.t).dt.to_period("M"))
    nk = int(okr.max()) + 1
    rsv = baza.rs.to_numpy()

    def sr(i):
        su = np.bincount(okr[i], weights=rsv[i], minlength=nk)
        cn = np.bincount(okr[i], minlength=nk)
        m = cn > 0
        return float((su[m] / cn[m]).mean())

    def kontrola(sel, n=200):
        w = np.where(sel)[0]
        if len(w) < 200:
            return np.nan, np.nan
        kw = kom[w]
        los = np.array([sr(kolp[stt[kw] + (rng.random(len(kw)) * lic[kw]).astype(np.int64)])
                        for _ in range(n)])
        return sr(w), float(np.percentile(los, 95))

    def ic(x, y):
        s = pd.DataFrame({"t": baza.t, "x": x, "y": y}).dropna()
        w = s.groupby("t").apply(lambda z: z.x.corr(z.y, method="spearman")
                                 if len(z) > 20 and z.x.nunique() > 5 else np.nan,
                                 include_groups=False).dropna()
        return float(w.mean()) if len(w) else np.nan

    print("=" * 112)
    print(f"{'cecha':<28} {'RUCH':>8} {'KIERUNEK':>9} {'LONG':>9} {'kontr':>8} {'SHORT':>9} {'kontr':>8} {'ASYM':>8}")
    print("=" * 112)
    W = []
    for nm, X in D.items():
        xv = X.reindex(idx).to_numpy().ravel()[ok]
        xv = xv[baza.index] if len(xv) != len(baza) else xv
        if len(xv) != len(baza):
            continue
        s = pd.DataFrame({"t": baza.t, "x": xv})
        rk = s.groupby("t").x.rank(pct=True).to_numpy()
        gora = rk >= 0.67
        dol = rk <= 0.33
        l_mod, l_kon = kontrola(gora)
        s_mod, s_kon = kontrola(dol)
        s_mod, s_kon = -s_mod, -s_kon        # short zarabia na ujemnej reszcie
        ruch = ic(xv, baza.aa.to_numpy())
        kier = ic(xv, rsv)
        W.append((nm, ruch, kier, l_mod, l_kon, s_mod, s_kon, abs(l_mod) - abs(s_mod)))
    for nm, ru, ki, lm, lk, sm, sk, asym in sorted(W, key=lambda x: -abs(x[1])):
        zl = "*" if lm > lk else " "
        zs = "*" if sm > sk else " "
        print(f"{nm:<28} {ru:>+8.4f} {ki:>+9.4f} {lm:>+9.4f}{zl} {lk:>+7.4f} {sm:>+9.4f}{zs} {sk:>+7.4f} {asym:>+8.4f}")
    print("\n  * = noga bije losowy koszyk DOPASOWANY po zmiennosci")
    print("  ASYM > 0 -> wartosc siedzi po stronie DLUGIEJ, < 0 -> po KROTKIEJ")


if __name__ == "__main__":
    main()
