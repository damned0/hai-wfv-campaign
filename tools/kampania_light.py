#!/usr/bin/env python3
"""KAMPANIA LIGHT (2026-09-26) — czy przewaga jest w dluzszym horyzoncie i wlasciwym celu uczenia.

Powod: przez pol roku uczylismy modele na pytaniu "czy ta moneta dotknie TP przed SL w 6 h"
(triple_barrier, jedna moneta w izolacji), a wybieralismy cechy po IC przekrojowym. 25.09 zmierzone:
IC uklada decyle POPRAWNIE wedlug mediany i ODWROTNIE wedlug sredniej — rozklad jest prawoskosny
(skosnosc +3,2..+6,5 w kazdym decylu), wiec mediana jest ujemna wszedzie. Mozna podnosic IC i tracic.

Tu sprawdzamy trzy rzeczy naraz, kazda osobno rozstrzygalna:
  HORYZONT  — koszt 0,22% placi sie RAZ, wiec przy 168 h potrzeba 0,031%/dobe zamiast 0,22%/dobe
  CEL       — triple_barrier (jak dotad) vs PRZEKROJOWY (czy pobijesz reszte rynku)
  CECHY     — rdzen czysty / rdzen+HTF / HTF+zmiana zmiennosci

Cechy liczone WYLACZNIE z przeszlosci, oknami kroczacymi konczacymi sie w t — bez mapowania na
domkniete swiece HTF, bo to bylo zrodlo przecieku 4h/1d naprawionego 13.09.

OCENA: sredni zwrot netto koszyka gornego decyla po KOSZCIE OD OBROTU Z DZWIGNIA, bootstrap blokowy
po tygodniach, kontrola filtrem losowym przepuszczajacym tyle samo transakcji (percentyl 95).
Zadnego IC, AUC ani f1 w kryterium GO — patrz wyzej dlaczego.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd

ROOT = os.environ.get("HAI_ROOT", "/root/ProjektHAI")
KOSZT = 0.22                      # % od obrotu Z DZWIGNIA (wejscie+wyjscie, taker)
H_HTF = (4, 12, 24, 72, 168, 720)


def wczytaj_swiece(symbole=None):
    C, H, L, V = {}, {}, {}, {}
    for p in sorted(glob.glob(f"{ROOT}/data_warehouse/ohlcv/binance/1h/*.parquet")):
        s = os.path.basename(p)[:-8]
        if symbole and s not in symbole:
            continue
        o = pd.read_parquet(p, columns=["timestamp", "open", "high", "low", "close", "volume"])
        o["timestamp"] = pd.to_datetime(o.timestamp).dt.tz_localize(None)
        o = o.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        C[s], H[s], L[s], V[s] = o.close, o.high, o.low, o.volume
    return (pd.DataFrame(C).sort_index(), pd.DataFrame(H).sort_index(),
            pd.DataFrame(L).sort_index(), pd.DataFrame(V).sort_index())


def cechy_symbolu(c, h, l, v, zestaw):
    """Wszystko rolling().shift(0) konczace sie w t — zadnej wartosci z przyszlosci."""
    D = {}
    if zestaw in ("rdzen", "rdzen_htf"):
        D["atr_pct"] = (h.rolling(14).max() - l.rolling(14).min()) / c * 100
        D["ret_24"] = (c / c.shift(24) - 1) * 100
        D["ema_mid_r"] = (c / c.ewm(span=21, adjust=False).mean() - 1) * 100
        D["ema_slow_r"] = (c / c.ewm(span=55, adjust=False).mean() - 1) * 100
        d = c.diff()
        zy = d.clip(lower=0).rolling(14).mean(); st = (-d.clip(upper=0)).rolling(14).mean()
        D["rsi"] = 100 - 100 / (1 + zy / st.replace(0, np.nan))
        sr = c.rolling(20).mean(); od = c.rolling(20).std()
        D["poz_bb"] = (c - (sr - 2 * od)) / (4 * od).replace(0, np.nan)
        D["szer_bb"] = (4 * od) / sr * 100
        D["vol_z"] = (v - v.rolling(168).mean()) / v.rolling(168).std()
    if zestaw in ("rdzen_htf", "htf_zmiana"):
        for Hh in (72, 168, 720):
            mx, mn = h.rolling(Hh).max(), l.rolling(Hh).min()
            D[f"atr_{Hh}"] = (mx - mn) / c * 100
            D[f"ret_{Hh}"] = (c / c.shift(Hh) - 1) * 100
            D[f"ema_{Hh}"] = (c / c.ewm(span=Hh, adjust=False).mean() - 1) * 100
        zn = np.sign(pd.DataFrame({k: D[f"ret_{k}"] for k in (72, 168, 720)}))
        D["zgodnosc_htf"] = zn.mean(axis=1)
    if zestaw == "htf_zmiana":
        # NOWA HIPOTEZA (26.09): nie POZIOM zmiennosci (skazony przetrwaniem), tylko ZMIANA.
        a24 = (h.rolling(24).max() - l.rolling(24).min()) / c * 100
        a168 = (h.rolling(168).max() - l.rolling(168).min()) / c * 100
        D["ekspansja"] = a24 / a168.replace(0, np.nan)          # zmiennosc krotka / dluga
        D["przysp_zmien"] = a24 / a24.shift(24) - 1             # przyspieszenie
        D["szok_zmien"] = (a24 - a24.rolling(168).mean()) / a24.rolling(168).std()
        r = c.pct_change()
        D["realvol_zm"] = r.rolling(24).std() / r.rolling(168).std().replace(0, np.nan)
    return pd.DataFrame(D)


def zbuduj(zestaw, symbole=None):
    C, H, L, V = wczytaj_swiece(symbole)
    C = C[C.index >= "2021-08-01"]
    H, L, V = H.reindex(C.index), L.reindex(C.index), V.reindex(C.index)
    kawalki = []
    for s in C.columns:
        d = cechy_symbolu(C[s], H[s], L[s], V[s], zestaw)
        d["symbol"] = s; d["timestamp"] = C.index
        kawalki.append(d)
    X = pd.concat(kawalki, ignore_index=True)
    for k in X.columns:                       # 134 symbole x 45 tys. godzin — float32 zamiast 64
        if X[k].dtype == "float64":
            X[k] = X[k].astype("float32")
    # cechy przekrojowe — ranga wewnatrz chwili, zawsze dostepne w t
    X["rank_ret24"] = X.groupby("timestamp")[[k for k in X.columns if k.startswith("ret_")][0]].rank(pct=True)
    return X, C


def zmiennosc_realna(C, hz):
    """zmiennosc realizowana w oknie hz konczacym sie w t — wylacznie z przeszlosci"""
    r = C.pct_change()
    return (r.rolling(hz).std() * 100).stack().rename("zmien").reset_index()


def etykieta(X, C, cel, hz):
    R = (C.shift(-hz) / C - 1) * 100
    RS = R.sub(R.mean(axis=1), axis=0)                 # zwrot minus srednia rynku
    dl = R.stack().rename("fwd").reset_index()
    dl.columns = ["timestamp", "symbol", "fwd"]
    dp = RS.stack().rename("fwd_prz").reset_index()
    dp.columns = ["timestamp", "symbol", "fwd_prz"]
    X = X.merge(dl, on=["timestamp", "symbol"], how="inner").merge(dp, on=["timestamp", "symbol"], how="inner")
    if cel == "bariera":
        # jak dotad: bezwzgledny zwrot ponad prog, jedna moneta w izolacji
        X["y"] = (X.fwd > 0).astype(int)
    elif cel == "przekrojowy":
        # czy moneta trafi w gorny kwintyl rynku
        prog = X.groupby("timestamp").fwd_prz.transform(lambda z: z.quantile(0.8))
        X["y"] = (X.fwd_prz >= prog).astype(int)
    elif cel == "koszt":
        # KOSZT W ETYKIECIE: model uczy sie tego, co jest HANDLOWALNE, nie tego co dodatnie.
        # Dotad koszt odejmowalismy po fakcie, przy ocenie — model go nie widzial.
        X["y"] = (X.fwd > KOSZT).astype(int)
    elif cel == "ogon":
        # CEL OGONOWY: gorne 5%, nie 20%. Zmierzone 25.09: caly pieniadz siedzi w ogonie
        # (wklad gornego 1% to +0,12..+0,35 pp), a nasz wybor go systematycznie odrzuca.
        prog = X.groupby("timestamp").fwd_prz.transform(lambda z: z.quantile(0.95))
        X["y"] = (X.fwd_prz >= prog).astype(int)
    elif cel == "zmiennosc":
        # ZWROT SKORYGOWANY O ZMIENNOSC: usuwa skazenie, przez ktore cel barierowy pyta
        # o zmiennosc monety, a nie o jej kierunek (TP/SL sa w wielokrotnosciach ATR).
        Zm = zmiennosc_realna(C, hz)
        Zm.columns = ["timestamp", "symbol", "zmien"]
        X = X.merge(Zm, on=["timestamp", "symbol"], how="left")
        X["skor"] = X.fwd_prz / X.zmien.replace(0, np.nan)
        prog = X.groupby("timestamp").skor.transform(lambda z: z.quantile(0.8))
        X["y"] = (X.skor >= prog).astype(int)
        X = X.dropna(subset=["skor"])
    elif cel == "ranking":
        # CEL RANKINGOWY: stopniowana istotnosc 0-4 po kwintylach zwrotu przekrojowego.
        # Uczony przez lambdarank z grupowaniem po chwili — model optymalizuje KOLEJNOSC
        # wewnatrz chwili, a nie klasyfikuje kazdy wiersz osobno. To wlasciwe sformulowanie
        # pytania "ktora moneta teraz", ktorego nigdy nie uzylismy.
        X["y"] = X.groupby("timestamp").fwd_prz.transform(
            lambda z: pd.qcut(z.rank(method="first"), 5, labels=False, duplicates="drop")).fillna(0).astype(int)
    else:
        raise ValueError(cel)
    return X.dropna(subset=["y", "fwd", "fwd_prz"])


def ocena(Q, rng):
    """Sredni zwrot netto koszyka gornego decyla + bootstrap tygodniowy + kontrola losowa.

    Kontrola losowa zwektoryzowana (bincount na kodach tygodni) — pierwotna wersja robila
    200x sample+groupby i sama zjadala ~40 s na okno, czyli wiecej niz trening.
    """
    if len(Q) < 200:
        return None
    Q = Q.copy()
    kody, _ = pd.factorize(Q.timestamp.dt.to_period("W"))
    fwd = Q.fwd.to_numpy(dtype="float64")
    nk = int(kody.max()) + 1
    prog = Q.p.quantile(0.9)
    maska = (Q.p >= prog).to_numpy()
    n_wyb = int(maska.sum())
    if n_wyb < 30:
        return None

    def sr_tyg(sel):
        su = np.bincount(kody[sel], weights=fwd[sel], minlength=nk)
        cn = np.bincount(kody[sel], minlength=nk)
        m = cn > 0
        return su[m] / cn[m]

    tyg = sr_tyg(maska)
    bs = np.array([tyg[rng.integers(0, len(tyg), len(tyg))].mean() for _ in range(2000)])
    N = len(Q)
    los = np.empty(500)
    for i in range(500):
        idx = rng.choice(N, n_wyb, replace=False)
        sel = np.zeros(N, bool); sel[idx] = True
        los[i] = sr_tyg(sel).mean()
    return {
        "n_wybranych": n_wyb,
        "brutto": float(tyg.mean()),
        "netto": float(tyg.mean() - KOSZT),
        "boot5": float(np.percentile(bs, 5) - KOSZT),
        "losowy_sr": float(los.mean() - KOSZT),
        "losowy_p95": float(np.percentile(los, 95) - KOSZT),
        "bije_losowy": bool(tyg.mean() > np.percentile(los, 95)),
        "transakcji_na_dzien": float(n_wyb / max(Q.timestamp.dt.normalize().nunique(), 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horyzont", type=int, required=True)
    ap.add_argument("--cel", choices=["bariera", "przekrojowy", "koszt", "ogon", "zmiennosc", "ranking"],
                    required=True)
    ap.add_argument("--zestaw", choices=["rdzen", "rdzen_htf", "htf_zmiana"], required=True)
    ap.add_argument("--okien", type=int, default=6)
    ap.add_argument("--dni-okna", type=int, default=30)
    ap.add_argument("--embargo-h", type=int, default=None)
    ap.add_argument("--wyjscie", default="wyniki")
    a = ap.parse_args()
    embargo = a.embargo_h if a.embargo_h is not None else a.horyzont * 2

    from lightgbm import LGBMClassifier, LGBMRanker
    rng = np.random.default_rng(0)
    print(f"zestaw={a.zestaw} cel={a.cel} horyzont={a.horyzont}h embargo={embargo}h", flush=True)
    X, C = zbuduj(a.zestaw)
    X = etykieta(X, C, a.cel, a.horyzont)
    CECHY = [k for k in X.columns if k not in ("timestamp", "symbol", "y", "fwd", "fwd_prz")]
    X = X.dropna(subset=CECHY).sort_values("timestamp").reset_index(drop=True)
    print(f"  wierszy {len(X):,} | cech {len(CECHY)} | {X.timestamp.min():%Y-%m} .. {X.timestamp.max():%Y-%m}", flush=True)

    koniec = X.timestamp.max()
    okna = []
    for i in range(a.okien):
        t_kon = koniec - pd.Timedelta(days=a.dni_okna * i)
        t_pocz = t_kon - pd.Timedelta(days=a.dni_okna)
        tr_kon = t_pocz - pd.Timedelta(hours=embargo)        # EMBARGO miedzy treningiem a testem
        tr = X[X.timestamp < tr_kon]
        te = X[(X.timestamp >= t_pocz) & (X.timestamp < t_kon)]
        if len(tr) < 5000 or len(te) < 200:
            print(f"  okno {i+1}: za malo danych (tr={len(tr)}, te={len(te)}) — pomijam", flush=True)
            continue
        wsp = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                   subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=4, verbose=-1)
        te = te.copy()
        if a.cel == "ranking":
            tr = tr.sort_values("timestamp")
            grupy = tr.groupby("timestamp", sort=False).size().values
            m = LGBMRanker(objective="lambdarank", label_gain=list(range(5)), **wsp)
            m.fit(tr[CECHY], tr.y, group=grupy)
            te["p"] = m.predict(te[CECHY])
        else:
            m = LGBMClassifier(**wsp)
            m.fit(tr[CECHY], tr.y)
            te["p"] = m.predict_proba(te[CECHY])[:, 1]
        os.makedirs(f"{a.wyjscie}/prognozy", exist_ok=True)
        te[["timestamp", "symbol", "p", "fwd", "fwd_prz"]].to_parquet(
            f"{a.wyjscie}/prognozy/{a.zestaw}_{a.cel}_h{a.horyzont}_okno{i+1}.parquet", index=False)
        o = ocena(te, rng)
        if o:
            o["okno"] = i + 1
            o["od"] = str(t_pocz)[:10]; o["do"] = str(t_kon)[:10]
            o["udzial_klasy"] = float(tr.y.mean())
            okna.append(o)
            print(f"  okno {i+1} {o['od']}..{o['do']}: netto {o['netto']:+.4f} "
                  f"boot5 {o['boot5']:+.4f} losowy_p95 {o['losowy_p95']:+.4f} "
                  f"bije={o['bije_losowy']} tx/dzien {o['transakcji_na_dzien']:.1f}", flush=True)

    if not okna:
        print("BRAK OKIEN"); sys.exit(0)
    netto = np.array([o["netto"] for o in okna])
    wynik = {
        "zestaw": a.zestaw, "cel": a.cel, "horyzont": a.horyzont, "embargo_h": embargo,
        "cech": len(CECHY), "cechy": CECHY, "okien": len(okna),
        "netto_srednio": float(netto.mean()),
        "okien_dodatnich": int((netto > 0).sum()),
        "okien_bijacych_losowy": int(sum(o["bije_losowy"] for o in okna)),
        "tx_na_dzien_srednio": float(np.mean([o["transakcji_na_dzien"] for o in okna])),
        "boot5_srednio": float(np.mean([o["boot5"] for o in okna])),
        "okien_boot5_dodatnich": int(sum(o["boot5"] > 0 for o in okna)),
        # GO wymaga TAKZE dodatniego dolnego przedzialu bootstrapu tygodniowego.
        # Bez tego warunku 9/12 wariantow pierwszego przebiegu pokazywalo GO=True przy
        # boot5 ujemnym we WSZYSTKICH — czyli wynik nieodroznialny od zera. Poprawione 26.09.
        "GO": bool(netto.mean() > 0 and (netto > 0).sum() >= len(okna) * 0.6
                   and sum(o["bije_losowy"] for o in okna) >= len(okna) * 0.6
                   and np.mean([o["boot5"] for o in okna]) > 0),
        "okna": okna,
    }
    os.makedirs(a.wyjscie, exist_ok=True)
    nz = f"{a.wyjscie}/{a.zestaw}_{a.cel}_h{a.horyzont}.json"
    json.dump(wynik, open(nz, "w"), indent=2)
    print(f"\n{'='*70}")
    print(f"NETTO srednio {wynik['netto_srednio']:+.4f}% | dodatnich {wynik['okien_dodatnich']}/{wynik['okien']} "
          f"| bije losowy {wynik['okien_bijacych_losowy']}/{wynik['okien']} | tx/dzien {wynik['tx_na_dzien_srednio']:.1f}")
    print(f"GO = {wynik['GO']}")
    print(f"zapisane: {nz}")


if __name__ == "__main__":
    main()
