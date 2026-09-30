#!/usr/bin/env python3
"""STRUKTURA FILTROW: czy kandydaci z drzew to NOWE filtry, czy jeden czynnik (2026-09-30).

30.09 `tools/drzewa_jako_hipotezy.py` wskazal kandydatow na trzeci filtr doktryny.
Dwaj najlepsi — `nad_sr168 <= -10,199` i `sila_wzgl <= -2,584` — to oba warianty
"moneta wlasnie oberwala", wiec moga byc JEDNYM czynnikiem w dwoch przebraniach.
Bez zmierzenia wzajemnego pokrycia liczylibysmy to samo dwa razy.

PROGI SA ZAMROZONE na wartosciach z odkrycia (dane < 2024-03). Caly pomiar leci
WYLACZNIE na sprawdzianie (>= 2024-03), ktorego drzewo nie widzialo. Mnoznik
`--mult` sluzy TYLKO sprawdzeniu, czy efekt nie wisi na dokladnej wartosci progu
— nie jest strojeniem.

CO JEST LICZONE:
  1. macierz pokrycia parami: P(X i Y) wobec P(X)*P(Y) — nadmiar w procentach
  2. kazdy filtr osobno: srednia, LOSOWA KONTROLA tej samej wielkosci (perc. 95),
     bootstrap po dniach (5. percentyl)
  3. wszystkie zlozenia zawierajace F+S — czy trzeci filtr DOKLADA
  4. rozbicie na lata

Kontrola losowa dopasowana rozmiarem jest obowiazkowa: uniewaznila w tym projekcie
meta-etykiete (4x) i bramke sily wzglednej. Bootstrap po DNIACH, bo srednia po
transakcjach klamie przy skupieniu w czasie (komorki z WR 85% mialy 90% transakcji
w pieciu dniach krachu).
"""
import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get("HAI_ROOT", "/root/ProjektHAI")
WAZNE, HZ, OSTROZ, KOSZT = 24, 24, 0.0005, 0.0885
OKNO_ZD = 3
rng = np.random.default_rng(0)

# progi z odkrycia (dane < 2024-03), tools/drzewa_jako_hipotezy.py
PROGI = {"N": -10.199, "W": -2.584, "A": 8.169, "D": -32.392}


def zbuduj(gleb, limit_monet=0):
    pliki = sorted(glob.glob(f"{ROOT}/data_warehouse/ohlcv/binance/1h/*.parquet"))
    if limit_monet:
        pliki = pliki[:limit_monet]
    zwr = {}
    for p in pliki:
        s = os.path.basename(p)[:-8]
        o = pd.read_parquet(p, columns=["timestamp", "close"])
        o["timestamp"] = pd.to_datetime(o.timestamp).dt.tz_localize(None)
        o = o.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        if len(o) < 800:
            continue
        zwr[s] = o.close.pct_change()
    R = pd.DataFrame(zwr).sort_index()
    IDX = (1 + R.mean(axis=1).fillna(0.0)).cumprod()
    RYN24 = IDX.pct_change(24) * 100
    print(f"indeks rynku: {len(IDX):,} h, {len(zwr)} monet", flush=True)

    T = []
    for p in pliki:
        sym = os.path.basename(p)[:-8]
        o = pd.read_parquet(p, columns=["timestamp", "high", "low", "close", "volume"])
        o["timestamp"] = pd.to_datetime(o.timestamp).dt.tz_localize(None)
        o = o.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        o = o[o.timestamp >= "2021-08-01"].reset_index(drop=True)
        if len(o) < 800:
            continue
        H, L, C, V = o.high, o.low, o.close, o.volume
        c = C.to_numpy(float); lo = L.to_numpy(float)
        mi = IDX.reindex(o.timestamp).to_numpy(float)
        if not np.isfinite(mi).all():
            mi = pd.Series(mi).ffill().bfill().to_numpy()
        r24 = RYN24.reindex(o.timestamp).to_numpy(float)
        atr = (H.rolling(14).max() - L.rolling(14).min()).to_numpy(float)
        sr = C.rolling(20).mean(); od = C.rolling(20).std()
        bbw = ((4 * od) / sr * 100).to_numpy(float)
        bmed = pd.Series(bbw).rolling(720, min_periods=240).median().to_numpy()
        tk = ((H.rolling(9).max() + L.rolling(9).min()) / 2).to_numpy(float)
        kj = ((H.rolling(26).max() + L.rolling(26).min()) / 2).to_numpy(float)
        sa = ((pd.Series(tk) + pd.Series(kj)) / 2).shift(26)
        sb = ((H.rolling(52).max() + L.rolling(52).min()) / 2).shift(26)
        kud = pd.concat([sa, sb], axis=1).min(axis=1).to_numpy(float)
        d = C.diff()
        zy = d.clip(lower=0).rolling(14).mean(); sd_ = (-d.clip(upper=0)).rolling(14).mean()
        rsi = (100 - 100 / (1 + zy / sd_.replace(0, np.nan))).to_numpy(float)
        vz = ((V - V.rolling(168).mean()) / V.rolling(168).std()).to_numpy(float)
        mn24 = L.rolling(24).min().shift(1)
        e_swd = (((L < mn24) & (C > mn24)).rolling(OKNO_ZD, min_periods=1).max()
                 .shift(1).fillna(0).to_numpy() > 0)
        mx720 = H.rolling(720, min_periods=240).max().to_numpy(float)
        sr168 = C.rolling(168).mean().to_numpy(float)
        ret24 = (C.pct_change(24) * 100).to_numpy(float)

        kon = len(c) - WAZNE - HZ - 2
        if kon <= 800:
            continue
        t = np.arange(800, kon)
        poz4 = ((rsi[t] < 40).astype(int) + (c[t] < kud[t])
                + (tk[t] <= kj[t]) + (vz[t] > 1.0))
        Lw = c[t] - gleb * atr[t]
        wa = np.isfinite(Lw) & (Lw > 0) & (bbw[t] > bmed[t]) & (poz4 >= 2)
        fk = np.full(len(t), -1, dtype=np.int64)
        for k in range(1, WAZNE + 1):
            j = (fk < 0) & wa & (lo[t + k] <= Lw * (1 - OSTROZ))
            fk[j] = k
        m = fk > 0
        if m.sum() == 0:
            continue
        we = t[m] + fk[m]; wy = we + HZ
        rm = (mi[wy] / mi[we] - 1) * 100
        i = t[m]
        T.append(pd.DataFrame({
            "sym": sym,
            "t_plac": o.timestamp.values[i],
            "t": o.timestamp.values[we],
            "atrp": atr[i] / c[i] * 100,
            "swd": e_swd[i].astype(int),
            "nad_sr168": (c[i] / np.where(sr168[i] > 0, sr168[i], np.nan) - 1) * 100,
            "sila_wzgl": ret24[i] - r24[i],
            "dd720": (c[i] / np.where(mx720[i] > 0, mx720[i], np.nan) - 1) * 100,
            "r": ((c[wy] / Lw[m] - 1) * 100 - rm) - KOSZT}))
    D = pd.concat(T, ignore_index=True)
    D["rok"] = pd.to_datetime(D.t).dt.year
    D["dzien"] = pd.to_datetime(D.t).dt.floor("D")
    D["kw_atr"] = D.groupby("t_plac").atrp.transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop"))
    return D.dropna(subset=["kw_atr", "nad_sr168", "sila_wzgl", "dd720", "r"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gleb", type=float, default=2.0)
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--podzial", default="2024-03-01")
    ap.add_argument("--limit-monet", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    print(f"### glebokosc {a.gleb} ATR | mnoznik progow {a.mult} | "
          f"sprawdzian od {a.podzial}\n", flush=True)

    D = zbuduj(a.gleb, a.limit_monet)
    SPR = D[pd.to_datetime(D.t) >= pd.Timestamp(a.podzial)].reset_index(drop=True)
    print(f"wypelnien razem {len(D):,} | sprawdzian {len(SPR):,} | "
          f"baza sprawdzianu {SPR.r.mean():+.4f}\n", flush=True)
    if len(SPR) < 2000:
        print("za malo danych na sprawdzian — koniec"); return

    mlt = a.mult
    FIL = {
        "F": (SPR.kw_atr >= 2).to_numpy(),                      # ATR gorne 3 kwintyle
        "S": (SPR.swd == 1).to_numpy(),                         # zmiecenie pod dolkiem 3 h
        "N": (SPR.nad_sr168 <= PROGI["N"] * mlt).to_numpy(),    # pod srednia 168 h
        "W": (SPR.sila_wzgl <= PROGI["W"] * mlt).to_numpy(),    # slabszy od rynku
        "A": (SPR.atrp >= PROGI["A"] * mlt).to_numpy(),         # ATR bezwzgledny
        "D": (SPR.dd720 <= PROGI["D"] * mlt).to_numpy(),        # obsuniecie od szczytu 720 h
    }
    OPIS = {"F": "ATR gorne 3 kw. (przekrojowo)", "S": "zmiecenie pod dolkiem 3 h",
            "N": f"cena <= {PROGI['N']*mlt:.2f}% pod sr.168h", "W": f"sila wzgl. <= {PROGI['W']*mlt:.2f}%",
            "A": f"ATR% >= {PROGI['A']*mlt:.2f}", "D": f"obsuniecie <= {PROGI['D']*mlt:.2f}% od szczytu 720h"}
    rv = SPR.r.to_numpy()
    baza = rv.mean()

    def kontrola(n, nboot=2000):
        return np.percentile(rv[rng.integers(0, len(rv), size=(nboot, n))].mean(axis=1), 95)

    def boot5(mm, nboot=800):
        g = SPR[mm].groupby("dzien").r.agg(["sum", "count"])
        su = g["sum"].to_numpy(float); li = g["count"].to_numpy(float)
        if len(su) < 2:
            return np.nan
        i = rng.integers(0, len(su), size=(nboot, len(su)))
        return np.percentile(su[i].sum(axis=1) / li[i].sum(axis=1), 5)

    print("=" * 96)
    print("1. MACIERZ POKRYCIA — nadmiar wobec niezaleznosci, w procentach")
    print("   (okolo 0 = rozne informacje; duzo powyzej = to samo w innym przebraniu)")
    print("=" * 96)
    kl = list(FIL)
    print("        " + "".join(f"{k:>9}" for k in kl))
    for x in kl:
        wiersz = ""
        for y in kl:
            if x == y:
                wiersz += f"{'—':>9}"
            else:
                ocz = FIL[x].mean() * FIL[y].mean()
                wiersz += f"{((FIL[x] & FIL[y]).mean() / ocz - 1) * 100:>+8.1f}%" if ocz > 0 else f"{'n/d':>9}"
        print(f"  {x:<6}" + wiersz)

    print("\n" + "=" * 110)
    print(f"2. KAZDY FILTR OSOBNO — baza {baza:+.4f} na {len(SPR):,} transakcjach")
    print("=" * 110)
    print(f"  {'':<3} {'opis':<34} {'tx':>7} {'udzial':>7} {'srednia':>9} "
          f"{'losowa95':>9} {'boot5':>9} {'werdykt':>9}")
    for k in kl:
        mm = FIL[k]; n = int(mm.sum())
        if n < 300:
            print(f"  {k:<3} {OPIS[k]:<34} {n:>7,}  za malo"); continue
        sr_ = rv[mm].mean(); los = kontrola(n)
        print(f"  {k:<3} {OPIS[k]:<34} {n:>7,} {mm.mean():>6.1%} {sr_:>+9.4f} "
              f"{los:>+9.4f} {boot5(mm):>+9.4f} {'PRZESZEDL' if sr_ > los else '—':>9}")

    print("\n" + "=" * 110)
    print("3. ZLOZENIA — czy trzeci (i czwarty) filtr DOKLADA do F+S")
    print("=" * 110)
    fs = FIL["F"] & FIL["S"]
    print(f"  {'F+S':<14} {int(fs.sum()):>7,} {rv[fs].mean():>+9.4f} "
          f"{kontrola(int(fs.sum())):>+9.4f} {boot5(fs):>+9.4f}")
    poz = [k for k in kl if k not in ("F", "S")]
    for rz in (1, 2):
        for kombo in itertools.combinations(poz, rz):
            mm = fs.copy()
            for k in kombo:
                mm = mm & FIL[k]
            n = int(mm.sum())
            nazwa = "F+S+" + "+".join(kombo)
            if n < 250:
                print(f"  {nazwa:<14} {n:>7,}  za malo"); continue
            sr_ = rv[mm].mean(); los = kontrola(n)
            print(f"  {nazwa:<14} {n:>7,} {sr_:>+9.4f} {los:>+9.4f} {boot5(mm):>+9.4f} "
                  f"  przyrost {sr_ - rv[fs].mean():>+7.4f} {'OK' if sr_ > los else '—'}")

    print("\n" + "=" * 110)
    print("4. ROZBICIE NA LATA")
    print("=" * 110)
    for nazwa, mm in [("F+S", fs)] + [(f"F+S+{k}", fs & FIL[k]) for k in poz]:
        sub = SPR[mm]
        if len(sub) < 250:
            continue
        lat = " | ".join(f"{r}: {sub[sub.rok == r].r.mean():+.3f} (n={(sub.rok == r).sum()})"
                         for r in sorted(sub.rok.unique()) if (sub.rok == r).sum() > 40)
        print(f"  {nazwa:<10} {lat}")

    if a.out:
        w = []
        for k in kl:
            mm = FIL[k]
            if mm.sum() < 300:
                continue
            w.append({"gleb": a.gleb, "mult": a.mult, "filtr": k, "n": int(mm.sum()),
                      "srednia": rv[mm].mean(), "losowa95": kontrola(int(mm.sum())),
                      "boot5": boot5(mm)})
        for rz in (1, 2):
            for kombo in itertools.combinations(poz, rz):
                mm = fs.copy()
                for k in kombo:
                    mm = mm & FIL[k]
                if mm.sum() < 250:
                    continue
                w.append({"gleb": a.gleb, "mult": a.mult, "filtr": "F+S+" + "+".join(kombo),
                          "n": int(mm.sum()), "srednia": rv[mm].mean(),
                          "losowa95": kontrola(int(mm.sum())), "boot5": boot5(mm)})
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        pd.DataFrame(w).to_csv(a.out, index=False)
        print(f"\nzapisano {a.out}")


if __name__ == "__main__":
    main()
