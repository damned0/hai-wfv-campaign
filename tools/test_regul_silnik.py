#!/usr/bin/env python3
"""Reguly funding-short / odbicie na zarzadzaniu pozycja SILNIKA + warstwy Risk i Macro (2026-09-14).

Sygnal: hai_common/backtester.py z HAI_REGULA (te same cechy feat_src, ktore widza modele).
Wyjscia jak w silniku: cel/stop z ATR, czesciowe zamkniecie (trigger 0.48, 50%), trailing,
koszty (prowizja 2x0.06%, poslizg), funding, cooldown po SL, limit dzienny — nic wlasnego.
Progi regul wyznaczone na danych do 2024-10-31, wiec liczymy WYLACZNIE wejscia od 2024-11-07.

Warstwa portfela (post-hoc, chronologicznie po wszystkich symbolach):
  limit rownoczesnych pozycji (LIV: 10), opcjonalnie:
  Risk  — limit pozycji w jednym kierunku; filtr trendu BTC (short tylko pod EMA200 1h BTC)
  Macro — blokada wejsc od 12 h przed do 12 h po decyzji FOMC (18:00 UTC drugiego dnia);
          daty z federalreserve.gov/monetarypolicy/fomccalendars.htm. CPI/NFP: brak zrodla
          (BLS 403, FRED niedostepny z VPS) — nie wpisujemy dat z pamieci.

Uzycie: python3 tools/test_regul_silnik.py --regula funding_short --tp 2.5 --sl 1.5 --wyniki X.pkl
"""
import argparse, os, sys, time, logging

ap = argparse.ArgumentParser()
ap.add_argument("--regula", default="", choices=["", "funding_short", "odbicie", "oba", "plik"])
ap.add_argument("--plik", default="", help="--regula plik: parquet z sygnalami (symbol, ts_ms, akcja)")
ap.add_argument("--tp", type=float, default=0.0); ap.add_argument("--sl", type=float, default=0.0)
ap.add_argument("--z-plikow", nargs="*", default=[],
                help="zamiast symulacji: polacz transakcje z wczesniejszych przebiegow (np. dwie reguly, "
                     "kazda z wlasna geometria) i policz portfel/warstwy na calosci")
ap.add_argument("--symbole", default="wl48", help="wl48 | wszystkie")
ap.add_argument("--procesy", type=int, default=3)
ap.add_argument("--od", default="2024-11-07")
ap.add_argument("--wyniki", required=True)
a = ap.parse_args()
# Stale backtestera czytane przy imporcie — ustawiamy PRZED importem (parametry = tools/kampania_baza.sh)
if a.plik:
    os.environ["HAI_REGULA_PLIK"] = a.plik
os.environ.update(HAI_REGULA=a.regula or "brak", HAI_PARTIAL_TRIGGER="0.480", HAI_PARTIAL_FRAC="0.50",
                  HAI_CONF_SIZING="off", HAI_ATR_TP=str(a.tp), HAI_ATR_SL=str(a.sl))
sys.path.insert(0, "/root/ProjektHAI/hai_common"); logging.disable(logging.WARNING)
import numpy as np, pandas as pd
from hai_common.backtester import Backtester, WH_BASE

FOMC = ["2024-11-07", "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
        "2025-09-17", "2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29"]          # drugi dzien posiedzenia (decyzja), zrodlo: federalreserve.gov
FOMC_MS = np.array([pd.Timestamp(d + " 18:00").value // 10**6 for d in FOMC])
OD_MS = pd.Timestamp(a.od).value // 10**6


def jeden(sym):
    bt = Backtester()
    dni = (pd.Timestamp.now() - pd.Timestamp(a.od)).days + 90      # + rozgrzewka wskaznikow
    c1, c4, cd = (bt.load_candles_from_warehouse(sym, tf, dni) for tf in ("1h", "4h", "1d"))
    if len(c1) < 500:
        return []
    tr = bt.run_simulation_ai(c1, c4, cd, f"{sym}/USDT:USDT", atr_tp=a.tp, atr_sl=a.sl)
    return [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in t.items()
             if k not in ("feature_snapshot", "model_votes")} for t in tr if t["open_ts"] >= OD_MS]


def portfel(T, limit=10, limit_kier=None, btc_filtr=False, fomc=False, btc=None):
    """Chronologicznie: wejscie przyjete, gdy jest miejsce w portfelu i przechodzi warstwy."""
    T = T.sort_values(["open_ts", "symbol"]).reset_index(drop=True)
    otwarte = []   # (close_ts, side)
    wziete = np.zeros(len(T), bool); powod = np.array([""] * len(T), dtype=object)
    for i, r in T.iterrows():
        otwarte = [o for o in otwarte if o[0] > r.open_ts]
        if fomc and np.any(np.abs(FOMC_MS - r.open_ts) <= 12 * 3_600_000):
            powod[i] = "macro_fomc"; continue
        if btc_filtr and r.side == "SHORT" and btc is not None:
            j = btc.index.searchsorted(r.open_ts, side="right") - 1
            if j >= 0 and btc.iloc[j] > 0:          # BTC nad EMA200 -> nie shortujemy
                powod[i] = "risk_btc"; continue
        if len(otwarte) >= limit:
            powod[i] = "limit_portfela"; continue
        if limit_kier is not None and sum(1 for o in otwarte if o[1] == r.side) >= limit_kier:
            powod[i] = "risk_kierunek"; continue
        otwarte.append((r.close_ts, r.side)); wziete[i] = True
    return T[wziete], pd.Series(powod[~wziete]).value_counts().to_dict()


def statystyki(T, dni, n_sym):
    if not len(T):
        return dict(n=0)
    p = T.pnl_pct.values
    krzywa = np.cumsum(p); dd = float(np.max(np.maximum.accumulate(krzywa) - krzywa))
    zysk, strata = p[p > 0].sum(), -p[p < 0].sum()
    kw = T.groupby(pd.to_datetime(T.open_ts, unit="ms").dt.to_period("Q")).pnl_pct.sum()
    return dict(n=len(T), tpd=len(T) / dni, tpd_48=len(T) / dni * 48 / n_sym, wr=(p > 0).mean(),
                sr_pct=p.mean(), suma_pct=p.sum(), pf=zysk / strata if strata else np.inf, max_dd_pct=dd,
                kw_plus=int((kw > 0).sum()), kw_n=len(kw), kwartaly=" ".join(f"{q}:{v:+.0f}" for q, v in kw.items()))


if __name__ == "__main__":
    import multiprocessing as mp
    if a.symbole == "wl48":
        from hai_common.symbols import TRADING_SYMBOLS
        symbole = [s.split("/")[0] for s in TRADING_SYMBOLS]
    else:
        from poszukiwania import MARTWE_COINY   # zdjete z gield, zamrozone ceny (2026-09-14)
        symbole = sorted(p.stem for p in (WH_BASE / "1h").glob("*.parquet") if p.stem not in MARTWE_COINY)
    t0 = time.time()
    if a.z_plikow:
        T = pd.concat([pd.read_pickle(f) for f in a.z_plikow], ignore_index=True)
        # jedna pozycja na symbol naraz (jak silnik): pozniejsze wejscie na zajetym symbolu odpada
        T = T.sort_values(["symbol", "open_ts"]).reset_index(drop=True)
        wolny, keep = {}, []
        for r in T.itertuples():
            ok = r.open_ts >= wolny.get(r.symbol, -1); keep.append(ok)
            if ok: wolny[r.symbol] = r.close_ts
        T = T[keep]; a.regula = "polaczone: " + ", ".join(os.path.basename(f) for f in a.z_plikow)
    else:
        if not a.regula or not a.tp or not a.sl:
            sys.exit("podaj --regula, --tp i --sl (albo --z-plikow)")
        with mp.get_context("fork").Pool(a.procesy) as pool:
            wsz = []
            for k, tr in enumerate(pool.imap_unordered(jeden, symbole), 1):
                wsz += tr
                if k % 12 == 0:
                    print(f"  {k}/{len(symbole)} symboli, {len(wsz)} transakcji, {time.time() - t0:.0f} s", flush=True)
        T = pd.DataFrame(wsz)
    if not a.z_plikow:
        T.to_pickle(a.wyniki)
    if not len(T):
        print("brak transakcji"); sys.exit(0)
    dni = (T.close_ts.max() - OD_MS) / 86_400_000; n_sym = T.symbol.nunique()
    b = pd.read_parquet(WH_BASE / "1h" / "BTC.parquet")
    # ms niezaleznie od zapisu w parquecie (magazyn ma datetime64[ms]; bylo astype(int64)//1e6,
    # co przy [ms] dawalo bzdure i filtr BTC widzial zawsze ostatnia wartosc)
    b["ts"] = pd.to_datetime(b.timestamp).astype("datetime64[ms]").astype("int64")
    b = b.drop_duplicates("ts").sort_values("ts")
    btc = pd.Series((b.close - b.close.ewm(span=200, adjust=False).mean()).values, index=b.ts.values)
    print(f"\n=== {a.regula} | cel {a.tp} / stop {a.sl} ATR | {n_sym} symboli z transakcjami | "
          f"{pd.Timestamp(OD_MS, unit='ms').date()} -> {pd.Timestamp(T.close_ts.max(), unit='ms').date()} ({dni:.0f} dni)")
    warianty = [("bez limitu portfela (kazdy sygnal)", dict(limit=10**6)),
                ("limit 10 pozycji (jak LIV)", dict(limit=10)),
                ("Risk: maks. 5 w jednym kierunku", dict(limit=10, limit_kier=5)),
                ("Risk: short tylko gdy BTC pod EMA200", dict(limit=10, btc_filtr=True)),
                ("Macro: bez wejsc +-12 h wokol FOMC", dict(limit=10, fomc=True)),
                ("Risk + Macro razem", dict(limit=10, limit_kier=5, btc_filtr=True, fomc=True))]
    wiersze = []
    for nazwa, kw in warianty:
        W, odrzucone = portfel(T, btc=btc, **kw)
        s = statystyki(W, dni, n_sym); s["wariant"] = nazwa; s["odrzucone"] = odrzucone; wiersze.append(s)
    R = pd.DataFrame(wiersze).set_index("wariant")
    kol = ["n", "tpd", "tpd_48", "wr", "sr_pct", "suma_pct", "pf", "max_dd_pct", "kw_plus", "kw_n"]
    print(R[kol].round(3).to_string())
    print("\nkwartaly (suma % na transakcje), wariant 'limit 10':", R.loc["limit 10 pozycji (jak LIV)", "kwartaly"])
    for nazwa in R.index:
        print(f"  odrzucone [{nazwa}]: {R.loc[nazwa, 'odrzucone']}")
    # uczciwa miara warstw: czy ODRZUCONE wejscia byly gorsze od przyjetych?
    fomc_mask = T.open_ts.apply(lambda x: bool(np.any(np.abs(FOMC_MS - x) <= 12 * 3_600_000)))
    print(f"\nwejscia w oknie FOMC: n={fomc_mask.sum()} WR {(T[fomc_mask].pnl_pct > 0).mean():.1%} "
          f"sr {T[fomc_mask].pnl_pct.mean():+.3f}% | poza oknem: WR {(T[~fomc_mask].pnl_pct > 0).mean():.1%} "
          f"sr {T[~fomc_mask].pnl_pct.mean():+.3f}%")
    j = np.searchsorted(btc.index.values, T.open_ts.values, side="right") - 1
    nad = (btc.values[np.clip(j, 0, None)] > 0) & (T.side.values == "SHORT")
    if (T.side == "SHORT").any():
        print(f"shorty gdy BTC NAD EMA200: n={nad.sum()} WR {(T.pnl_pct[nad] > 0).mean():.1%} sr {T.pnl_pct[nad].mean():+.3f}% | "
              f"pod EMA200: n={((~nad) & (T.side == 'SHORT')).sum()} WR {(T.pnl_pct[(~nad) & (T.side == 'SHORT')] > 0).mean():.1%} "
              f"sr {T.pnl_pct[(~nad) & (T.side == 'SHORT')].mean():+.3f}%")
    print("\nwyniki wg powodu zamkniecia:", T.groupby("result").pnl_pct.agg(["count", "mean"]).round(3).to_dict("index"))
