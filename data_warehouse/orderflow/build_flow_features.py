#!/usr/bin/env python3
# ===========================================
# gen.Flow — agregacja tradow -> cechy order-flow per swieca 1h
# ===========================================
# Ze znormalizowanych tradow (ts/price/qty/side z dowolnego providera) liczy
# cechy, ktorych DRZEWA NIE WIDZA z OHLCV (agresja/absorpcja, mikrostruktura):
#   delta_1h        = vol_agresywnego_kupna - vol_agresywnej_sprzedazy (w oknie)
#   delta_pct       = delta / total_vol  (znormalizowana agresja)
#   cvd             = skumulowana delta (running) - to na czym bazuje dywergencja
#   buy_vol/sell_vol= wolumeny agresorow
#   trades_n        = liczba tradow (aktywnosc)
#   large_delta_pct = delta liczona TYLKO z duzych tradow (>90 percentyl qty) /
#                     total - proxy footprintu (kto duzy jest agresorem)
#   absorption      = |delta_pct| przy MALYM ruchu ceny = absorpcja agresji
#                     przez pasywne limity (sygnal usera: CVD HH, cena LH).
#                     Liczona jako |delta_pct| * (1 - |price_ret|/atr_proxy).
# Wynik: ramka per swieca 1h (timestamp = poczatek swiecy) do parquet.
# ===========================================
import numpy as np
import pandas as pd


def build_flow_candles(trades: list, tf_ms: int = 3600_000) -> pd.DataFrame:
    """Trady -> cechy order-flow per swieca tf (domyslnie 1h)."""
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["bucket"] = (df["ts"] // tf_ms) * tf_ms
    df["signed_qty"] = df["qty"] * df["side"]
    # prog "duzego" tradu = 90 percentyl qty w calym oknie (footprint proxy)
    big_thr = df["qty"].quantile(0.90) if len(df) > 20 else df["qty"].max()
    df["big_signed"] = np.where(df["qty"] >= big_thr, df["signed_qty"], 0.0)

    g = df.groupby("bucket")
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(g["bucket"].first().values, unit="ms"),
        "of_delta": g["signed_qty"].sum().values,
        "of_buy_vol": g.apply(lambda x: x.loc[x.side == 1, "qty"].sum(), include_groups=False).values,
        "of_sell_vol": g.apply(lambda x: x.loc[x.side == -1, "qty"].sum(), include_groups=False).values,
        "of_trades_n": g.size().values,
        "of_big_delta": g["big_signed"].sum().values,
        "of_close": g["price"].last().values,
        "of_open": g["price"].first().values,
    })
    tot = (out["of_buy_vol"] + out["of_sell_vol"]).replace(0, np.nan)
    out["of_delta_pct"] = (out["of_delta"] / tot).fillna(0.0)
    out["of_big_delta_pct"] = (out["of_big_delta"] / tot).fillna(0.0)
    out["of_cvd"] = out["of_delta"].cumsum()
    # absorpcja: duza delta przy malym ruchu ceny wewnatrz swiecy
    ret = ((out["of_close"] - out["of_open"]) / out["of_open"].replace(0, np.nan)).fillna(0.0)
    ret_scale = ret.abs().rolling(24, min_periods=1).mean().replace(0, np.nan)
    move_factor = (1.0 - (ret.abs() / ret_scale).clip(0, 1)).fillna(0.5)
    out["of_absorption"] = (out["of_delta_pct"].abs() * move_factor).round(4)
    for c in ["of_delta", "of_buy_vol", "of_sell_vol", "of_big_delta", "of_cvd",
              "of_delta_pct", "of_big_delta_pct"]:
        out[c] = out[c].round(4)
    return out.drop(columns=["of_open", "of_close"]).reset_index(drop=True)


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, ".")
    from providers import get_provider
    p = get_provider("binance")
    now = int(time.time() * 1000)
    tr = p.fetch_trades("BTC", now - 3 * 3600_000, now)  # 3h
    print(f"tradow: {len(tr)}")
    fc = build_flow_candles(tr)
    print(fc.to_string())
