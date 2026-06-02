"""
Main entry for the Chanlun analysis workflow.
1. Fetch OHLCV data
2. Merge inclusion bars
3. Run Chanlun analysis
4. Export chart and CSV outputs
"""

import os
import webbrowser

import pandas as pd
import requests

from chanlun_core import ChanlunAnalyzer, plot_chanlun_chart, process_inclusion


def calculate_macd_area(df, span1=12, span2=26, span3=9):
    """Simple MACD implementation with pandas only."""
    ema12 = df["close"].ewm(span=span1, adjust=False).mean()
    ema26 = df["close"].ewm(span=span2, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=span3, adjust=False).mean()
    hist = macd - signal
    return macd, hist


def _normalize_cn_ohlcv(df, timestamp_col):
    if df.empty:
        return pd.DataFrame()

    rename_map = {
        timestamp_col: "timestamp",
        "日期": "timestamp",
        "时间": "timestamp",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    }
    df = df.rename(columns=rename_map).copy()

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_localize(None)
    df["timestamp"] = timestamps

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df[["timestamp", "open", "high", "low", "close", "volume"]]
        .dropna()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _format_intraday_range(start_date, end_date):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if start_ts.hour == 0 and start_ts.minute == 0 and start_ts.second == 0:
        start_ts = start_ts.replace(hour=9, minute=30, second=0)
    if end_ts.hour == 0 and end_ts.minute == 0 and end_ts.second == 0:
        end_ts = end_ts.replace(hour=15, minute=0, second=0)

    return start_ts.strftime("%Y-%m-%d %H:%M:%S"), end_ts.strftime("%Y-%m-%d %H:%M:%S")


def _candidate_market_ids(symbol):
    if symbol.startswith(("5", "6")):
        return [1, 0]
    if symbol.startswith(("0", "1", "2", "3")):
        return [0, 1]
    return [1, 0]


def _eastmoney_request(url, params):
    session = requests.Session()
    session.trust_env = False
    response = session.get(
        url,
        params=params,
        timeout=15,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()
    return response.json()


def _fetch_eastmoney_daily(symbol, start_date, end_date):
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": "0",
        "beg": pd.Timestamp(start_date).strftime("%Y%m%d"),
        "end": pd.Timestamp(end_date).strftime("%Y%m%d"),
    }

    for market_id in _candidate_market_ids(symbol):
        data_json = _eastmoney_request(url, {**params, "secid": f"{market_id}.{symbol}"})
        if data_json.get("data") and data_json["data"].get("klines"):
            df = pd.DataFrame([item.split(",") for item in data_json["data"]["klines"]])
            df.columns = [
                "日期",
                "开盘",
                "收盘",
                "最高",
                "最低",
                "成交量",
                "成交额",
                "振幅",
                "涨跌幅",
                "涨跌额",
                "换手率",
            ]
            return df

    return pd.DataFrame()


def _fetch_eastmoney_intraday(symbol, timeframe, start_date, end_date):
    start_ts, end_ts = _format_intraday_range(start_date, end_date)

    if timeframe == "1":
        url = "https://push2his.eastmoney.com/api/qt/stock/trends2/get"
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "ndays": "5",
            "iscr": "0",
        }
        columns = ["时间", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "均价"]
        data_key = "trends"
    else:
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "klt": timeframe,
            "fqt": "0",
            "beg": "0",
            "end": "20500000",
        }
        columns = [
            "时间",
            "开盘",
            "收盘",
            "最高",
            "最低",
            "成交量",
            "成交额",
            "振幅",
            "涨跌幅",
            "涨跌额",
            "换手率",
        ]
        data_key = "klines"

    for market_id in _candidate_market_ids(symbol):
        data_json = _eastmoney_request(url, {**params, "secid": f"{market_id}.{symbol}"})
        if data_json.get("data") and data_json["data"].get(data_key):
            df = pd.DataFrame([item.split(",") for item in data_json["data"][data_key]])
            df.columns = columns
            df.index = pd.to_datetime(df["时间"], errors="coerce")
            df = df[start_ts:end_ts].reset_index(drop=True)
            return df

    return pd.DataFrame()


def fetch_sh_etf_ohlcv(symbol="510300", timeframe="30", start_date="2025-01-01", end_date="2026-02-11"):
    """
    Fetch CN ETF or A-share OHLCV data from Eastmoney via AkShare.
    symbol: 510300
    timeframe: 1, 5, 15, 30, 60, daily
    """
    timeframe = str(timeframe).lower()
    print(f"正在通过东方财富获取 {symbol} {timeframe} 数据...")

    try:
        if timeframe == "daily":
            raw_df = _fetch_eastmoney_daily(symbol, start_date, end_date)
            df = _normalize_cn_ohlcv(raw_df, "日期")
            if df.empty:
                print("未获取到日线数据，请检查代码或日期范围。")
            return df

        if timeframe not in {"1", "5", "15", "30", "60"}:
            raise ValueError(f"不支持的周期: {timeframe}")

        raw_df = _fetch_eastmoney_intraday(symbol, timeframe, start_date, end_date)
        df = _normalize_cn_ohlcv(raw_df, "时间")
        if df.empty:
            print("未获取到分钟数据，可能是代码错误，或请求区间超出了东方财富分钟线可回溯范围。")
            return df

        requested_start = pd.Timestamp(start_date)
        actual_start = df["timestamp"].min()
        if actual_start > requested_start:
            print(f"提示: 分钟线实际从 {actual_start} 开始返回，早于该时间的数据未由接口提供。")
        return df

    except Exception as e:
        print(f"获取东方财富数据失败: {e}")
        return pd.DataFrame()


def main():
    SYMBOL = "510300"
    TIMEFRAME = "30"
    START_DATE = "2025-12-15"
    END_DATE = "2026-02-11"

    OUTPUT_CSV = f"sh{SYMBOL}_{TIMEFRAME}m_raw.csv"
    CHART_HTML = f"chanlun_{SYMBOL}_{TIMEFRAME}m.html"

    DIVERGENCE_THRESHOLD = 0.85
    SIGNAL_COOLDOWN = 10
    ENABLE_B2_S2 = True
    ENABLE_B3_S3 = True
    USE_TREND_FILTER = True

    df_raw = fetch_sh_etf_ohlcv(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    if df_raw.empty:
        print("错误: 未能获取到有效数据")
        return

    df_raw.to_csv(OUTPUT_CSV, index=False)

    df_merged = process_inclusion(df_raw)

    trend_data = None
    if USE_TREND_FILTER:
        ma60 = df_merged["close"].rolling(window=60).mean()
        trend_data = (ma60.diff() > 0).map({True: 1, False: -1}).fillna(0).values

    print(f"正在执行缠论分析: {SYMBOL}...")
    analyzer = ChanlunAnalyzer(df_merged)
    analyzer.detect_fractals()
    analyzer.get_strokes()

    signals = analyzer.detect_signals(
        divergence_threshold=DIVERGENCE_THRESHOLD,
        signal_cooldown=SIGNAL_COOLDOWN,
        enable_b2_s2=ENABLE_B2_S2,
        enable_b3_s3=ENABLE_B3_S3,
        trend_filter=trend_data,
    )

    print(f"分析完成，识别到 {len(signals)} 个买卖点。")
    if not signals.empty:
        print(signals[["time", "signal", "price"]].tail(10))

    analyzer.export_to_csv("chanlun_export.csv")
    plot_chanlun_chart(df_merged, analyzer.strokes_df, signals, CHART_HTML)

    full_path = "file://" + os.path.abspath(CHART_HTML)
    print(f"图表已生成: {full_path}")
    webbrowser.open(full_path)


if __name__ == "__main__":
    main()
