# chanlun_core.py
"""
整合模块：包含关系处理 + 缠论分析（MACD加强版） + 可视化 + CSV导出
"""

import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import plotly.graph_objects as go

def calculate_macd_area(df, span1=12, span2=26, span3=9):
    """手写简易MACD，避免外部 ta 依赖"""
    ema12 = df['close'].ewm(span=span1, adjust=False).mean()
    ema26 = df['close'].ewm(span=span2, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=span3, adjust=False).mean()
    hist = macd - signal
    return macd, hist

# ======================
# 第一部分：包含关系处理
# ======================

def process_inclusion(df: pd.DataFrame) -> pd.DataFrame:
    """严格处理K线包含关系，输出连续时间序列"""
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column")
    
    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    if len(df) < 2: 
        return df[['timestamp', 'open', 'high', 'low', 'close']].copy()

    rows = []
    for _, r in df.iterrows():
        rows.append({
            'timestamp': r['timestamp'],
            'open': float(r['open']),
            'high': float(r['high']),
            'low': float(r['low']),
            'close': float(r['close'])
        })

    merged_rows = [rows[0]]
    i = 1
    while i < len(rows):
        current = rows[i]
        last = merged_rows[-1]

        # 判断是否包含
        is_included = (
            (current['high'] <= last['high'] and current['low'] >= last['low']) or
            (current['high'] >= last['high'] and current['low'] <= last['low'])
        )

        if is_included:
            # 更新前一根K线的高低点（保持缠论处理逻辑）
            new_high = max(current['high'], last['high'])
            new_low = min(current['low'], last['low'])
            merged_rows[-1] = {
                'timestamp': current['timestamp'],
                'open': last['open'],
                'high': new_high,
                'low': new_low,
                'close': current['close']
            }
        else:
            merged_rows.append(current)
        i += 1

    result_df = pd.DataFrame(merged_rows)
    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
    return result_df.reset_index(drop=True)


# ======================
# 第二部分：缠论分析引擎
# ======================

class ChanlunAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()
        # 不再调用 ta.macd，直接用手写函数
        macd_line, macd_hist = calculate_macd_area(self.df)
        self.df['macd_h'] = macd_line
        self.df['macd_hist'] = macd_hist
        
        self.df['is_fractal'] = 0
        self.fractals = None
        self.strokes_df = None
        self.signals_df = None
        
        
        self.df['is_fractal'] = 0
        self.fractals = None
        self.strokes_df = None
        self.signals_df = None

    def detect_fractals(self) -> pd.DataFrame:
        df = self.df
        n = len(df)
        is_fractal = [0] * n
        for i in range(2, n - 2):
            highs = df.iloc[i-2:i+3]['high'].values
            lows = df.iloc[i-2:i+3]['low'].values
            # 顶分型
            if (highs[2] == max(highs) and highs[1] < highs[2] and highs[3] < highs[2]):
                is_fractal[i] = 1
            # 底分型
            elif (lows[2] == min(lows) and lows[1] > lows[2] and lows[3] > lows[2]):
                is_fractal[i] = -1
        self.df['is_fractal'] = is_fractal
        fractals = self.df[self.df['is_fractal'] != 0].copy()
        fractals = fractals.reset_index().rename(columns={'index': 'original_idx'})
        self.fractals = fractals[['original_idx', 'is_fractal', 'high', 'low']].rename(columns={'original_idx': 'index'})
        return self.fractals

    def get_strokes(self) -> pd.DataFrame:
        if self.fractals is None: self.detect_fractals()
        fracs = self.fractals
        if len(fracs) < 2: 
            self.strokes_df = pd.DataFrame()
            return self.strokes_df

        # 严格笔逻辑：相邻分型方向相反，且K线数量包含处理后间隔至少4根
        valid_fracs = [fracs.iloc[0].to_dict()]
        for i in range(1, len(fracs)):
            curr = fracs.iloc[i].to_dict()
            last = valid_fracs[-1]
            if curr['is_fractal'] != last['is_fractal'] and (curr['index'] - last['index']) >= 4:
                valid_fracs.append(curr)
        
        strokes = []
        for i in range(1, len(valid_fracs)):
            p1, p2 = valid_fracs[i-1], valid_fracs[i]
            direction = 'up' if p2['is_fractal'] == 1 else 'down'
            
            # 【关键修复】MACD面积计算：上涨笔只算红柱，下跌笔只算绿柱
            hist = self.df.iloc[int(p1['index']):int(p2['index'])]['macd_hist']
            if direction == 'up':
                area = hist[hist > 0].sum()  # 只计算红柱
            else:
                area = hist[hist < 0].abs().sum()  # 只计算绿柱绝对值
            
            strokes.append({
                'start_idx': int(p1['index']),
                'end_idx': int(p2['index']),
                'start_price': float(p1['low'] if direction == 'up' else p1['high']),
                'end_price': float(p2['high'] if direction == 'up' else p2['low']),
                'direction': direction,
                'macd_area': area,
                'amplitude': abs(p2['high'] - p1['low']) if direction == 'up' else abs(p1['high'] - p2['low'])
            })
        self.strokes_df = pd.DataFrame(strokes)
        return self.strokes_df

    def _check_trend_filter(self, idx, side, trend_filter):
        """趋势过滤器实现：基于MA60或自定义序列"""
        if trend_filter is None or idx >= len(trend_filter):
            return True
        # 如果是买点(B)，要求趋势向上(1)；如果是卖点(S)，要求趋势向下(-1)
        current_trend = trend_filter[int(idx)]
        if side == 'B':
            return current_trend >= 0  # 允许震荡或上涨
        else:
            return current_trend <= 0
        return True

    def detect_signals(self, 
                       divergence_threshold=0.85,    # 适配A股，稍微放宽背驰要求
                       signal_cooldown=10,           # 信号冷却，防止密集操作
                       enable_b2_s2=True, 
                       enable_b3_s3=True, 
                       trend_filter=None             # 建议传入：df['close'].rolling(60).mean().diff()
                       ) -> pd.DataFrame:
        """
        深度优化版：平衡逻辑严谨度与A股实战信号捕捉
        """
        if self.strokes_df is None: self.get_strokes()
        stk = self.strokes_df
        signals = []
        signal_positions = {'B': -999, 'S': -999} 
        
        if stk is None or len(stk) < 5: 
            self.signals_df = pd.DataFrame(columns=['idx', 'time', 'signal', 'side', 'price', 'stroke_idx', 'description'])
            return self.signals_df

        # === 第一轮：B1/S1 (MACD面积背驰) ===
        for i in range(2, len(stk)):
            curr = stk.iloc[i]
            # 寻找同方向的前一笔进行力度对比
            prev_same = None
            for j in range(i-2, -1, -2):
                if stk.iloc[j]['direction'] == curr['direction']:
                    prev_same = stk.iloc[j]
                    break
            
            if prev_same is not None:
                # 价格创新高/新低
                price_new = (curr['end_price'] > prev_same['end_price'] if curr['direction'] == 'up' 
                           else curr['end_price'] < prev_same['end_price'])
                # 力度衰减判定
                macd_divergence = curr['macd_area'] < prev_same['macd_area'] * divergence_threshold
                
                if price_new and macd_divergence:
                    sig_type = 'S1' if curr['direction'] == 'up' else 'B1'
                    side_key = sig_type[0]
                    
                    # 冷却期检查
                    if curr['end_idx'] - signal_positions[side_key] >= signal_cooldown:
                        if self._check_trend_filter(curr['end_idx'], side_key, trend_filter):
                            signals.append(self._create_sig(curr, sig_type, i, 
                                          f"背驰确认:面积缩减至{curr['macd_area']/prev_same['macd_area']:.1%}"))
                            signal_positions[side_key] = curr['end_idx']

        # === 第二轮：B2/S2 (类二买/卖，回调确认) ===
        if enable_b2_s2:
            for i in range(4, len(stk)):
                curr = stk.iloc[i]  # 当前笔（回调笔）
                prev = stk.iloc[i-1] # 前一笔（主升/跌笔）
                
                if curr['direction'] == 'down':
                    # B2判定：回调不创新低 + 力度弱于主升笔
                    is_higher_low = curr['end_price'] > stk.iloc[i-2]['start_price'] * 0.998
                    retracement = abs(curr['end_price'] - curr['start_price'])
                    prev_advance = abs(prev['end_price'] - prev['start_price'])
                    # A股中，回调在 0.618 以内通常都是健康的二买机会
                    is_shallow = retracement < prev_advance * 0.618
                    
                    if is_higher_low and is_shallow:
                        if curr['end_idx'] - signal_positions['B'] >= signal_cooldown:
                            signals.append(self._create_sig(curr, 'B2', i, f"二买:不创新低+回调{retracement/prev_advance:.1%}"))
                            signal_positions['B'] = curr['end_idx']
                
                elif curr['direction'] == 'up':
                    # S2判定：对称逻辑
                    is_lower_high = curr['end_price'] < stk.iloc[i-2]['start_price'] * 1.002
                    rebound = abs(curr['end_price'] - curr['start_price'])
                    prev_decline = abs(prev['end_price'] - prev['start_price'])
                    is_shallow_rebound = rebound < prev_decline * 0.618
                    
                    if is_lower_high and is_shallow_rebound:
                        if curr['end_idx'] - signal_positions['S'] >= signal_cooldown:
                            signals.append(self._create_sig(curr, 'S2', i, f"二卖:反弹无力{rebound/prev_decline:.1%}"))
                            signal_positions['S'] = curr['end_idx']

        # === 第三轮：B3/S3 (中枢突破回踩) ===
        if enable_b3_s3:
            marked_idxs = set()
            for i in range(len(stk) - 4):
                s1, s2, s3 = stk.iloc[i], stk.iloc[i+1], stk.iloc[i+2]
                zgn = min(max(s1['start_price'], s1['end_price']), max(s2['start_price'], s2['end_price']))
                zdd = max(min(s1['start_price'], s1['end_price']), min(s2['start_price'], s2['end_price']))
                
                if zdd < zgn: # 存在合法中枢
                    # 检查后续笔是否形成突破回踩
                    for j in range(i+3, min(i+12, len(stk))):
                        test = stk.iloc[j]
                        if j in marked_idxs: continue
                        
                        # B3: 突破后回踩不入中枢 (容差 0.2%)
                        if test['direction'] == 'down' and test['end_price'] > zgn * 0.998:
                            # 确认前一笔确实产生过突破
                            if stk.iloc[j-1]['end_price'] > zgn * 1.01:
                                if test['end_idx'] - signal_positions['B'] >= signal_cooldown:
                                    signals.append(self._create_sig(test, 'B3', j, "三买:中枢上方强支撑"))
                                    signal_positions['B'] = test['end_idx']
                                    marked_idxs.add(j)
                                    break
                                    
                        # S3: 对称逻辑
                        elif test['direction'] == 'up' and test['end_price'] < zdd * 1.002:
                            if stk.iloc[j-1]['end_price'] < zdd * 0.99:
                                if test['end_idx'] - signal_positions['S'] >= signal_cooldown:
                                    signals.append(self._create_sig(test, 'S3', j, "三卖:突破回抽遇阻"))
                                    signal_positions['S'] = test['end_idx']
                                    marked_idxs.add(j)
                                    break

        # 整理输出
        sig_df = pd.DataFrame(signals)
        if not sig_df.empty:
            # 优先级排序：三买 > 二买 > 一买
            prio = {'B3':3, 'S3':3, 'B2':2, 'S2':2, 'B1':1, 'S1':1}
            sig_df['p'] = sig_df['signal'].map(prio)
            sig_df = sig_df.sort_values('p', ascending=False).drop_duplicates(subset=['idx']).drop(columns=['p'])
            self.signals_df = sig_df.sort_values('idx')
        else:
            self.signals_df = pd.DataFrame(columns=['idx', 'time', 'signal', 'side', 'price', 'stroke_idx', 'description'])
            
        return self.signals_df
    
    def _check_trend_filter(self, idx, side, trend_filter):
        """趋势过滤器：side='B'时检查是否处于上升趋势"""
        if 'ma60' not in self.df.columns:
            return True  # 没有MA60就不过滤
        
        ma60_current = self.df.at[idx, 'ma60']
        ma60_before = self.df.at[max(0, idx-20), 'ma60']
        
        if side == 'B':
            return ma60_current > ma60_before  # 买点需要均线向上
        else:
            return ma60_current < ma60_before  # 卖点需要均线向下

    def _create_sig(self, stroke, sig_name, s_idx, description="缠论确认"):
        idx = int(stroke['end_idx'])
        side = 'buy' if sig_name.startswith('B') else 'sell'
        return {
            'idx': idx, 'time': self.df.at[idx, 'timestamp'] if 'timestamp' in self.df.columns else idx,
            'signal': sig_name, 'side': side, 'price': float(stroke['end_price']),
            'stroke_idx': s_idx, 'description': f"{sig_name}: {description}"
        }

    def export_to_csv(self, filename: str):
        """适配原版 main.py 的导出接口"""
        if self.strokes_df is not None:
            self.strokes_df.to_csv(filename.replace('.csv', '_strokes.csv'), index=False)
        if self.signals_df is not None:
            self.signals_df.to_csv(filename.replace('.csv', '_signals.csv'), index=False)


# ======================
# 第三部分：可视化
# ======================

def plot_chanlun_chart(df: pd.DataFrame, strokes: pd.DataFrame, signals: pd.DataFrame, out_html='chanlun_chart.html'):
    use_time = 'timestamp' in df.columns
    x_axis = df['timestamp'] if use_time else df.index
    
    fig = go.Figure()
    
    # K线图
    fig.add_trace(go.Candlestick(
        x=x_axis, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='K线', increasing_line_color='lightgray', decreasing_line_color='lightgray'
    ))
    
    # 绘制笔 (Strokes)
    for _, s in strokes.iterrows():
        t0 = df.loc[int(s['start_idx']), 'timestamp'] if use_time else int(s['start_idx'])
        t1 = df.loc[int(s['end_idx']), 'timestamp'] if use_time else int(s['end_idx'])
        fig.add_trace(go.Scatter(
            x=[t0, t1], y=[s['start_price'], s['end_price']],
            mode='lines', line=dict(color='black', width=2), showlegend=False
        ))
        
    # 绘制信号
    if not signals.empty:
        sig_styles = {
            'B1': ('triangle-up', 'green'), 'B2': ('pentagon', 'darkgreen'), 'B3': ('star', 'lime'),
            'S1': ('triangle-down', 'red'), 'S2': ('pentagon', 'darkred'), 'S3': ('star', 'orange')
        }
        for sig_type, (symbol, color) in sig_styles.items():
            subset = signals[signals['signal'] == sig_type]
            if subset.empty: continue
            fig.add_trace(go.Scatter(
                x=subset['time'], y=subset['price'],
                mode='markers+text', name=sig_type, text=sig_type,
                marker=dict(symbol=symbol, color=color, size=12),
                textposition="top center"
            ))

    fig.update_layout(title='缠论分析加强版 (MACD背驰判定)', xaxis_rangeslider_visible=False, template='plotly_white')
    fig.write_html(out_html)
    print(f"图表已保存至: {os.path.abspath(out_html)}")
    return fig