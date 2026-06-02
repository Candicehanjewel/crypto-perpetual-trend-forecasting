"""
策略实现模块 - 包含交易策略和结果分析
"""

import pandas as pd
import numpy as np
import talib
from typing import Tuple, Dict, List, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class SmartChannelTradingSystem:
    """
    智能动态通道交易系统（改进版）
    特点：4小时图交易 + 日线趋势过滤
    """
    
    def __init__(self, 
                 base_period: int = 20,
                 band_period: int = 18,
                 stop_multiplier: float = 4.2,
                 trend_threshold: float = 0.35,
                 use_daily_filter: bool = True):
        """
        初始化系统参数（针对以太坊优化）
        """
        self.base_period = base_period
        self.band_period = band_period
        self.stop_multiplier = stop_multiplier
        self.trend_threshold = trend_threshold
        self.use_daily_filter = use_daily_filter
        
        # 状态跟踪
        self.market_state = "未知"
        self.position = 0
        self.entry_price = 0
        self.position_size = 0
        
    def calculate_weighted_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """核心1：双轨道加权计算"""
        def weighted_xma(series, period, weight_factor=0.9):
            """指数加权的移动平均"""
            weights = np.exp(-weight_factor * np.arange(period))
            weights = weights[::-1] / weights.sum()
            return series.rolling(period).apply(lambda x: np.dot(x, weights), raw=True)
        
        # 上轨道：高点加权 + 平滑值加权
        high_xma = weighted_xma(df['high'], self.base_period, 0.8)
        close_smooth = df['close'].rolling(self.base_period).mean()
        
        # 下轨道：低点加权 + 平滑值加权
        low_xma = weighted_xma(df['low'], self.base_period, 0.8)
        
        # 双重加权
        df['upper_band'] = 0.7 * high_xma + 0.3 * close_smooth
        df['lower_band'] = 0.7 * low_xma + 0.3 * close_smooth
        
        # 计算平滑的高低点
        df['smooth_high'] = df['high'].rolling(self.base_period).mean()
        df['smooth_low'] = df['low'].rolling(self.base_period).mean()
        
        return df
    
    def calculate_smart_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """核心2：智能带状识别"""
        def wma(series, period):
            weights = np.arange(1, period + 1)
            return series.rolling(period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        
        # 计算WMA通道
        wma_high = wma(df['high'], self.band_period)
        wma_low = wma(df['low'], self.band_period)
        
        # EMA平滑
        span = max(5, self.band_period // 4)
        df['smart_upper'] = wma_high.ewm(span=span, adjust=False).mean()
        df['smart_lower'] = wma_low.ewm(span=span, adjust=False).mean()
        
        # 计算通道宽度
        df['band_width'] = (df['smart_upper'] - df['smart_lower']) / df['close'].rolling(20).mean()
        
        return df
    
    def calculate_stop_lines(self, df: pd.DataFrame) -> pd.DataFrame:
        """核心3：止损线优化计算"""
        # 计算基础波动率
        df['price_range'] = df['smooth_high'] - df['smooth_low']
        
        # 自适应止损倍数
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        volatility_ratio = atr / df['close'].rolling(20).mean()
        
        # 动态调整倍数
        volatility_adjustment = 0.5 * (volatility_ratio / volatility_ratio.mean() if volatility_ratio.mean() > 0 else 1)
        dynamic_multiplier = self.stop_multiplier * (1 + volatility_adjustment)
        df['dynamic_multiplier'] = dynamic_multiplier
        
        # 计算止损线
        df['sell_stop_line'] = df['smooth_low'] - dynamic_multiplier * df['price_range']
        df['buy_stop_line'] = df['smooth_high'] + dynamic_multiplier * df['price_range']
        
        return df
    
    def classify_market_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """核心4：趋势通道自动分类"""
        states = []
        
        for i in range(len(df)):
            if pd.isna(df['upper_band'].iloc[i]) or pd.isna(df['sell_stop_line'].iloc[i]):
                states.append("未知")
                continue
            
            upper = df['upper_band'].iloc[i]
            lower = df['lower_band'].iloc[i]
            sell_stop = df['sell_stop_line'].iloc[i]
            buy_stop = df['buy_stop_line'].iloc[i]
            
            # 分类逻辑
            if lower > sell_stop and upper > sell_stop:
                state = "上涨趋势"
            elif upper < buy_stop and lower < buy_stop:
                state = "下跌趋势"
            else:
                band_width_ratio = df['band_width'].iloc[i] if i > 0 else 0
                if band_width_ratio < 0.03:
                    state = "窄幅震荡"
                else:
                    state = "宽幅震荡"
            
            states.append(state)
        
        df['market_state'] = states
        return df
    
    def check_daily_filter(self, df: pd.DataFrame, i: int, signal_direction: int) -> bool:
        """
        检查日线过滤条件
        """
        if not self.use_daily_filter or 'daily_trend' not in df.columns:
            return True
        
        daily_trend = df['daily_trend'].iloc[i]
        
        # 日线过滤规则
        if signal_direction == 1:  # 买入信号
            return daily_trend >= 0
        elif signal_direction == -1:  # 卖出信号
            return daily_trend <= 0
        
        return True
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """核心5：信号生成算法（支持日线过滤）"""
        df['signal'] = 0
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        df['signal_reason'] = ""
        
        # 计算辅助指标
        df['prev_upper'] = df['upper_band'].shift(1)
        df['prev_lower'] = df['lower_band'].shift(1)
        
        for i in range(1, len(df)):
            if pd.isna(df['upper_band'].iloc[i]) or pd.isna(df['prev_upper'].iloc[i]):
                continue
            
            current_state = df['market_state'].iloc[i]
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            prev_high = df['high'].iloc[i-1]
            prev_low = df['low'].iloc[i-1]
            
            upper = df['upper_band'].iloc[i]
            lower = df['lower_band'].iloc[i]
            prev_upper = df['prev_upper'].iloc[i]
            prev_lower = df['prev_lower'].iloc[i]
            
            # 条件1：轨道突破检查
            upper_break = (prev_upper < prev_low) and (upper > current_low)
            lower_break = (prev_lower > prev_high) and (lower < current_high)
            
            # 条件2：市场状态确认
            trend_confirmation = False
            reason = ""
            
            if current_state == "上涨趋势" and upper_break:
                trend_confirmation = True
                reason = "上涨趋势中的突破"
            elif current_state == "下跌趋势" and lower_break:
                trend_confirmation = True
                reason = "下跌趋势中的突破"
            elif "震荡" in current_state:
                if upper_break and current_low < df['smooth_low'].iloc[i]:
                    trend_confirmation = True
                    reason = "震荡市超跌反弹"
                elif lower_break and current_high > df['smooth_high'].iloc[i]:
                    trend_confirmation = True
                    reason = "震荡市超涨回调"
            
            # 条件3：避免在极端位置追单
            price_position = (df['close'].iloc[i] - lower) / (upper - lower + 1e-10)
            
            # 生成买入信号
            if upper_break and trend_confirmation and price_position < 0.7:
                if self.check_daily_filter(df, i, 1):
                    df.loc[df.index[i], 'signal'] = 1
                    df.loc[df.index[i], 'stop_loss'] = df['sell_stop_line'].iloc[i]
                    df.loc[df.index[i], 'take_profit'] = upper + (upper - lower) * 1.5
                    df.loc[df.index[i], 'signal_reason'] = reason
            
            # 生成卖出信号
            elif lower_break and trend_confirmation and price_position > 0.3:
                if self.check_daily_filter(df, i, -1):
                    df.loc[df.index[i], 'signal'] = -1
                    df.loc[df.index[i], 'stop_loss'] = df['buy_stop_line'].iloc[i]
                    df.loc[df.index[i], 'take_profit'] = lower - (upper - lower) * 1.5
                    df.loc[df.index[i], 'signal_reason'] = reason
        
        return df
    
    def calculate_position_size(self, capital: float, risk_per_trade: float, 
                               stop_loss: float, entry_price: float) -> float:
        """资金管理"""
        risk_amount = capital * risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0
        
        position_size = risk_amount / risk_per_unit
        
        # 限制最大仓位
        max_position = capital * 0.1 / entry_price
        return min(position_size, max_position)
    
    def run(self, df: pd.DataFrame, initial_capital: float = 10000, 
            risk_per_trade: float = 0.02) -> pd.DataFrame:
        """运行完整系统"""
        print("=" * 60)
        print("以太坊智能通道交易系统 - 开始运行")
        print(f"使用日线过滤: {self.use_daily_filter}")
        print("=" * 60)
        
        # 数据预处理
        if 'daily_trend' in df.columns and self.use_daily_filter:
            print("检测到日线数据，启用日线过滤...")
        
        # 步骤1：计算双轨道
        print("步骤1/5: 计算双轨道...")
        df = self.calculate_weighted_bands(df)
        
        # 步骤2：计算智能通道
        print("步骤2/5: 计算智能通道...")
        df = self.calculate_smart_bands(df)
        
        # 步骤3：计算止损线
        print("步骤3/5: 计算止损线...")
        df = self.calculate_stop_lines(df)
        
        # 步骤4：市场状态分类
        print("步骤4/5: 市场状态分类...")
        df = self.classify_market_state(df)
        
        # 步骤5：生成交易信号
        print("步骤5/5: 生成交易信号...")
        df = self.generate_signals(df)
        
        # 模拟交易
        print("\n开始模拟交易...")
        df['position'] = 0
        df['capital'] = initial_capital
        df['pnl'] = 0
        
        capital = initial_capital
        position = 0
        entry_price = 0
        position_size = 0
        
        trade_log = []
        trade_id = 1
        
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]
            
            # 开仓逻辑
            if position == 0 and signal != 0:
                position = signal
                entry_price = current_price
                stop_loss = df['stop_loss'].iloc[i]
                
                # 计算仓位
                position_size = self.calculate_position_size(
                    capital, risk_per_trade, stop_loss, entry_price
                )
                
                df.loc[df.index[i], 'position'] = position * position_size
                
                # 记录交易
                trade_log.append({
                    'id': trade_id,
                    'date': df.index[i],
                    'type': '买入' if position == 1 else '卖出',
                    'price': entry_price,
                    'size': position_size,
                    'stop_loss': stop_loss,
                    'reason': df['signal_reason'].iloc[i]
                })
                trade_id += 1
            
            # 平仓逻辑
            elif position != 0:
                current_stop_loss = df['stop_loss'].iloc[i-1]
                
                # 止损检查
                stop_loss_triggered = False
                if position == 1 and current_price <= current_stop_loss:
                    stop_loss_triggered = True
                elif position == -1 and current_price >= current_stop_loss:
                    stop_loss_triggered = True
                
                # 止盈检查
                take_profit_price = df['take_profit'].iloc[i-1] if i > 0 else None
                take_profit_triggered = False
                if not pd.isna(take_profit_price):
                    if position == 1 and current_price >= take_profit_price:
                        take_profit_triggered = True
                    elif position == -1 and current_price <= take_profit_price:
                        take_profit_triggered = True
                
                # 反向信号平仓
                reverse_signal = signal == -position
                
                # 执行平仓
                if stop_loss_triggered or take_profit_triggered or reverse_signal:
                    # 计算盈亏
                    pnl = position * (current_price - entry_price) * position_size
                    capital += pnl
                    
                    # 更新记录
                    df.loc[df.index[i], 'position'] = 0
                    df.loc[df.index[i], 'pnl'] = pnl
                    
                    # 更新交易记录
                    if trade_log:
                        last_trade = trade_log[-1]
                        last_trade['exit_date'] = df.index[i]
                        last_trade['exit_price'] = current_price
                        last_trade['pnl'] = pnl
                        last_trade['pnl_percent'] = (pnl / (entry_price * position_size)) * 100 * position
                        last_trade['exit_reason'] = '止损' if stop_loss_triggered else '止盈' if take_profit_triggered else '反向信号'
                    
                    position = 0
                    entry_price = 0
                    position_size = 0
            
            df.loc[df.index[i], 'capital'] = capital
        
        # 输出结果
        final_capital = df['capital'].iloc[-1]
        total_return = (final_capital / initial_capital - 1) * 100
        max_capital = df['capital'].max()
        min_capital = df['capital'].min()
        max_drawdown = (max_capital - min_capital) / max_capital * 100
        
        print(f"\n模拟交易完成!")
        print(f"初始资金: ${initial_capital:,.2f}")
        print(f"最终资金: ${final_capital:,.2f}")
        print(f"总收益率: {total_return:.2f}%")
        print(f"最大回撤: {max_drawdown:.2f}%")
        print(f"总交易次数: {len(trade_log)}")
        
        # 如果有交易记录，显示详细统计
        if trade_log:
            winning_trades = [t for t in trade_log if 'pnl' in t and t['pnl'] > 0]
            losing_trades = [t for t in trade_log if 'pnl' in t and t['pnl'] <= 0]
            
            if winning_trades:
                avg_win = np.mean([t['pnl'] for t in winning_trades])
                max_win = np.max([t['pnl'] for t in winning_trades])
            else:
                avg_win = max_win = 0
                
            if losing_trades:
                avg_loss = np.mean([t['pnl'] for t in losing_trades])
                max_loss = np.min([t['pnl'] for t in losing_trades])
            else:
                avg_loss = max_loss = 0
            
            win_rate = len(winning_trades) / len(trade_log) * 100 if trade_log else 0
            
            print(f"\n交易统计:")
            print(f"胜率: {win_rate:.1f}%")
            print(f"盈利交易: {len(winning_trades)} 笔")
            print(f"亏损交易: {len(losing_trades)} 笔")
            print(f"平均盈利: ${avg_win:.2f}")
            print(f"平均亏损: ${avg_loss:.2f}")
            print(f"最大盈利: ${max_win:.2f}")
            print(f"最大亏损: ${max_loss:.2f}")
        
        return df, trade_log


class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, results_df, trade_log=None):  
        self.results = results_df
        self.trade_log = trade_log or []  # 存储交易日志
    
    def summary(self):
        """输出总结"""
        print("\n" + "="*60)
        print("交易系统表现总结")
        print("="*60)
        
        # 基础统计
        buy_signals = (self.results['signal'] == 1).sum()
        sell_signals = (self.results['signal'] == -1).sum()
        total_signals = buy_signals + sell_signals
        
        print(f"买入信号数量: {buy_signals}")
        print(f"卖出信号数量: {sell_signals}")
        print(f"总信号数量: {total_signals}")
        
        # 市场状态分布
        print(f"\n市场状态分布:")
        state_counts = self.results['market_state'].value_counts()
        for state, count in state_counts.items():
            percentage = count / len(self.results) * 100
            print(f"  {state}: {count}根K线 ({percentage:.1f}%)")
        
        # 日线趋势分布（如果有）
        if 'daily_trend' in self.results.columns:
            print(f"\n日线趋势分布:")
            trend_counts = self.results['daily_trend'].value_counts()
            for trend, count in trend_counts.items():
                percentage = count / len(self.results) * 100
                if trend == 1:
                    print(f"  上涨趋势: {count}根 ({percentage:.1f}%)")
                elif trend == -1:
                    print(f"  下跌趋势: {count}根 ({percentage:.1f}%)")
                else:
                    print(f"  无趋势: {count}根 ({percentage:.1f}%)")

        # 添加交易统计
        if self.trade_log:
            print(f"\n交易统计:")
            print(f"总交易次数: {len(self.trade_log)}")
            
            # 计算胜率
            winning_trades = [t for t in self.trade_log if 'pnl' in t and t['pnl'] > 0]
            losing_trades = [t for t in self.trade_log if 'pnl' in t and t['pnl'] <= 0]
            
            if self.trade_log:
                win_rate = len(winning_trades) / len(self.trade_log) * 100
                print(f"胜率: {win_rate:.1f}%")
                print(f"盈利交易: {len(winning_trades)} 笔")
                print(f"亏损交易: {len(losing_trades)} 笔")
                
                if winning_trades:
                    avg_win = np.mean([t['pnl'] for t in winning_trades])
                    max_win = np.max([t['pnl'] for t in winning_trades])
                    print(f"平均盈利: ${avg_win:.2f}")
                    print(f"最大盈利: ${max_win:.2f}")
                
                if losing_trades:
                    avg_loss = np.mean([t['pnl'] for t in losing_trades])
                    max_loss = np.min([t['pnl'] for t in losing_trades])
                    print(f"平均亏损: ${avg_loss:.2f}")
                    print(f"最大亏损: ${max_loss:.2f}")

    def save_to_csv(self, filename='eth_trading_results.csv'):
        """保存结果到CSV"""
        # 保存主要数据
        self.results.to_csv(filename)
        
        # 如果有交易记录，单独保存
        if self.trade_log:
            trade_df = pd.DataFrame(self.trade_log)
            trade_filename = filename.replace('.csv', '_trades.csv')
            trade_df.to_csv(trade_filename, index=False)
            print(f"交易记录已保存至: {trade_filename}")
        
        print(f"主要结果已保存至: {filename}")
        
    def plot_results(self, save_path=None):
        """绘制结果图表"""
        import matplotlib
        # 尝试使用系统字体
        try:
            # Windows系统
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        except:
            # Linux/Mac系统
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei', 'AppleGothic']
            matplotlib.rcParams['axes.unicode_minus'] = False
    
        plt.rcParams['figure.dpi'] = 150  # 提高分辨率
    
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 图表1：价格和轨道
        ax1 = axes[0]
        ax1.plot(self.results.index, self.results['close'], label='价格', linewidth=1, alpha=0.7)
        ax1.plot(self.results.index, self.results['upper_band'], label='上轨道', linewidth=0.8, alpha=0.7, color='red')
        ax1.plot(self.results.index, self.results['lower_band'], label='下轨道', linewidth=0.8, alpha=0.7, color='green')
        
        # 标记买入信号
        buy_signals = self.results[self.results['signal'] == 1]
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, buy_signals['close'], 
                       color='green', marker='^', s=100, label='买入信号')
        
        # 标记卖出信号
        sell_signals = self.results[self.results['signal'] == -1]
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index, sell_signals['close'], 
                       color='red', marker='v', s=100, label='卖出信号')
        
        ax1.set_title('ETH/USDT 价格与交易信号')
        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图表2：市场状态
        ax2 = axes[1]
        
        # 创建状态颜色映射
        state_colors = {
            '上涨趋势': 'lightgreen',
            '下跌趋势': 'lightcoral',
            '窄幅震荡': 'lightyellow',
            '宽幅震荡': 'lightblue',
            '未知': 'lightgray'
        }
        
        # 为每个状态创建区域
        prev_state = None
        start_idx = None
        
        for i, state in enumerate(self.results['market_state']):
            if state != prev_state:
                if prev_state is not None:
                    end_idx = self.results.index[i-1]
                    ax2.axvspan(start_idx, end_idx, 
                               alpha=0.3, color=state_colors.get(prev_state, 'lightgray'),
                               label=prev_state if prev_state not in ax2.get_legend_handles_labels()[1] else "")
                start_idx = self.results.index[i]
                prev_state = state
        
        # 最后一段
        if prev_state is not None:
            ax2.axvspan(start_idx, self.results.index[-1], 
                       alpha=0.3, color=state_colors.get(prev_state, 'lightgray'),
                       label=prev_state if prev_state not in ax2.get_legend_handles_labels()[1] else "")
        
        ax2.set_title('市场状态分布')
        ax2.set_ylabel('状态')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 图表3：资金曲线
        ax3 = axes[2]
        ax3.plot(self.results.index, self.results['capital'], 
                label='资金曲线', linewidth=2, color='blue')
        
        # 标记最高点和最低点
        max_idx = self.results['capital'].idxmax()
        min_idx = self.results['capital'].idxmin()
        ax3.scatter([max_idx], [self.results.loc[max_idx, 'capital']], 
                   color='green', s=100, label=f'最高: ${self.results.loc[max_idx, "capital"]:,.0f}')
        ax3.scatter([min_idx], [self.results.loc[min_idx, 'capital']], 
                   color='red', s=100, label=f'最低: ${self.results.loc[min_idx, "capital"]:,.0f}')
        
        ax3.set_title('资金曲线')
        ax3.set_ylabel('资金')
        ax3.set_xlabel('时间')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.show()
    
    def save_to_csv(self, filename='eth_trading_results.csv'):
        """保存结果到CSV"""
        # 提取交易记录
        trade_log = self.results['trade_log'].iloc[-1] if 'trade_log' in self.results.columns else []
        
        # 保存主要数据
        self.results.to_csv(filename)
        
        # 如果有交易记录，单独保存
        if trade_log:
            trade_df = pd.DataFrame(trade_log)
            trade_filename = filename.replace('.csv', '_trades.csv')
            trade_df.to_csv(trade_filename, index=False)
            print(f"交易记录已保存至: {trade_filename}")
        
        print(f"主要结果已保存至: {filename}")