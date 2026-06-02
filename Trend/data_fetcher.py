"""
数据获取模块 - 负责从交易所获取和处理数据
"""

import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta

class EthereumDataFetcher:
    """以太坊数据获取器"""
    
    def __init__(self, exchange_name='binance'):
        """
        初始化交易所连接
        
        Args:
            exchange_name: 交易所名称，支持 'binance', 'okx', 'bybit'
        """
        self.exchange_name = exchange_name
        self.exchange = self._init_exchange()
        
    def _init_exchange(self):
        """初始化交易所连接"""
        if self.exchange_name == 'binance':
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                }
            })
        elif self.exchange_name == 'okx':
            exchange = ccxt.okx({
                'enableRateLimit': True,
            })
        elif self.exchange_name == 'bybit':
            exchange = ccxt.bybit({
                'enableRateLimit': True,
            })
        else:
            raise ValueError(f"不支持的交易所: {self.exchange_name}")
            
        print(f" 已连接 {self.exchange_name} 交易所")
        return exchange
    
    def fetch_ohlcv(self, symbol='ETH/USDT', timeframe='4h', limit=1000):
        """
        获取OHLCV数据
        
        Args:
            symbol: 交易对，如 'ETH/USDT'
            timeframe: 时间框架，如 '4h', '1d', '1h'
            limit: 获取K线数量
        """
        print(f"正在获取 {symbol} {timeframe} 数据...")
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('date', inplace=True)
            df.drop('timestamp', axis=1, inplace=True)
            
            print(f"✓ 成功获取 {len(df)} 根 {timeframe} K线")
            print(f"  时间范围: {df.index[0]} 到 {df.index[-1]}")
            
            return df
            
        except Exception as e:
            print(f"✗ 获取数据失败: {e}")
            return None
    
    def fetch_multiple_timeframes(self, symbol='ETH/USDT', main_tf='4h', filter_tf='1d', days=180):
        """
        获取多个时间框架数据
        
        Args:
            symbol: 交易对
            main_tf: 主时间框架（交易用）
            filter_tf: 过滤时间框架（趋势判断用）
            days: 获取天数
        """
        print(f"\n开始获取 {symbol} 数据...")
        print(f"主时间框架: {main_tf}, 过滤时间框架: {filter_tf}")
        print(f"获取最近 {days} 天数据")
        
        # 计算需要获取的K线数量
        tf_to_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360,
            '12h': 720, '1d': 1440
        }
        
        main_minutes = tf_to_minutes.get(main_tf, 240)
        filter_minutes = tf_to_minutes.get(filter_tf, 1440)
        
        # 计算需要的K线数量
        total_minutes = days * 24 * 60
        main_limit = int(total_minutes / main_minutes) + 100
        filter_limit = int(total_minutes / filter_minutes) + 50
        
        print(f"主时间框架需要 {main_limit} 根K线")
        print(f"过滤时间框架需要 {filter_limit} 根K线")
        
        # 获取数据
        main_df = self.fetch_ohlcv(symbol, main_tf, main_limit)
        filter_df = self.fetch_ohlcv(symbol, filter_tf, filter_limit)
        
        if main_df is None or filter_df is None:
            print("✗ 数据获取失败")
            return None, None
        
        return main_df, filter_df
    
    def resample_daily_to_main(self, main_df, daily_df, main_tf='4h'):
        """
        将日线数据重采样到主时间框架
        
        Args:
            main_df: 主时间框架数据
            daily_df: 日线数据
            main_tf: 主时间框架
            
        Returns:
            合并后的DataFrame
        """
        # 确保日期索引
        main_df = main_df.copy()
        daily_df = daily_df.copy()
        
        # 重命名日线列，避免冲突
        daily_df = daily_df.rename(columns={
            'open': 'daily_open',
            'high': 'daily_high',
            'low': 'daily_low',
            'close': 'daily_close',
            'volume': 'daily_volume'
        })
        
        # 计算日线简单趋势指标
        daily_df['daily_sma_20'] = daily_df['daily_close'].rolling(20).mean()
        daily_df['daily_sma_50'] = daily_df['daily_close'].rolling(50).mean()
        daily_df['daily_trend'] = np.where(
            daily_df['daily_sma_20'] > daily_df['daily_sma_50'], 1, -1
        )
        
        # 将日线数据重采样到主时间框架（向前填充）
        daily_df.index = pd.to_datetime(daily_df.index)
        
        # 根据主时间框架设置重采样频率
        tf_to_freq = {
            '1h': 'H',
            '4h': '4H',
            '6h': '6H',
            '12h': '12H'
        }
        
        resample_freq = tf_to_freq.get(main_tf, '4H')
        
        # 重采样并向前填充
        daily_resampled = daily_df.resample(resample_freq).ffill()
        
        # 对齐索引
        daily_resampled = daily_resampled.reindex(main_df.index, method='ffill')
        
        # 合并数据
        merged_df = pd.concat([main_df, daily_resampled], axis=1)
        
        # 填充NaN值
        merged_df['daily_trend'] = merged_df['daily_trend'].fillna(0)
        
        print(f" 成功合并数据，总数据量: {len(merged_df)} 根K线")
        print(f"  日线趋势分布: ")
        trend_counts = merged_df['daily_trend'].value_counts()
        for trend, count in trend_counts.items():
            if trend == 1:
                print(f"    上涨趋势: {count} 根 ({count/len(merged_df)*100:.1f}%)")
            elif trend == -1:
                print(f"    下跌趋势: {count} 根 ({count/len(merged_df)*100:.1f}%)")
            else:
                print(f"    无趋势: {count} 根 ({count/len(merged_df)*100:.1f}%)")
        
        return merged_df