"""
主程序入口 - 整合数据获取和策略执行
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_fetcher import EthereumDataFetcher
from strategy import SmartChannelTradingSystem, ResultAnalyzer


def main():
    """主函数"""
    print("以太坊智能通道交易系统")
    print("=" * 60)
    
    # 1. 获取数据
    print("\n[1/4] 获取数据...")
    fetcher = EthereumDataFetcher('binance')
    
    # 获取4小时和日线数据
    main_df, daily_df = fetcher.fetch_multiple_timeframes(
        symbol='ETH/USDT',
        main_tf='4h',
        filter_tf='1d',
        days=180  # 获取180天数据
    )
    
    if main_df is None or daily_df is None:
        print("数据获取失败，退出程序")
        return
    
    # 合并数据
    merged_df = fetcher.resample_daily_to_main(main_df, daily_df, '4h')
    
    # 2. 初始化交易系统
    print("\n[2/4] 初始化交易系统...")
    system = SmartChannelTradingSystem(
        base_period=20,          # 针对ETH优化
        band_period=18,
        stop_multiplier=4.2,
        trend_threshold=0.35,
        use_daily_filter=True    # 启用日线过滤
    )
    
    # 3. 运行策略
    print("\n[3/4] 运行交易策略...")
    results, trade_log = system.run(
        merged_df,
        initial_capital=10000,   # 初始资金1万美元
        risk_per_trade=0.02      # 单笔风险2%
    )
    
    # 4. 分析结果
    print("\n[4/4] 分析结果...")
    analyzer = ResultAnalyzer(results, trade_log)
    analyzer.summary()
    
    # 保存结果
    analyzer.save_to_csv('eth_4h_trading_results.csv')
    
    # 绘制图表
    print("\n生成图表...")
    analyzer.plot_results(save_path='eth_trading_chart.png')
    
    print("\n" + "="*60)
    print("系统运行完成!")
    print("="*60)
    
    # 给出建议
    final_capital = results['capital'].iloc[-1]
    initial_capital = 10000
    total_return = (final_capital / initial_capital - 1) * 100
    
    if total_return > 20:
        print(" 系统表现优秀! 建议进入小资金实盘测试阶段")
    elif total_return > 0:
        print(" 系统表现良好! 建议继续优化参数")
    else:
        print(" 系统需要进一步优化! 建议:")
        print("   1. 调整止损倍数")
        print("   2. 尝试不同的时间框架组合")
        print("   3. 增加更多过滤条件")


if __name__ == "__main__":
    # 设置显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # 运行主程序
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()