"""
衍生品与多资产数据域（Derivatives & Multi-Asset）采集模块

数据类型包括：
1. ETF与基金：
   - ETF基本信息
   - ETF/LOF行情
   - 基金净值/持仓/规模
   - 基金复权因子

2. 期货与期权：
   - 期货合约信息
   - 期货行情（主力/连续）
   - 仓单/持仓排名
   - 期权行情/Greeks/波动率

3. 债券与可转债：
   - 国债收益率曲线
   - 可转债行情
   - 可转债溢价率
   - 回购利率

使用方法:
    from src.data_pipeline.collectors.structured.derivatives import (
        # ETF与基金
        get_fund_basic,
        get_fund_daily,
        get_fund_nav,
        get_fund_portfolio,
        get_fund_share,
        get_fund_adj,
        
        # 期货与期权
        get_fut_basic,
        get_fut_daily,
        get_fut_holding,
        get_fut_wsr,
        get_opt_basic,
        get_opt_daily,
        
        # 债券与可转债
        get_yield_curve,
        get_cb_basic,
        get_cb_daily,
        get_repo_daily,
        get_cb_premium,
    )
"""

# ETF与基金采集器
from .etf_fund import (
    FundBasicCollector,
    FundDailyCollector,
    FundNavCollector,
    FundPortfolioCollector,
    FundShareCollector,
    FundAdjCollector,
    get_fund_basic,
    get_fund_daily,
    get_fund_nav,
    get_fund_portfolio,
    get_fund_share,
    get_fund_adj,
)

# 期货与期权采集器
from .futures_options import (
    FuturesBasicCollector,
    FuturesDailyCollector,
    FuturesHoldingCollector,
    FuturesWarehouseCollector,
    OptionsBasicCollector,
    OptionsDailyCollector,
    get_fut_basic,
    get_fut_daily,
    get_fut_holding,
    get_fut_wsr,
    get_opt_basic,
    get_opt_daily,
)

# 债券与可转债采集器
from .bond_convertible import (
    YieldCurveCollector,
    CBBasicCollector,
    CBDailyCollector,
    RepoDailyCollector,
    CBPremiumCollector,
    get_yield_curve,
    get_cb_basic,
    get_cb_daily,
    get_repo_daily,
    get_cb_premium,
)

__all__ = [
    # ETF与基金采集器类
    'FundBasicCollector',
    'FundDailyCollector',
    'FundNavCollector',
    'FundPortfolioCollector',
    'FundShareCollector',
    'FundAdjCollector',
    # ETF与基金便捷函数
    'get_fund_basic',
    'get_fund_adj',
    'get_fund_daily',
    'get_fund_nav',
    'get_fund_portfolio',
    'get_fund_share',
    
    # 期货与期权采集器类
    'FuturesBasicCollector',
    'FuturesDailyCollector',
    'FuturesHoldingCollector',
    'FuturesWarehouseCollector',
    'OptionsBasicCollector',
    'OptionsDailyCollector',
    # 期货与期权便捷函数
    'get_fut_basic',
    'get_fut_daily',
    'get_fut_holding',
    'get_fut_wsr',
    'get_opt_basic',
    'get_opt_daily',
    
    # 债券与可转债采集器类
    'YieldCurveCollector',
    'CBBasicCollector',
    'CBDailyCollector',
    'RepoDailyCollector',
    'CBPremiumCollector',
    # 债券与可转债便捷函数
    'get_yield_curve',
    'get_cb_basic',
    'get_cb_daily',
    'get_repo_daily',
    'get_cb_premium',
]
