#!/usr/bin/env python3
"""
向量化回测引擎

采用 "外层日循环 + 内层向量化" 的混合架构:
- 外层按日循环处理交易逻辑（停牌锁仓、涨跌停限制）
- 内层使用 cuDF 向量化计算信号和收益

支持:
- 停牌股不可买入
- 涨停股不可买入（买不进）
- 跌停股不可卖出（卖不出）
- 停牌锁仓：持仓股停牌时保留持仓
- 交易成本计算
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回测配置"""
    top_k: int = 50                 # 持仓数量
    cost_rate: float = 0.003        # 双边交易成本（千三）
    capital: float = 10_000_000     # 初始资金
    use_gpu: bool = True            # 是否使用 GPU


@dataclass
class BacktestResult:
    """回测结果"""
    # 净值序列
    nav_series: pd.Series = None
    # 每日收益率
    daily_returns: pd.Series = None
    # 持仓记录
    positions: pd.DataFrame = None
    # 交易记录
    trades: pd.DataFrame = None
    # 统计指标
    metrics: Dict = field(default_factory=dict)
    
    def summary(self) -> str:
        """生成回测摘要"""
        m = self.metrics
        return f"""
================== 回测结果 ==================
📊 收益指标:
   年化收益率: {m.get('annual_return', 0):.2%}
   累计收益率: {m.get('total_return', 0):.2%}
   最大回撤: {m.get('max_drawdown', 0):.2%}

📊 风险指标:
   年化波动率: {m.get('annual_volatility', 0):.2%}
   夏普比率: {m.get('sharpe_ratio', 0):.4f}
   Calmar比率: {m.get('calmar_ratio', 0):.4f}

📊 交易统计:
   交易日数: {m.get('trading_days', 0)}
   日均换手率: {m.get('avg_turnover', 0):.2%}
   总交易成本: {m.get('total_cost', 0):,.0f}
   胜率: {m.get('win_rate', 0):.2%}
===============================================
"""


class VectorBacktester:
    """
    向量化回测器
    
    采用 "外层日循环 + 内层向量化" 的混合架构:
    - 纯矩阵运算虽然快，但处理"停牌强行持仓"逻辑极易出错
    - 日循环确保逻辑正确，内部向量化保证速度
    
    Usage:
        backtester = VectorBacktester(config)
        result = backtester.run(pred_df, market_df)
        print(result.summary())
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        初始化回测器
        
        Args:
            config: 回测配置
        """
        self.config = config or BacktestConfig()
        
        # 尝试导入 cuDF
        self.use_gpu = self.config.use_gpu
        if self.use_gpu:
            try:
                import cudf
                self.pd = cudf
                logger.info("🚀 VectorBacktester: GPU 加速已启用 (cuDF)")
            except ImportError:
                self.pd = pd
                self.use_gpu = False
                logger.warning("⚠️ cuDF 不可用，回退到 pandas")
        else:
            self.pd = pd
    
    def run(
        self,
        pred_df: pd.DataFrame,
        market_df: pd.DataFrame,
    ) -> BacktestResult:
        """
        执行回测
        
        Args:
            pred_df: 预测数据，必须包含 trade_date, ts_code, score
            market_df: 行情数据，必须包含 trade_date, ts_code, close, return_1d,
                       is_trading, is_limit_up, is_limit_down
        
        Returns:
            BacktestResult: 回测结果
        """
        logger.info("=" * 60)
        logger.info("🚀 开始向量化回测")
        logger.info("=" * 60)
        logger.info(f"  📋 配置:")
        logger.info(f"     持仓数量: {self.config.top_k}")
        logger.info(f"     交易成本: {self.config.cost_rate:.4f}")
        logger.info(f"     初始资金: {self.config.capital:,.0f}")
        
        # 1. 数据预处理
        pred_df, market_df = self._preprocess(pred_df, market_df)
        
        # 2. 合并数据
        merged_df = self._merge_data(pred_df, market_df)
        
        # 3. 获取交易日列表
        trade_dates = sorted(merged_df['trade_date'].unique())
        logger.info(f"  📅 回测区间: {trade_dates[0]} ~ {trade_dates[-1]}")
        logger.info(f"  📅 交易日数: {len(trade_dates)}")
        
        # 4. 按日循环回测
        result = self._run_daily_loop(merged_df, trade_dates)
        
        # 5. 计算统计指标
        self._calculate_metrics(result)
        
        logger.info("=" * 60)
        logger.info("✅ 回测完成")
        logger.info("=" * 60)
        
        return result
    
    def _preprocess(
        self,
        pred_df: pd.DataFrame,
        market_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """数据预处理"""
        logger.info("  📊 数据预处理...")
        
        # 转换为 GPU DataFrame (如果可用)
        if self.use_gpu and not isinstance(pred_df, self.pd.DataFrame):
            import cudf
            pred_df = cudf.DataFrame.from_pandas(pred_df)
            market_df = cudf.DataFrame.from_pandas(market_df)
        
        # 确保日期格式一致
        if pred_df['trade_date'].dtype == 'object':
            pred_df['trade_date'] = self.pd.to_datetime(pred_df['trade_date'])
        if market_df['trade_date'].dtype == 'object':
            market_df['trade_date'] = self.pd.to_datetime(market_df['trade_date'])
        
        # 检查必要列
        required_pred = ['trade_date', 'ts_code', 'score']
        required_market = ['trade_date', 'ts_code', 'close', 'return_1d', 
                          'is_trading', 'is_limit_up', 'is_limit_down']
        
        missing_pred = [c for c in required_pred if c not in pred_df.columns]
        if missing_pred:
            raise ValueError(f"pred_df 缺少必要列: {missing_pred}")
        
        missing_market = [c for c in required_market if c not in market_df.columns]
        if missing_market:
            raise ValueError(f"market_df 缺少必要列: {missing_market}")
        
        logger.info(f"    pred_df: {len(pred_df):,} 行")
        logger.info(f"    market_df: {len(market_df):,} 行")
        
        return pred_df, market_df
    
    def _merge_data(
        self,
        pred_df: pd.DataFrame,
        market_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        合并预测数据和行情数据
        
        关键：将 return_1d 向后移一天，得到 next_return
        - T日选股，使用 T+1 日的收益
        - 这样避免前视偏差
        """
        logger.info("  📊 合并数据...")
        
        # 转回 pandas 进行处理
        if self.use_gpu:
            if hasattr(pred_df, 'to_pandas'):
                pred_df = pred_df.to_pandas()
            if hasattr(market_df, 'to_pandas'):
                market_df = market_df.to_pandas()
        
        # 创建 next_return: T日的 next_return = T+1日的 return_1d
        # 按股票分组，将 return_1d 向前移动（今天的 next_return 是明天的 return_1d）
        logger.info("    计算 T+1 收益率...")
        market_df = market_df.sort_values(['ts_code', 'trade_date'])
        market_df['next_return'] = market_df.groupby('ts_code')['return_1d'].shift(-1)
        
        # Inner join
        merged = pred_df.merge(
            market_df,
            on=['trade_date', 'ts_code'],
            how='inner'
        )
        
        # 删除缺少 next_return 的行（最后一天）
        merged = merged.dropna(subset=['next_return'])
        
        logger.info(f"    合并后: {len(merged):,} 行")
        
        return merged
    
    def _run_daily_loop(
        self,
        merged_df: pd.DataFrame,
        trade_dates: List
    ) -> BacktestResult:
        """
        按日循环执行回测
        
        核心逻辑:
        1. 每天先处理停牌锁仓：停牌股不能卖出，保留持仓
        2. 选股时排除：停牌股、涨停股
        3. 卖出时检查：跌停股不能卖出
        """
        logger.info("  📊 按日循环回测...")
        
        # 初始化
        capital = self.config.capital
        nav = capital  # 净值
        positions = {}  # {ts_code: shares}
        
        # 记录
        nav_list = []
        daily_returns = []
        position_records = []
        trade_records = []
        
        # 预处理：按日期分组
        date_groups = merged_df.groupby('trade_date')
        
        prev_portfolio_value = capital
        
        for i, date in enumerate(trade_dates):
            if date not in date_groups.groups:
                continue
            
            # 当日数据（已在内存中向量化）
            daily_data = date_groups.get_group(date).copy()
            daily_data = daily_data.set_index('ts_code')
            
            # ========== 1. 计算当前持仓市值 ==========
            portfolio_value = 0
            for code, shares in list(positions.items()):
                if code in daily_data.index:
                    price = daily_data.loc[code, 'close']
                    portfolio_value += price * shares
                else:
                    # 股票退市？按之前价格计算
                    portfolio_value += shares  # 简化处理
            
            cash = nav - portfolio_value if portfolio_value < nav else 0
            
            # ========== 2. 处理停牌锁仓 ==========
            locked_positions = {}  # 停牌锁仓的持仓
            available_positions = {}  # 可交易的持仓
            
            for code, shares in positions.items():
                if code in daily_data.index:
                    is_trading = daily_data.loc[code, 'is_trading']
                    is_limit_down = daily_data.loc[code, 'is_limit_down']
                    
                    if is_trading == 0:
                        # 停牌，锁仓
                        locked_positions[code] = shares
                    elif is_limit_down == 1:
                        # 跌停，不能卖出
                        locked_positions[code] = shares
                    else:
                        available_positions[code] = shares
                else:
                    # 不在今日数据中，可能退市
                    pass
            
            # ========== 3. 选股 ==========
            # 排除停牌和涨停股
            tradable = daily_data[
                (daily_data['is_trading'] == 1) &
                (daily_data['is_limit_up'] == 0)
            ]
            
            # 按 score 降序排列
            tradable = tradable.sort_values('score', ascending=False)
            
            # 考虑锁仓占用的名额
            available_slots = self.config.top_k - len(locked_positions)
            
            # 选出 top_k (减去锁仓的)
            target_codes = set(tradable.head(max(0, available_slots)).index.tolist())
            
            # 加上锁仓的
            final_target = target_codes | set(locked_positions.keys())
            
            # ========== 4. 计算调仓 ==========
            # 目标持仓（等权重）
            n_holding = len(final_target)
            if n_holding > 0:
                target_weight = 1.0 / n_holding
            else:
                target_weight = 0
            
            # 计算目标市值
            total_value = nav  # 使用当前净值
            target_value_per_stock = total_value * target_weight
            
            # ========== 5. 执行调仓 ==========
            new_positions = {}
            turnover = 0
            
            for code in final_target:
                if code not in daily_data.index:
                    continue
                
                price = daily_data.loc[code, 'close']
                if price <= 0:
                    continue
                
                # 目标股数
                target_shares = target_value_per_stock / price
                
                # 计算换手
                old_shares = positions.get(code, 0)
                old_value = old_shares * price
                new_value = target_shares * price
                
                turnover += abs(new_value - old_value)
                new_positions[code] = target_shares
            
            # ========== 6. 计算收益 ==========
            # 使用 T+1 日收益 (next_return)
            daily_return = 0
            for code, shares in new_positions.items():
                if code in daily_data.index:
                    ret = daily_data.loc[code, 'next_return']
                    if pd.notna(ret):
                        weight = shares * daily_data.loc[code, 'close'] / total_value if total_value > 0 else 0
                        daily_return += weight * ret
            
            # 扣除交易成本
            cost = turnover * self.config.cost_rate
            daily_return -= cost / total_value if total_value > 0 else 0
            
            # 更新净值
            nav = nav * (1 + daily_return)
            
            # ========== 7. 记录 ==========
            nav_list.append({'date': date, 'nav': nav})
            daily_returns.append({'date': date, 'return': daily_return})
            
            # 更新持仓
            positions = new_positions
            
            # 进度显示
            if (i + 1) % 50 == 0 or i == len(trade_dates) - 1:
                logger.info(f"    进度: {i+1}/{len(trade_dates)} | NAV: {nav:,.0f} | 收益: {(nav/capital-1)*100:.2f}%")
        
        # 构建结果
        nav_df = pd.DataFrame(nav_list)
        ret_df = pd.DataFrame(daily_returns)
        
        result = BacktestResult(
            nav_series=nav_df.set_index('date')['nav'],
            daily_returns=ret_df.set_index('date')['return'],
        )
        
        return result
    
    def _calculate_metrics(self, result: BacktestResult):
        """计算统计指标"""
        logger.info("  📊 计算统计指标...")
        
        nav = result.nav_series
        returns = result.daily_returns
        
        # 基本统计
        total_return = nav.iloc[-1] / nav.iloc[0] - 1
        trading_days = len(returns)
        annual_factor = 252 / trading_days
        
        # 年化收益
        annual_return = (1 + total_return) ** annual_factor - 1
        
        # 年化波动率
        annual_volatility = returns.std() * np.sqrt(252)
        
        # 夏普比率 (假设无风险利率 2%)
        rf = 0.02
        excess_return = annual_return - rf
        sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0
        
        # 最大回撤
        cummax = nav.cummax()
        drawdown = (nav - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Calmar 比率
        calmar_ratio = -annual_return / max_drawdown if max_drawdown < 0 else 0
        
        # 胜率
        win_rate = (returns > 0).sum() / len(returns)
        
        # 平均换手率
        avg_turnover = 0  # TODO: 从交易记录计算
        
        # 总交易成本
        total_cost = 0  # TODO: 从交易记录计算
        
        result.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'trading_days': trading_days,
            'avg_turnover': avg_turnover,
            'total_cost': total_cost,
        }
        
        logger.info(f"    年化收益: {annual_return:.2%}")
        logger.info(f"    夏普比率: {sharpe_ratio:.4f}")
        logger.info(f"    最大回撤: {max_drawdown:.2%}")


def load_market_data(use_gpu: bool = True) -> pd.DataFrame:
    """
    从 DWD 加载行情数据
    
    Returns:
        包含 trade_date, ts_code, close, return_1d, is_trading, 
        is_limit_up, is_limit_down 的 DataFrame
    """
    from pathlib import Path
    
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    dwd_path = project_root / "data" / "processed" / "structured" / "dwd"
    
    if use_gpu:
        try:
            import cudf
            _pd = cudf
        except ImportError:
            _pd = pd
    else:
        _pd = pd
    
    logger.info("📊 加载 DWD 行情数据...")
    
    # 价格数据
    price_df = _pd.read_parquet(dwd_path / "dwd_stock_price.parquet")
    price_cols = ['trade_date', 'ts_code', 'close', 'return_1d', 'is_trading']
    price_df = price_df[price_cols]
    
    # 状态数据
    status_df = _pd.read_parquet(dwd_path / "dwd_stock_status.parquet")
    status_cols = ['trade_date', 'ts_code', 'is_limit_up', 'is_limit_down']
    status_df = status_df[status_cols]
    
    # 合并
    market_df = price_df.merge(status_df, on=['trade_date', 'ts_code'], how='left')
    
    # 填充缺失值
    market_df['is_limit_up'] = market_df['is_limit_up'].fillna(0)
    market_df['is_limit_down'] = market_df['is_limit_down'].fillna(0)
    
    logger.info(f"  ✓ 加载完成: {len(market_df):,} 行")
    
    if use_gpu and hasattr(market_df, 'to_pandas'):
        return market_df.to_pandas()
    return market_df
