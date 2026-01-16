"""
预期与预测分析（Expectations & Forecasts）数据域

数据类型包括：
1. 盈利预测（earnings_forecast.py）
   - 业绩预告（公司官方）
   - 券商盈利预测
   - 一致预期数据

2. 机构评级（institutional_rating.py）
   - 机构投资评级
   - 评级汇总统计
   - 机构调研记录

3. 研究员指数（analyst_index.py）
   - 分析师排行
   - 分析师详情
   - 券商金股组合
   - 预测修正数据
"""

from .earnings_forecast import (
    EarningsForecastCollector,
    BrokerForecastCollector,
    ConsensusForecastCollector,
)

from .institutional_rating import (
    InstitutionalRatingCollector,
    RatingSummaryCollector,
    InstitutionalSurveyCollector,
)

from .analyst_index import (
    AnalystRankCollector,
    AnalystDetailCollector,
    BrokerGoldStockCollector,
    ForecastRevisionCollector,
)


# 便捷函数
def get_earnings_forecast(**kwargs):
    """获取业绩预告数据"""
    return EarningsForecastCollector().collect(**kwargs)


def get_broker_forecast(**kwargs):
    """获取券商盈利预测数据"""
    return BrokerForecastCollector().collect(**kwargs)


def get_consensus_forecast(**kwargs):
    """获取一致预期数据"""
    return ConsensusForecastCollector().collect(**kwargs)


def get_inst_rating(**kwargs):
    """获取机构评级数据"""
    return InstitutionalRatingCollector().collect(**kwargs)


def get_rating_summary(**kwargs):
    """获取评级汇总统计"""
    return RatingSummaryCollector().collect(**kwargs)


def get_inst_survey(**kwargs):
    """获取机构调研数据"""
    return InstitutionalSurveyCollector().collect(**kwargs)


def get_analyst_rank(**kwargs):
    """获取分析师排行数据"""
    return AnalystRankCollector().collect(**kwargs)


def get_analyst_detail(**kwargs):
    """获取分析师详情数据"""
    return AnalystDetailCollector().collect(**kwargs)


def get_broker_gold_stock(**kwargs):
    """获取券商金股数据"""
    return BrokerGoldStockCollector().collect(**kwargs)


def get_forecast_revision(**kwargs):
    """获取预测修正数据"""
    return ForecastRevisionCollector().collect(**kwargs)


__all__ = [
    # 采集器类
    'EarningsForecastCollector',
    'BrokerForecastCollector',
    'ConsensusForecastCollector',
    'InstitutionalRatingCollector',
    'RatingSummaryCollector',
    'InstitutionalSurveyCollector',
    'AnalystRankCollector',
    'AnalystDetailCollector',
    'BrokerGoldStockCollector',
    'ForecastRevisionCollector',
    
    # 便捷函数
    'get_earnings_forecast',
    'get_broker_forecast',
    'get_consensus_forecast',
    'get_inst_rating',
    'get_rating_summary',
    'get_inst_survey',
    'get_analyst_rank',
    'get_analyst_detail',
    'get_broker_gold_stock',
    'get_forecast_revision',
]
