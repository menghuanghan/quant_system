"""
Cross-sectional Structure 数据域（板块/行业/主题数据）

包含采集器：
- 行业体系：申万/中信/同花顺行业分类、指数、成分
- 概念与主题板块：同花顺/东方财富/开盘啦概念板块及成分
- 板块行情与强弱：板块涨跌幅、热度排行、历史行情

使用示例：
    from src.data_pipeline.collectors.structured.cross_sectional import (
        # 行业数据
        get_sw_index_classify,
        get_sw_index_member,
        get_sw_daily,
        # 概念数据
        get_ths_index,
        get_ths_member,
        get_dc_index,
        get_dc_member,
        # 板块行情
        get_sector_performance,
        get_sector_rank,
        get_limit_up_pool,
    )
"""

# 行业体系采集器
from .industry import (
    SWIndexClassifyCollector,
    SWIndexMemberCollector,
    SWDailyCollector,
    get_sw_index_classify,
    get_sw_index_member,
    get_sw_daily,
)

# 概念与主题板块采集器
from .concept import (
    THSIndexCollector,
    THSMemberCollector,
    THSDailyCollector,
    DCIndexCollector,
    DCMemberCollector,
    KPLConceptCollector,
    KPLConceptConsCollector,
    get_ths_index,
    get_ths_member,
    get_ths_daily,
    get_dc_index,
    get_dc_member,
    get_kpl_concept,
    get_kpl_concept_cons,
)

# 板块行情与强弱采集器
from .performance import (
    SectorPerformanceCollector,
    IndustryBoardEMCollector,
    ConceptBoardEMCollector,
    SectorHistCollector,
    SectorRankCollector,
    LimitUpPoolCollector,
    get_sector_performance,
    get_industry_board_em,
    get_concept_board_em,
    get_sector_hist,
    get_sector_rank,
    get_limit_up_pool,
)

__all__ = [
    # 行业分类
    'SWIndexClassifyCollector',
    'SWIndexMemberCollector', 
    'SWDailyCollector',
    'get_sw_index_classify',
    'get_sw_index_member',
    'get_sw_daily',
    # 概念板块
    'THSIndexCollector',
    'THSMemberCollector',
    'THSDailyCollector',
    'DCIndexCollector',
    'DCMemberCollector',
    'KPLConceptCollector',
    'KPLConceptConsCollector',
    'get_ths_index',
    'get_ths_member',
    'get_ths_daily',
    'get_dc_index',
    'get_dc_member',
    'get_kpl_concept',
    'get_kpl_concept_cons',
    # 板块行情
    'SectorPerformanceCollector',
    'IndustryBoardEMCollector',
    'ConceptBoardEMCollector',
    'SectorHistCollector',
    'SectorRankCollector',
    'LimitUpPoolCollector',
    'get_sector_performance',
    'get_industry_board_em',
    'get_concept_board_em',
    'get_sector_hist',
    'get_sector_rank',
    'get_limit_up_pool',
]
