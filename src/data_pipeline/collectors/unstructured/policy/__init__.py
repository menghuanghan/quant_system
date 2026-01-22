"""
政策与监管文本采集模块

数据来源（权威源）：
1. 国务院 (gov.cn)
2. 发改委 (ndrc.gov.cn)
3. 证监会 (csrc.gov.cn)

采集策略：
- 权威源定向采集：保证时效性和原文准确性

数据特点：
- 低频：更新频率低于行情数据
- 高权：对市场影响极大
- 非标：格式不标准（网页文本、PDF、Word等）
"""

from .base_policy import (
    BasePolicyCollector,
    PolicyDocument,
    PolicySource,
    PolicyCategory,
    get_policy_collector
)

from .csrc import (
    CSRCCollector,
    get_csrc_policy
)

from .gov_council import (
    GovCouncilCollector,
    NDRCCollector,
    get_gov_policy,
    get_ndrc_policy
)

__all__ = [
    # 基类
    'BasePolicyCollector',
    'PolicyDocument',
    'PolicySource',
    'PolicyCategory',
    'get_policy_collector',
    # 证监会
    'CSRCCollector',
    'get_csrc_policy',
    # 国务院/发改委
    'GovCouncilCollector',
    'NDRCCollector',
    'get_gov_policy',
    'get_ndrc_policy',
]
