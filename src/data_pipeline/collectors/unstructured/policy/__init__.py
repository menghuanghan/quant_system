"""
政策与监管文本采集模块

数据来源（权威源）：
1. 国务院 (gov.cn)
2. 发改委 (ndrc.gov.cn)

注意：证监会(CSRC)采集器已移除，其网站使用SPA架构无法获取历史数据
"""

from .base_policy import (
    BasePolicyCollector,
    PolicyDocument,
    PolicySource,
    PolicyCategory,
    get_policy_collector
)

from .gov_council import (
    GovCouncilCollector,
    get_gov_policies
)

from .ndrc import (
    NDRCCollector,
    get_ndrc_policies
)

__all__ = [
    # 基类
    'BasePolicyCollector',
    'PolicyDocument',
    'PolicySource',
    'PolicyCategory',
    'get_policy_collector',
    # 国务院
    'GovCouncilCollector',
    'get_gov_policies',
    # 发改委
    'NDRCCollector',
    'get_ndrc_policies',
]
