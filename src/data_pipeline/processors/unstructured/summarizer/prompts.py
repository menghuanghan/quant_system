"""
Prompt模板模块

为不同类型的金融文本定义专业的摘要生成Prompt。
每个Prompt都针对量化交易需求设计，重点提取：
- 对股价有影响的关键信息
- 利好/利空判断依据
- 行业影响范围
- 政策受益方向
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass
from string import Template

from .base import DataType

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Prompt模板数据结构"""
    system: str              # 系统提示
    user_template: str       # 用户提示模板（支持变量替换）
    data_type: DataType
    description: str = ""
    
    def format_user(self, **kwargs) -> str:
        """格式化用户提示"""
        return Template(self.user_template).safe_substitute(**kwargs)


class PromptTemplates:
    """
    Prompt模板管理器
    
    为量化交易系统设计的专业金融文本摘要Prompt，
    重点关注对股价有影响的信息，而非简单的文本压缩。
    """
    
    # ==================== 系统提示 ====================
    
    # 金融分析师系统提示（通用）
    SYSTEM_FINANCIAL_ANALYST = """你是一个专业的金融量化分析师，具有丰富的A股市场研究经验。
你的任务是分析金融文本，提取对股价、行业走势有重要影响的核心信息。
你需要：
1. 识别利好/利空因素
2. 关注具体的数字、指标、时间节点
3. 忽略套话、免责声明等无关内容
4. 输出简洁、专业、可用于量化分析的摘要"""

    # 政策分析师系统提示
    SYSTEM_POLICY_ANALYST = """你是一个专注于宏观经济和政策研究的分析师。
你的任务是分析政策文件，提取：
1. 核心政策措施和目标
2. 受益行业和领域
3. 政策实施时间和力度
4. 对资本市场可能的影响方向"""

    # 新闻分析师系统提示
    SYSTEM_NEWS_ANALYST = """你是一个财经新闻分析师，专注于从新闻中提取市场相关信息。
你需要：
1. 识别新闻的核心事件
2. 判断事件对市场的潜在影响
3. 关注涉及的行业、公司、指标
4. 排除情绪化表述，提炼客观事实"""

    # ==================== 公告类Prompt ====================
    
    ANNOUNCEMENT_GENERAL = PromptTemplate(
        data_type=DataType.ANNOUNCEMENT,
        system=SYSTEM_FINANCIAL_ANALYST,
        user_template="""请阅读以下上市公司公告，提取对股价有重要影响的核心信息。

要求：
1. 忽略格式化内容、免责声明、套话
2. 重点关注：业绩数据、重大事项、风险提示
3. 字数控制在50-100字
4. 语言简洁专业

公告原文：
$content

请输出摘要：""",
        description="通用公告摘要"
    )
    
    ANNOUNCEMENT_FINANCIAL = PromptTemplate(
        data_type=DataType.ANNOUNCEMENT_FINANCIAL,
        system=SYSTEM_FINANCIAL_ANALYST,
        user_template="""请阅读以下财务公告（可能是业绩预告、年报、季报等），提取关键财务信息。

重点关注：
1. 营收、净利润及同比变化
2. 业绩变动原因
3. 未来展望或指引
4. 风险提示

财务公告原文：
$content

请用50-100字总结核心财务信息和变动原因：""",
        description="财务公告摘要"
    )
    
    ANNOUNCEMENT_MAJOR = PromptTemplate(
        data_type=DataType.ANNOUNCEMENT_MAJOR,
        system=SYSTEM_FINANCIAL_ANALYST,
        user_template="""请阅读以下重大事项公告（可能涉及并购重组、股权变动、重大合同等），提取关键信息。

重点关注：
1. 交易/事项的具体内容
2. 涉及金额、股权比例等具体数字
3. 交易对手方
4. 对公司业务和财务的影响
5. 审批进度和时间节点

公告原文：
$content

请用50-100字总结这一重大事项的核心要点：""",
        description="重大事项公告摘要"
    )
    
    ANNOUNCEMENT_MANAGEMENT = PromptTemplate(
        data_type=DataType.ANNOUNCEMENT_MANAGEMENT,
        system=SYSTEM_FINANCIAL_ANALYST,
        user_template="""请阅读以下管理层变动公告，提取关键人事变动信息。

重点关注：
1. 变动职位和人员
2. 变动原因
3. 新任人员背景
4. 对公司治理的潜在影响

公告原文：
$content

请用50-100字总结人事变动要点：""",
        description="管理层变动公告摘要"
    )
    
    # ==================== 研报类Prompt ====================
    
    REPORT_GENERAL = PromptTemplate(
        data_type=DataType.REPORT,
        system=SYSTEM_FINANCIAL_ANALYST,
        user_template="""请阅读以下研究报告，提取核心投资观点。

重点关注：
1. 核心结论和评级
2. 关键盈利预测和估值
3. 主要逻辑和催化剂
4. 风险提示

研报原文：
$content

请用50-100字总结研报核心观点：""",
        description="通用研报摘要"
    )
    
    REPORT_COMPANY = PromptTemplate(
        data_type=DataType.REPORT_COMPANY,
        system=SYSTEM_FINANCIAL_ANALYST,
        user_template="""请阅读以下个股研究报告，提取投资建议和核心逻辑。

重点关注：
1. 投资评级（买入/增持/持有/减持）
2. 目标价和当前估值
3. 核心投资逻辑
4. 业绩预测（EPS、PE等）
5. 关键假设和风险

个股研报原文：
$content

请用50-100字总结投资建议和核心逻辑：""",
        description="个股研报摘要"
    )
    
    REPORT_INDUSTRY = PromptTemplate(
        data_type=DataType.REPORT_INDUSTRY,
        system=SYSTEM_FINANCIAL_ANALYST,
        user_template="""请阅读以下行业研究报告，提取行业观点和投资策略。

重点关注：
1. 行业景气度判断
2. 核心驱动因素
3. 重点推荐标的
4. 行业风险点

行业研报原文：
$content

请用50-100字总结行业观点和投资方向：""",
        description="行业研报摘要"
    )
    
    REPORT_STRATEGY = PromptTemplate(
        data_type=DataType.REPORT_STRATEGY,
        system=SYSTEM_FINANCIAL_ANALYST,
        user_template="""请阅读以下策略研究报告，提取市场观点和配置建议。

重点关注：
1. 对大盘走势的判断
2. 推荐的行业配置方向
3. 风格切换建议
4. 关键风险因素

策略研报原文：
$content

请用50-100字总结市场观点和配置建议：""",
        description="策略研报摘要"
    )
    
    # ==================== 政策类Prompt ====================
    
    POLICY_GENERAL = PromptTemplate(
        data_type=DataType.POLICY,
        system=SYSTEM_POLICY_ANALYST,
        user_template="""请阅读以下政策文件，提取核心措施和市场影响。

要求：
1. 概括政策核心内容
2. 明确受益行业/领域
3. 关注政策力度和时间安排
4. 分析对资本市场的潜在影响

政策原文：
$content

请用50-100字简述政策核心措施及受益行业：""",
        description="通用政策摘要"
    )
    
    POLICY_FISCAL = PromptTemplate(
        data_type=DataType.POLICY_FISCAL,
        system=SYSTEM_POLICY_ANALYST,
        user_template="""请阅读以下财政政策文件，提取核心要点。

重点关注：
1. 财政支出/税收政策变化
2. 补贴、减税降费措施
3. 受益行业和企业类型
4. 政策规模和实施节奏

财政政策原文：
$content

请用50-100字总结财政政策要点和受益方向：""",
        description="财政政策摘要"
    )
    
    POLICY_MONETARY = PromptTemplate(
        data_type=DataType.POLICY_MONETARY,
        system=SYSTEM_POLICY_ANALYST,
        user_template="""请阅读以下货币政策相关文件，提取核心要点。

重点关注：
1. 利率、准备金率等政策工具变化
2. 流动性投放/回收规模
3. 信贷政策导向
4. 对金融市场的影响

货币政策原文：
$content

请用50-100字总结货币政策取向和市场影响：""",
        description="货币政策摘要"
    )
    
    POLICY_INDUSTRY = PromptTemplate(
        data_type=DataType.POLICY_INDUSTRY,
        system=SYSTEM_POLICY_ANALYST,
        user_template="""请阅读以下产业政策文件，提取核心措施和受益方向。

重点关注：
1. 政策支持的具体行业/领域
2. 具体支持措施（补贴、税收、准入等）
3. 政策目标和时间表
4. 利好的细分赛道

产业政策原文：
$content

请用50-100字总结产业政策支持方向和受益领域：""",
        description="产业政策摘要"
    )
    
    # ==================== 新闻类Prompt ====================
    
    NEWS_GENERAL = PromptTemplate(
        data_type=DataType.NEWS,
        system=SYSTEM_NEWS_ANALYST,
        user_template="""请阅读以下新闻，提取核心事件和市场影响。

要求：
1. 概括新闻核心事件
2. 涉及的行业、公司
3. 对市场的潜在影响
4. 排除情绪化描述

新闻原文：
$content

请用50-100字总结新闻要点：""",
        description="通用新闻摘要"
    )
    
    NEWS_MARKET = PromptTemplate(
        data_type=DataType.NEWS_MARKET,
        system=SYSTEM_NEWS_ANALYST,
        user_template="""请阅读以下市场新闻，提取对市场走势有影响的关键信息。

重点关注：
1. 涉及的市场、指数、板块
2. 具体数据和指标
3. 原因分析
4. 后续影响判断

市场新闻原文：
$content

请用50-100字总结市场动态要点：""",
        description="市场新闻摘要"
    )
    
    NEWS_COMPANY = PromptTemplate(
        data_type=DataType.NEWS_COMPANY,
        system=SYSTEM_NEWS_ANALYST,
        user_template="""请阅读以下公司相关新闻，提取关键事件和影响。

重点关注：
1. 涉及的公司和事件
2. 事件的具体内容
3. 对公司业务/股价的潜在影响
4. 利好还是利空

公司新闻原文：
$content

请用50-100字总结公司事件要点：""",
        description="公司新闻摘要"
    )
    
    # ==================== 事件类Prompt ====================
    
    EVENT_GENERAL = PromptTemplate(
        data_type=DataType.EVENT,
        system=SYSTEM_FINANCIAL_ANALYST,
        user_template="""请阅读以下事件公告，提取关键信息。

重点关注：
1. 事件类型和具体内容
2. 涉及的公司和股票
3. 关键时间节点
4. 对投资者的影响

事件公告原文：
$content

请用50-100字总结事件要点：""",
        description="事件公告摘要"
    )
    
    # ==================== CCTV新闻专用Prompt ====================
    
    # CCTV新闻分析系统提示
    SYSTEM_CCTV_ANALYST = """你是一个专注于分析新闻联播内容的金融市场分析师。
你的任务是从官方新闻中提取对股市大盘有影响的政策信号和舆论定调。
你需要：
1. 识别关键政策信号词（如"把...放在首位"、"坚决遏制"、"稳字当头"等）
2. 判断新闻的政治定调是正面、负面还是中性
3. 分析涉及的领域（如房地产、科技、金融等）
4. 给出对大盘多空的判断依据"""

    CCTV_ANALYSIS = PromptTemplate(
        data_type=DataType.NEWS,
        system=SYSTEM_CCTV_ANALYST,
        user_template="""分析这条新闻联播内容，提取对股市大盘的影响信号。

新闻内容：
$content

请按以下格式输出（JSON格式）：
{
    "keywords": ["关键词1", "关键词2", ...],
    "tone": "正面/负面/中性",
    "tone_reason": "定调判断理由",
    "domains": ["涉及领域1", "涉及领域2"],
    "market_signal": "利好大盘/利空大盘/中性",
    "signal_reason": "市场信号判断理由"
}

重要判断逻辑：
- "稳增长"、"流动性合理充裕"、"降准降息"、"支持民营经济" -> 利好大盘
- "去杠杆"、"遏制"、"泡沫"、"防范风险"、"整顿"、"规范" -> 利空大盘
- 领导人外事访问、外交新闻、文化体育、社会民生 -> 中性

请输出JSON：""",
        description="CCTV新闻关键词提取与语气分析"
    )
    
    # ==================== 政策分析专用Prompt ====================
    
    # 政策行业映射系统提示
    SYSTEM_POLICY_SECTOR_ANALYST = """你是一个宏观政策分析师，专注于分析政策对A股各行业的影响。
你需要精确识别政策的受益行业和受损行业，使用申万行业分类标准。

申万一级行业列表：
农林牧渔、基础化工、钢铁、有色金属、电子、汽车、家用电器、食品饮料、
纺织服饰、轻工制造、医药生物、公用事业、交通运输、房地产、商贸零售、
社会服务、银行、非银金融、综合、建筑材料、建筑装饰、电力设备、
机械设备、国防军工、计算机、传媒、通信、煤炭、石油石化、环保、美容护理

你的输出必须使用上述标准行业名称。"""

    POLICY_SECTOR_MAPPING = PromptTemplate(
        data_type=DataType.POLICY,
        system=SYSTEM_POLICY_SECTOR_ANALYST,
        user_template="""阅读这篇政策文件，分析对各行业的影响。

政策标题：$title

政策内容：
$content

请按以下格式输出（JSON格式）：
{
    "summary": "200字以内的政策摘要",
    "benefited_industries": [
        {"industry": "行业名称", "reason": "受益原因", "impact_level": "高/中/低"},
        ...
    ],
    "harmed_industries": [
        {"industry": "行业名称", "reason": "受损原因", "impact_level": "高/中/低"},
        ...
    ],
    "policy_direction": "扩张性/收缩性/中性",
    "implementation_timeline": "立即/短期/中长期"
}

打分逻辑参考：
- "加大投入"、"补贴"、"支持"、"鼓励"、"重点发展" -> 受益行业
- "限制"、"规范"、"整顿"、"收紧"、"严控"、"淘汰" -> 受损行业

行业名称必须从申万一级行业中选择，请输出JSON：""",
        description="政策行业映射分析"
    )
    
    # ==================== 通用Prompt ====================
    
    GENERIC = PromptTemplate(
        data_type=DataType.GENERIC,
        system=SYSTEM_FINANCIAL_ANALYST,
        user_template="""请阅读以下文本，提取核心信息并生成摘要。

要求：
1. 提取关键信息点
2. 语言简洁专业
3. 字数控制在50-100字
4. 关注对股市可能有影响的内容

原文：
$content

请输出摘要：""",
        description="通用文本摘要"
    )
    
    # ==================== Prompt索引 ====================
    
    _TEMPLATES: Dict[DataType, PromptTemplate] = {
        DataType.ANNOUNCEMENT: ANNOUNCEMENT_GENERAL,
        DataType.ANNOUNCEMENT_FINANCIAL: ANNOUNCEMENT_FINANCIAL,
        DataType.ANNOUNCEMENT_MAJOR: ANNOUNCEMENT_MAJOR,
        DataType.ANNOUNCEMENT_MANAGEMENT: ANNOUNCEMENT_MANAGEMENT,
        DataType.REPORT: REPORT_GENERAL,
        DataType.REPORT_COMPANY: REPORT_COMPANY,
        DataType.REPORT_INDUSTRY: REPORT_INDUSTRY,
        DataType.REPORT_STRATEGY: REPORT_STRATEGY,
        DataType.POLICY: POLICY_GENERAL,
        DataType.POLICY_FISCAL: POLICY_FISCAL,
        DataType.POLICY_MONETARY: POLICY_MONETARY,
        DataType.POLICY_INDUSTRY: POLICY_INDUSTRY,
        DataType.NEWS: NEWS_GENERAL,
        DataType.NEWS_MARKET: NEWS_MARKET,
        DataType.NEWS_COMPANY: NEWS_COMPANY,
        DataType.EVENT: EVENT_GENERAL,
        DataType.GENERIC: GENERIC,
    }
    
    @classmethod
    def get_template(cls, data_type: DataType) -> PromptTemplate:
        """
        获取指定数据类型的Prompt模板
        
        Args:
            data_type: 数据类型
            
        Returns:
            PromptTemplate: 对应的Prompt模板
        """
        return cls._TEMPLATES.get(data_type, cls.GENERIC)
    
    @classmethod
    def get_all_templates(cls) -> Dict[DataType, PromptTemplate]:
        """获取所有Prompt模板"""
        return cls._TEMPLATES.copy()
    
    @classmethod
    def build_messages(
        cls,
        content: str,
        data_type: DataType,
        custom_system: Optional[str] = None,
        **kwargs
    ) -> list:
        """
        构建完整的消息列表
        
        Args:
            content: 待摘要的文本内容
            data_type: 数据类型
            custom_system: 自定义系统提示（覆盖默认）
            **kwargs: 额外的模板变量
            
        Returns:
            list: 消息列表 [{"role": "system", ...}, {"role": "user", ...}]
        """
        template = cls.get_template(data_type)
        
        system = custom_system or template.system
        user = template.format_user(content=content, **kwargs)
        
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


# 简化的Prompt别名（方便直接引用）
ANNOUNCEMENT_PROMPT = PromptTemplates.ANNOUNCEMENT_GENERAL
REPORT_PROMPT = PromptTemplates.REPORT_GENERAL
POLICY_PROMPT = PromptTemplates.POLICY_GENERAL
NEWS_PROMPT = PromptTemplates.NEWS_GENERAL
