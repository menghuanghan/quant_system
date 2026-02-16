"""
处理流水线

实现多种处理流水线：
1. PDFPipeline: 用于 announcements/reports/events
   流程: URL -> content_extractor -> summarizer -> scorer -> score
   
2. ExchangePipeline: 用于 exchange
   流程: title -> rule_scorer -> score
   
3. CCTVPipeline: 用于 news/cctv
   流程: content -> keyword_extractor -> sentiment_scorer -> market_sentiment + beta_signal
   
4. PolicyPipeline: 用于 policy/gov, policy/ndrc
   流程: URL -> html_extractor -> sector_mapper -> industry_scorer -> industry_scores
"""

import logging
import time
import json
import re
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

import pandas as pd

from .base import (
    ProcessingConfig,
    ProcessingResult,
    DataCategory,
    ProcessingStatus,
    FIELD_MAPPING,
)

# 导入处理工具
from ..content_extractor import ContentExtractor, ExtractorResult
from ..summarizer import Summarizer, SummarizerConfig, DataType
from ..scorer import Scorer, ScorerConfig, ScoringMethod, RuleScorer

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """处理流水线基类"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process_single(
        self,
        record: Dict[str, Any],
        category: DataCategory
    ) -> ProcessingResult:
        """处理单条记录"""
        pass
    
    @abstractmethod
    def process_batch(
        self,
        records: List[Dict[str, Any]],
        category: DataCategory
    ) -> List[ProcessingResult]:
        """批量处理"""
        pass


class PDFPipeline(BasePipeline):
    """
    PDF处理流水线
    
    用于 announcements/reports/events
    流程: URL -> content_extractor -> summarizer -> scorer -> score
    """
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        
        # 初始化内容提取器
        self.content_extractor = ContentExtractor()
        
        # 初始化摘要生成器（使用LLM）
        summarizer_config = SummarizerConfig(
            model_name=config.model_name,
            ollama_host=config.ollama_host,
            timeout=config.llm_timeout,
            max_retries=config.llm_max_retries,
        )
        self.summarizer = Summarizer(config=summarizer_config)
        
        # 初始化打分器（使用LLM）
        scorer_config = ScorerConfig(
            default_method=ScoringMethod.LLM,
            model_name=config.model_name,
            ollama_host=config.ollama_host,
            timeout=config.llm_timeout,
            max_retries=config.llm_max_retries,
        )
        self.scorer = Scorer(config=scorer_config)
        
        self.logger.info("PDF流水线初始化完成")
    
    def _get_data_type(self, category: DataCategory) -> DataType:
        """获取对应的数据类型"""
        type_mapping = {
            DataCategory.ANNOUNCEMENTS: DataType.ANNOUNCEMENT,
            DataCategory.REPORTS: DataType.REPORT,
            DataCategory.EVENTS: DataType.EVENT,
        }
        return type_mapping.get(category, DataType.ANNOUNCEMENT)
    
    def process_single(
        self,
        record: Dict[str, Any],
        category: DataCategory
    ) -> ProcessingResult:
        """
        处理单条PDF记录
        
        流程: URL -> content_extractor -> summarizer -> scorer
        """
        start_time = time.time()
        
        # 获取字段映射
        field_map = FIELD_MAPPING.get(category, {})
        
        # 提取基础信息
        record_id = str(record.get(field_map.get('id', 'id'), ''))
        ts_code = str(record.get(field_map.get('ts_code', 'ts_code'), ''))
        date = str(record.get(field_map.get('date', 'date'), ''))
        url = record.get(field_map.get('url', 'url'), '')
        title = record.get(field_map.get('title', 'title'), '')
        
        result = ProcessingResult(
            success=False,
            record_id=record_id,
            ts_code=ts_code,
            date=date,
        )
        
        try:
            # 1. 提取内容
            if not url:
                result.error_message = "URL为空"
                result.elapsed_time = time.time() - start_time
                return result
            
            extract_result = self.content_extractor.extract(url)
            
            if not extract_result.success or not extract_result.content_text:
                # 如果提取失败，尝试使用标题
                self.logger.warning(f"内容提取失败: {extract_result.error_message}, 尝试使用标题")
                if title:
                    content = title
                    result.content_extracted = True
                    result.content_length = len(title)
                else:
                    result.error_message = f"内容提取失败: {extract_result.error_message}"
                    result.elapsed_time = time.time() - start_time
                    return result
            else:
                content = extract_result.content_text
                result.content_extracted = True
                result.content_length = len(content)
            
            # 2. 生成摘要
            data_type = self._get_data_type(category)
            summary_result = self.summarizer.summarize(
                content_text=content,
                data_type=data_type,
                title=title
            )
            
            if not summary_result.success:
                # 摘要失败时，使用标题或截断内容
                self.logger.warning(f"摘要生成失败: {summary_result.error_message}, 使用标题或截断内容")
                if title:
                    summary = title
                else:
                    summary = content[:500] if len(content) > 500 else content
                result.summary_generated = False
            else:
                summary = summary_result.content
                result.summary_generated = True
                result.summary_length = len(summary) if summary else 0
            
            # 3. 打分
            score_result = self.scorer.score(
                content=summary,
                data_type=category.value
            )
            
            if score_result.success:
                result.success = True
                result.score = float(score_result.score) if score_result.score is not None else 0.0  # 确保float类型
                result.reason = score_result.reason
            else:
                result.error_message = f"打分失败: {score_result.reason}"
            
        except Exception as e:
            result.error_message = f"处理异常: {str(e)}"
            self.logger.error(f"处理记录 {record_id} 异常: {e}")
        
        result.elapsed_time = time.time() - start_time
        return result
    
    def process_batch(
        self,
        records: List[Dict[str, Any]],
        category: DataCategory
    ) -> List[ProcessingResult]:
        """批量处理PDF记录"""
        results = []
        
        for i, record in enumerate(records):
            result = self.process_single(record, category)
            results.append(result)
            
            # 添加延迟防止API限流
            if i < len(records) - 1:
                time.sleep(self.config.request_delay)
            
            # 进度日志
            if (i + 1) % 10 == 0:
                success = sum(1 for r in results if r.success)
                self.logger.info(
                    f"PDF批量处理进度: {i+1}/{len(records)}, 成功: {success}"
                )
        
        return results


class ExchangePipeline(BasePipeline):
    """
    交易所公告处理流水线
    
    用于 exchange
    流程: title -> rule_scorer -> score
    使用规则打分，不需要LLM，速度快
    """
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        
        # 初始化规则打分器
        self.rule_scorer = RuleScorer()
        
        # GPU加速支持
        self.use_gpu = config.use_gpu
        self._cudf_available = False
        
        if self.use_gpu:
            try:
                import cudf
                self._cudf_available = True
                self.logger.info("Exchange流水线: cuDF GPU加速已启用")
            except ImportError:
                self.logger.warning("cuDF不可用，将使用CPU处理")
        
        self.logger.info("Exchange流水线初始化完成")
    
    def process_single(
        self,
        record: Dict[str, Any],
        category: DataCategory
    ) -> ProcessingResult:
        """处理单条交易所公告"""
        start_time = time.time()
        
        # 获取字段映射
        field_map = FIELD_MAPPING.get(category, {})
        
        # 提取基础信息
        record_id = str(record.get(field_map.get('id', 'news_id'), ''))
        ts_code = str(record.get(field_map.get('ts_code', 'stock_code'), ''))
        date = str(record.get(field_map.get('date', 'date'), ''))
        title = record.get(field_map.get('title', 'title'), '')
        
        result = ProcessingResult(
            success=False,
            record_id=record_id,
            ts_code=ts_code,
            date=date,
        )
        
        try:
            if not title:
                result.error_message = "标题为空"
                result.elapsed_time = time.time() - start_time
                return result
            
            # 使用规则打分
            score_result = self.rule_scorer.score(title)
            
            result.success = True
            result.score = float(score_result.score) if score_result.score is not None else 0.0  # 确保float类型
            # Exchange不需要reason
            result.reason = None
            
        except Exception as e:
            result.error_message = f"处理异常: {str(e)}"
            self.logger.error(f"处理记录 {record_id} 异常: {e}")
        
        result.elapsed_time = time.time() - start_time
        return result
    
    def process_batch(
        self,
        records: List[Dict[str, Any]],
        category: DataCategory
    ) -> List[ProcessingResult]:
        """
        批量处理交易所公告
        
        使用GPU加速的规则打分
        """
        field_map = FIELD_MAPPING.get(category, {})
        
        # 提取标题列表
        titles = [r.get(field_map.get('title', 'title'), '') for r in records]
        
        # 批量规则打分
        start_time = time.time()
        score_results = self.rule_scorer.score_batch(titles)
        batch_time = time.time() - start_time
        
        # 构建结果
        results = []
        for i, (record, score_result) in enumerate(zip(records, score_results)):
            record_id = str(record.get(field_map.get('id', 'news_id'), ''))
            ts_code = str(record.get(field_map.get('ts_code', 'stock_code'), ''))
            date = str(record.get(field_map.get('date', 'date'), ''))
            
            result = ProcessingResult(
                success=True,
                record_id=record_id,
                ts_code=ts_code,
                date=date,
                score=float(score_result.score) if score_result.score is not None else 0.0,  # 确保float类型
                reason=None,  # Exchange不需要reason
                elapsed_time=batch_time / len(records),
            )
            results.append(result)
        
        self.logger.info(
            f"Exchange批量处理完成: {len(records)}条, "
            f"耗时 {batch_time:.3f}s ({batch_time/len(records)*1000:.1f}ms/条)"
        )
        
        return results
    
    def process_dataframe_gpu(self, df: pd.DataFrame, category: DataCategory) -> pd.DataFrame:
        """
        使用GPU加速处理DataFrame
        
        Args:
            df: 输入DataFrame
            category: 数据类别
            
        Returns:
            处理后的DataFrame（包含id, ts_code, date, score列）
        """
        field_map = FIELD_MAPPING.get(category, {})
        
        # 提取必要列
        id_col = field_map.get('id', 'news_id')
        code_col = field_map.get('ts_code', 'stock_code')
        date_col = field_map.get('date', 'date')
        title_col = field_map.get('title', 'title')
        
        # 批量打分
        titles = df[title_col].tolist()
        score_results = self.rule_scorer.score_batch(titles)
        
        # 构建输出DataFrame（确保score是float类型）
        result_df = pd.DataFrame({
            'id': df[id_col].astype(str),
            'ts_code': df[code_col].astype(str),
            'date': df[date_col].astype(str),
            'score': [float(r.score) for r in score_results],  # 确保float类型
        })
        
        # 再次确保score列是float类型
        result_df['score'] = result_df['score'].astype(float)
        
        return result_df


@dataclass
class CCTVProcessingResult:
    """CCTV新闻处理结果"""
    success: bool
    record_id: str
    date: str
    market_sentiment: float = 0.0  # 市场情绪分 (-100 ~ 100)
    beta_signal: float = 1.0       # Beta信号 (0.5 ~ 1.5)
    keywords: List[str] = None     # 关键词
    tone_analysis: str = ""        # 语气分析
    error_message: Optional[str] = None
    elapsed_time: float = 0.0
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


@dataclass  
class PolicyProcessingResult:
    """政策数据处理结果"""
    success: bool
    record_id: str
    date: str
    summary: str = ""                           # 政策摘要
    benefited_industries: List[str] = None      # 受益行业
    harmed_industries: List[str] = None         # 受损行业
    industry_scores: Dict[str, float] = None    # 行业打分详情
    error_message: Optional[str] = None
    elapsed_time: float = 0.0
    
    def __post_init__(self):
        if self.benefited_industries is None:
            self.benefited_industries = []
        if self.harmed_industries is None:
            self.harmed_industries = []
        if self.industry_scores is None:
            self.industry_scores = {}


class CCTVPipeline(BasePipeline):
    """
    CCTV新闻处理流水线
    
    用于 news/cctv
    流程: content -> keyword_extractor -> sentiment_scorer -> market_sentiment + beta_signal
    
    特点：
    - 跳过content_extractor，直接使用已有的content字段
    - 使用LLM进行关键词提取和语气分析
    - 生成市场情绪指数和Beta信号
    """
    
    # 利好关键词及权重
    BULLISH_KEYWORDS = {
        "稳增长": 15,
        "流动性合理充裕": 20,
        "降准": 25,
        "降息": 25,
        "支持民营经济": 15,
        "减税": 15,
        "降费": 15,
        "扩大内需": 10,
        "稳就业": 10,
        "稳投资": 10,
        "促消费": 10,
        "新基建": 12,
        "科技创新": 12,
        "高质量发展": 8,
        "改革开放": 8,
        "市场化": 10,
        "优化营商环境": 10,
        "放管服": 8,
        "激发市场活力": 12,
        "纾困": 15,
        "复工复产": 10,
    }
    
    # 利空关键词及权重
    BEARISH_KEYWORDS = {
        "去杠杆": -20,
        "遏制": -15,
        "泡沫": -20,
        "防范风险": -12,
        "整顿": -15,
        "规范": -10,
        "收紧": -18,
        "严控": -15,
        "问责": -10,
        "反垄断": -12,
        "房住不炒": -15,
        "调控": -10,
        "去产能": -12,
        "淘汰": -12,
        "清理": -10,
        "限制": -12,
        "暂停": -15,
        "叫停": -18,
        "查处": -15,
        "处罚": -12,
    }
    
    # 中性关键词（外事访问等）
    NEUTRAL_KEYWORDS = [
        "外事访问", "会见", "国事访问", "友好访问",
        "文化交流", "体育", "赛事", "艺术节",
        "社会主义核心价值观", "精神文明",
    ]
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        
        # 初始化LLM客户端用于复杂分析
        from ..summarizer.llm_client import LLMClient
        from ..summarizer import SummarizerConfig
        
        summarizer_config = SummarizerConfig(
            model_name=config.model_name,
            ollama_host=config.ollama_host,
            timeout=config.llm_timeout,
        )
        self.llm_client = LLMClient(config=summarizer_config)
        
        self.logger.info("CCTV流水线初始化完成")
    
    def _rule_based_score(self, content: str) -> Tuple[float, List[str], str]:
        """
        基于规则的情绪打分
        
        Returns:
            (score, keywords, tone)
        """
        score = 0.0
        found_keywords = []
        
        # 检测利好关键词
        for keyword, weight in self.BULLISH_KEYWORDS.items():
            if keyword in content:
                score += weight
                found_keywords.append(f"+{keyword}")
        
        # 检测利空关键词
        for keyword, weight in self.BEARISH_KEYWORDS.items():
            if keyword in content:
                score += weight  # weight已经是负数
                found_keywords.append(f"-{keyword}")
        
        # 检测中性关键词
        neutral_count = sum(1 for kw in self.NEUTRAL_KEYWORDS if kw in content)
        if neutral_count > 0 and not found_keywords:
            # 如果只有中性关键词，倾向于中性
            score = 0.0
            found_keywords.append("中性内容")
        
        # 限制分数范围
        score = max(-100, min(100, score))
        
        # 判断语气
        if score > 10:
            tone = "正面/利好"
        elif score < -10:
            tone = "负面/利空"
        else:
            tone = "中性"
        
        return score, found_keywords, tone
    
    def _llm_analysis(self, content: str) -> Dict[str, Any]:
        """使用LLM进行深度分析"""
        from ..summarizer.prompts import PromptTemplates
        
        try:
            # 使用CCTV专用prompt
            prompt_template = PromptTemplates.CCTV_ANALYSIS
            messages = [
                {"role": "system", "content": prompt_template.system},
                {"role": "user", "content": prompt_template.format_user(content=content)},
            ]
            
            response = self.llm_client.chat(messages)
            
            # 解析JSON响应 - response是LLMResponse对象
            if response:
                response_text = response.content if hasattr(response, 'content') else str(response)
                # 尝试提取JSON
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
        except Exception as e:
            self.logger.warning(f"LLM分析失败: {e}")
        
        return {}
    
    def _calculate_beta_signal(self, sentiment_score: float) -> float:
        """
        根据市场情绪计算Beta信号
        
        Args:
            sentiment_score: 市场情绪分 (-100 ~ 100)
            
        Returns:
            Beta信号 (0.5 ~ 1.5)
            - 1.0 表示正常仓位
            - >1.0 表示可以加仓
            - <1.0 表示应该减仓
        """
        # 线性映射: -100 -> 0.5, 0 -> 1.0, 100 -> 1.5
        beta = 1.0 + (sentiment_score / 200.0)
        return max(0.5, min(1.5, beta))
    
    def process_single(
        self,
        record: Dict[str, Any],
        category: DataCategory
    ) -> CCTVProcessingResult:
        """
        处理单条CCTV新闻
        
        完整流程:
        1. 获取content字段（CCTV数据已有content，无需提取）
        2. 规则预打分（快速筛选明显利好/利空）
        3. LLM深度分析（关键词提取、语气分析、市场信号）
        4. 综合计算市场情绪和Beta信号
        """
        start_time = time.time()
        
        field_map = FIELD_MAPPING.get(category, {})
        
        record_id = str(record.get(field_map.get('id', 'news_id'), ''))
        date = str(record.get(field_map.get('date', 'date'), ''))
        content = record.get(field_map.get('content', 'content'), '')
        title = record.get(field_map.get('title', 'title'), '')
        
        result = CCTVProcessingResult(
            success=False,
            record_id=record_id,
            date=date,
        )
        
        try:
            # 验证：必须有content
            if not content or not content.strip():
                # 如果没有content，尝试使用title
                if title and title.strip():
                    content = title
                    self.logger.warning(f"记录 {record_id} 无content，使用title代替")
                else:
                    result.error_message = "content为空"
                    result.elapsed_time = time.time() - start_time
                    return result
            
            # 合并title用于完整分析
            full_content = f"{title}\n{content}" if title and title != content else content
            
            # 1. 规则打分（快速预筛选）
            rule_score, keywords, tone = self._rule_based_score(full_content)
            
            # 2. LLM深度分析（强制使用LLM对所有CCTV新闻进行分析）
            llm_analysis = {}
            if self.config.llm_timeout > 0:
                # 对所有新闻使用LLM分析，不仅限于规则打分不确定的
                llm_analysis = self._llm_analysis(full_content[:3000])  # 限制长度
            
            # 3. 综合打分
            if llm_analysis:
                # 优先使用LLM的关键词分析
                if 'keywords' in llm_analysis and llm_analysis['keywords']:
                    keywords = llm_analysis.get('keywords', [])
                
                # 使用LLM的语气分析
                if 'tone' in llm_analysis:
                    tone = llm_analysis.get('tone', tone)
                    tone_reason = llm_analysis.get('tone_reason', '')
                    if tone_reason:
                        tone = f"{tone}: {tone_reason}"
                
                # 根据market_signal调整分数
                signal = llm_analysis.get('market_signal', '')
                if '强烈利好' in signal or '重大利好' in signal:
                    rule_score = max(rule_score, 40)
                elif '利好' in signal:
                    rule_score = max(rule_score, 20) if rule_score >= 0 else max(rule_score, 10)
                elif '强烈利空' in signal or '重大利空' in signal:
                    rule_score = min(rule_score, -40)
                elif '利空' in signal:
                    rule_score = min(rule_score, -20) if rule_score <= 0 else min(rule_score, -10)
            
            # 4. 计算Beta信号
            beta_signal = self._calculate_beta_signal(rule_score)
            
            result.success = True
            result.market_sentiment = float(rule_score)
            result.beta_signal = float(beta_signal)
            result.keywords = keywords if isinstance(keywords, list) else [keywords]
            result.tone_analysis = tone
            
        except Exception as e:
            result.error_message = f"处理异常: {str(e)}"
            self.logger.error(f"处理CCTV记录 {record_id} 异常: {e}")
        
        result.elapsed_time = time.time() - start_time
        return result
    
    def process_batch(
        self,
        records: List[Dict[str, Any]],
        category: DataCategory
    ) -> List[CCTVProcessingResult]:
        """批量处理CCTV新闻"""
        results = []
        
        for i, record in enumerate(records):
            result = self.process_single(record, category)
            results.append(result)
            
            if i < len(records) - 1:
                time.sleep(self.config.request_delay * 0.5)  # CCTV处理较快
            
            if (i + 1) % 10 == 0:
                success = sum(1 for r in results if r.success)
                self.logger.info(f"CCTV批量处理进度: {i+1}/{len(records)}, 成功: {success}")
        
        return results
    
    def aggregate_daily_sentiment(
        self,
        results: List[CCTVProcessingResult],
        date: str
    ) -> Dict[str, Any]:
        """
        聚合当日所有新闻的市场情绪
        
        Args:
            results: 当日所有新闻的处理结果
            date: 日期
            
        Returns:
            当日综合情绪指标
        """
        if not results:
            return {
                'date': date,
                'market_sentiment': 0.0,
                'beta_signal': 1.0,
                'news_count': 0,
            }
        
        valid_results = [r for r in results if r.success]
        if not valid_results:
            return {
                'date': date,
                'market_sentiment': 0.0,
                'beta_signal': 1.0,
                'news_count': len(results),
            }
        
        # 加权平均（可以按新闻重要性加权，这里简单平均）
        avg_sentiment = sum(r.market_sentiment for r in valid_results) / len(valid_results)
        avg_beta = sum(r.beta_signal for r in valid_results) / len(valid_results)
        
        # 收集所有关键词
        all_keywords = []
        for r in valid_results:
            all_keywords.extend(r.keywords)
        
        return {
            'date': date,
            'market_sentiment': round(avg_sentiment, 2),
            'beta_signal': round(avg_beta, 3),
            'news_count': len(valid_results),
            'top_keywords': list(set(all_keywords))[:10],
        }


class PolicyPipeline(BasePipeline):
    """
    政策数据处理流水线
    
    用于 policy/gov, policy/ndrc
    流程: URL -> HTMLExtractor提取原始文本 -> LLM摘要生成 -> LLM行业映射打分
    
    特点：
    - 使用HTMLExtractor提取HTML正文（基于trafilatura + readability）
    - LLM生成政策摘要
    - LLM进行行业映射和打分
    - 生成行业打分和轮动信号
    """
    
    # 申万一级行业列表
    SW_INDUSTRIES = [
        "农林牧渔", "基础化工", "钢铁", "有色金属", "电子", "汽车", 
        "家用电器", "食品饮料", "纺织服饰", "轻工制造", "医药生物",
        "公用事业", "交通运输", "房地产", "商贸零售", "社会服务", 
        "银行", "非银金融", "综合", "建筑材料", "建筑装饰", "电力设备",
        "机械设备", "国防军工", "计算机", "传媒", "通信", "煤炭", 
        "石油石化", "环保", "美容护理"
    ]
    
    # 高影响政策关键词
    HIGH_IMPACT_KEYWORDS = {
        "加大投入": 80,
        "补贴": 70,
        "支持": 50,
        "鼓励": 40,
        "重点发展": 60,
        "战略性": 50,
        "优先": 40,
        "扶持": 60,
    }
    
    # 负面政策关键词
    NEGATIVE_KEYWORDS = {
        "限制": -60,
        "规范": -30,
        "整顿": -50,
        "收紧": -60,
        "严控": -70,
        "淘汰": -80,
        "禁止": -90,
        "清退": -70,
    }
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        
        # 初始化HTML提取器（使用专用的HTMLExtractor）
        from ..content_extractor import HTMLExtractor
        self.html_extractor = HTMLExtractor(
            timeout=30,
            max_retries=3,
            prefer_trafilatura=True,
            include_tables=True,
        )
        
        # 初始化LLM客户端
        from ..summarizer.llm_client import LLMClient
        from ..summarizer import SummarizerConfig, Summarizer, DataType
        
        summarizer_config = SummarizerConfig(
            model_name=config.model_name,
            ollama_host=config.ollama_host,
            timeout=config.llm_timeout,
        )
        self.llm_client = LLMClient(config=summarizer_config)
        
        # 初始化摘要生成器（用于清洗和生成概要）
        self.summarizer = Summarizer(config=summarizer_config)
        self.DataType = DataType
        
        self.logger.info("Policy流水线初始化完成")
    
    def _extract_html_content(self, url: str) -> Tuple[bool, str]:
        """
        使用HTMLExtractor提取HTML正文
        
        流程: URL -> HTMLExtractor(trafilatura + readability) -> 原始文本
        """
        try:
            # 使用HTMLExtractor提取内容
            result = self.html_extractor.extract(url)
            if result.success and result.content_text:
                content = result.content_text
                if len(content) > 100:
                    self.logger.debug(f"HTML提取成功: {len(content)} 字符")
                    return True, content
                else:
                    self.logger.warning(f"HTML内容过短: {len(content)} 字符")
        except Exception as e:
            self.logger.warning(f"HTMLExtractor提取失败: {e}")
        
        return False, ""
    
    def _llm_sector_mapping(self, title: str, content: str) -> Dict[str, Any]:
        """使用LLM进行行业映射"""
        from ..summarizer.prompts import PromptTemplates
        
        try:
            prompt_template = PromptTemplates.POLICY_SECTOR_MAPPING
            
            # 限制内容长度
            truncated_content = content[:3000] if len(content) > 3000 else content
            
            messages = [
                {"role": "system", "content": prompt_template.system},
                {"role": "user", "content": prompt_template.format_user(
                    title=title,
                    content=truncated_content
                )},
            ]
            
            response = self.llm_client.chat(messages)
            
            if response:
                # response是LLMResponse对象
                response_text = response.content if hasattr(response, 'content') else str(response)
                # 尝试提取JSON
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
        except Exception as e:
            self.logger.warning(f"LLM行业映射失败: {e}")
        
        return {}
    
    def _calculate_industry_scores(
        self,
        llm_result: Dict[str, Any],
        content: str
    ) -> Dict[str, float]:
        """
        计算行业打分
        
        基于LLM分析结果和关键词权重
        """
        scores = {}
        
        # 从LLM结果提取受益行业
        benefited = llm_result.get('benefited_industries', [])
        for item in benefited:
            if isinstance(item, dict):
                industry = item.get('industry', '')
                impact = item.get('impact_level', '中')
                if industry in self.SW_INDUSTRIES:
                    base_score = 80 if impact == '高' else 50 if impact == '中' else 30
                    scores[industry] = float(base_score)
            elif isinstance(item, str) and item in self.SW_INDUSTRIES:
                scores[item] = 50.0
        
        # 从LLM结果提取受损行业
        harmed = llm_result.get('harmed_industries', [])
        for item in harmed:
            if isinstance(item, dict):
                industry = item.get('industry', '')
                impact = item.get('impact_level', '中')
                if industry in self.SW_INDUSTRIES:
                    base_score = -80 if impact == '高' else -50 if impact == '中' else -30
                    scores[industry] = float(base_score)
            elif isinstance(item, str) and item in self.SW_INDUSTRIES:
                scores[item] = -50.0
        
        # 基于关键词调整
        for keyword, weight in self.HIGH_IMPACT_KEYWORDS.items():
            if keyword in content:
                for industry, score in list(scores.items()):
                    if score > 0:
                        scores[industry] = float(min(100.0, score + weight * 0.2))
        
        for keyword, weight in self.NEGATIVE_KEYWORDS.items():
            if keyword in content:
                for industry, score in list(scores.items()):
                    if score < 0:
                        scores[industry] = float(max(-100.0, score + weight * 0.2))
        
        # 确保所有值都是float类型
        return {k: float(v) for k, v in scores.items()}
    
    def _llm_summarize_content(self, title: str, content: str) -> str:
        """
        使用LLM生成政策摘要
        
        流程: 原始文本 -> LLM清洗/整理 -> 摘要
        """
        try:
            # 使用Summarizer生成摘要
            summary_result = self.summarizer.summarize(
                content_text=content,
                data_type=self.DataType.POLICY,
                title=title
            )
            
            if summary_result.success and summary_result.content:
                return summary_result.content
        except Exception as e:
            self.logger.warning(f"LLM摘要生成失败: {e}")
        
        # 降级：返回标题
        return title
    
    def process_single(
        self,
        record: Dict[str, Any],
        category: DataCategory
    ) -> PolicyProcessingResult:
        """
        处理单条政策数据
        
        完整流程:
        1. URL -> HTMLExtractor提取原始文本
        2. 原始文本 -> LLM生成摘要(清洗、整理、概要)
        3. 摘要 -> LLM行业映射和打分
        """
        start_time = time.time()
        
        field_map = FIELD_MAPPING.get(category, {})
        
        record_id = str(record.get(field_map.get('id', 'id'), ''))
        date = str(record.get(field_map.get('date', 'date'), ''))
        title = record.get(field_map.get('title', 'title'), '')
        url = record.get(field_map.get('url', 'url'), '')
        
        # 检查是否直接提供了内容
        direct_content = record.get('content', '')
        
        result = PolicyProcessingResult(
            success=False,
            record_id=record_id,
            date=date,
        )
        
        try:
            # 1. 获取原始内容（优先从URL提取，其次使用直接提供的内容）
            raw_content = ''
            if url:
                success, extracted_content = self._extract_html_content(url)
                if success and extracted_content:
                    raw_content = extracted_content
                    self.logger.info(f"从URL提取内容成功: {len(raw_content)} 字符")
            
            if not raw_content and direct_content:
                raw_content = direct_content
            
            if not raw_content:
                # 最后降级：使用标题
                raw_content = title
                if not raw_content:
                    result.error_message = "无法提取内容"
                    result.elapsed_time = time.time() - start_time
                    return result
                self.logger.warning(f"降级使用标题作为内容: {title[:50]}...")
            
            # 2. LLM生成摘要（清洗、整理、概要）
            summary = self._llm_summarize_content(title, raw_content)
            
            # 3. LLM行业映射（使用摘要，确保LLM能理解关键信息）
            llm_result = self._llm_sector_mapping(title, summary if summary else raw_content)
            
            # 4. 计算行业打分
            industry_scores = self._calculate_industry_scores(llm_result, raw_content)
            
            # 5. 构建结果
            result.success = True
            # 使用LLM生成的摘要，如果没有则使用LLM行业映射中的summary
            result.summary = summary if summary else llm_result.get('summary', title)[:500]
            
            # 提取受益/受损行业名称
            benefited = llm_result.get('benefited_industries', [])
            harmed = llm_result.get('harmed_industries', [])
            
            result.benefited_industries = [
                (item.get('industry') if isinstance(item, dict) else item)
                for item in benefited
                if (isinstance(item, dict) and item.get('industry') in self.SW_INDUSTRIES)
                   or (isinstance(item, str) and item in self.SW_INDUSTRIES)
            ]
            
            result.harmed_industries = [
                (item.get('industry') if isinstance(item, dict) else item)
                for item in harmed
                if (isinstance(item, dict) and item.get('industry') in self.SW_INDUSTRIES)
                   or (isinstance(item, str) and item in self.SW_INDUSTRIES)
            ]
            
            result.industry_scores = industry_scores
            
        except Exception as e:
            result.error_message = f"处理异常: {str(e)}"
            self.logger.error(f"处理Policy记录 {record_id} 异常: {e}")
        
        result.elapsed_time = time.time() - start_time
        return result
    
    def process_batch(
        self,
        records: List[Dict[str, Any]],
        category: DataCategory
    ) -> List[PolicyProcessingResult]:
        """批量处理政策数据"""
        results = []
        
        for i, record in enumerate(records):
            result = self.process_single(record, category)
            results.append(result)
            
            if i < len(records) - 1:
                time.sleep(self.config.request_delay)
            
            if (i + 1) % 5 == 0:
                success = sum(1 for r in results if r.success)
                self.logger.info(f"Policy批量处理进度: {i+1}/{len(records)}, 成功: {success}")
        
        return results


def create_pipeline(
    category: DataCategory,
    config: ProcessingConfig
) -> BasePipeline:
    """
    创建对应类别的处理流水线
    
    Args:
        category: 数据类别
        config: 处理配置
        
    Returns:
        对应的流水线实例
    """
    if category == DataCategory.EXCHANGE:
        return ExchangePipeline(config)
    elif category == DataCategory.CCTV:
        return CCTVPipeline(config)
    elif category in (DataCategory.POLICY_GOV, DataCategory.POLICY_NDRC):
        return PolicyPipeline(config)
    else:
        return PDFPipeline(config)
