"""
时间元数据标准化模块 (Time Metadata Cleaner)

核心职责：防止回测中的"未来函数"问题

设计原则：
1. 统一时间格式为 'YYYY-MM-DD HH:MM:SS'
2. 对只有日期的数据采用"保守策略"填充时分秒
3. 避免数据泄露：使用盘后时间（17:00）确保 T 日内不可用

背景：
- 实时数据源常提供不完整的时间戳（如仅日期 2023-01-05）
- 在回测中直接使用会导致"未来函数"：即在 2023-01-05 盘中就看到该日发布的新闻
- 保守策略：假设该消息在 17:00（盘后）发布 → T 日盘中不可用 → 只在 T+1 生效
  这是防止作弊的基本防线

关键概念：
- publish_time（发布时间）：新闻/公告的标准化发布时间
- time_accuracy（时间精度）：字段精度等级（'Y'/'M'/'D'/'H'/'Mi'/'S'）
  用于策略回测时判断是否能使用该数据
"""

import re
import logging
from typing import Optional, Tuple, Literal
from datetime import datetime, time
from enum import Enum

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)


class TimeMode(str, Enum):
    """时间填充模式"""
    CONSERVATIVE = 'conservative'  # 盘后 17:00（最安全）
    AGGRESSIVE = 'aggressive'      # 盘前 08:00（可能泄露，谨慎使用）
    ULTRA_CONSERVATIVE = 'ultra_conservative'  # 次日盘前 09:30（最保守）


class TimeAccuracy(str, Enum):
    """时间精度级别"""
    YEAR = 'Y'          # 仅年份
    MONTH = 'M'         # 年月
    DAY = 'D'           # 完整日期（此时需要填充时分秒）
    HOUR = 'H'          # 小时
    MINUTE = 'Mi'       # 分钟
    SECOND = 'S'        # 秒（最精确）


class TimeNormalizer:
    """
    时间标准化器
    
    负责将各种格式的时间字符串转换为标准格式，
    并对缺失的时分秒部分进行保守填充。
    """
    
    # 常见日期格式（按优先级排序）
    DATE_FORMATS = [
        # ISO 格式
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d',
        '%Y/%m/%d %H:%M:%S',
        '%Y/%m/%d %H:%M',
        '%Y/%m/%d',
        
        # 中文格式
        '%Y年%m月%d日 %H:%M:%S',
        '%Y年%m月%d日 %H:%M',
        '%Y年%m月%d日',
        '%Y.%m.%d %H:%M:%S',
        '%Y.%m.%d',
        
        # 其他格式
        '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y',
    ]
    
    # 中文时间关键词正则
    CHINESE_TIME_PATTERNS = [
        # 标准中文格式：2023年5月12日 14:30
        r'(\d{4})年(\d{1,2})月(\d{1,2})日\s*(?:(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?',
        
        # 英文月份：2023-May-12 14:30
        r'(\d{4})-([A-Za-z]{3,})-(\d{1,2})\s*(?:(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?',
        
        # 相对时间：今天 14:30，昨天，前天
        r'(今天|昨天|前天|今年|本月|上月)\s*(?:(\d{1,2}):(\d{1,2}))?',
    ]
    
    @staticmethod
    def detect_time_accuracy(raw_time_str: str) -> TimeAccuracy:
        """
        检测原始时间字符串的精度级别
        
        Args:
            raw_time_str: 原始时间字符串
            
        Returns:
            时间精度等级
            
        Examples:
            '2024' → TimeAccuracy.YEAR
            '2024-01' → TimeAccuracy.MONTH
            '2024-01-15' → TimeAccuracy.DAY
            '2024-01-15 14:30:45' → TimeAccuracy.SECOND
        """
        if not raw_time_str:
            return TimeAccuracy.DAY  # 默认按天处理
        
        s = str(raw_time_str).strip()
        
        # 检查是否包含秒（精度最高）
        if re.search(r':\d{2}:\d{2}$', s):
            return TimeAccuracy.SECOND
        
        # 检查是否包含分钟但无秒
        if re.search(r':\d{1,2}(?::\d{2})?', s):
            # 再检查是否真的有分钟部分（不是小时后直接没了）
            if re.search(r':\d{1,2}(?:[^\d:]|$)', s):
                # 有时:分 格式，但没有秒 → MINUTE
                if not re.search(r':\d{2}:\d{2}', s):
                    return TimeAccuracy.MINUTE
            return TimeAccuracy.SECOND
        
        # 检查是否包含完整日期
        if re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日)', s):
            return TimeAccuracy.DAY
        
        # 检查是否包含月份
        if re.search(r'(\d{4}[-/]\d{1,2}|\d{4}年\d{1,2}月)', s):
            return TimeAccuracy.MONTH
        
        # 否则按年处理
        return TimeAccuracy.YEAR
    
    @classmethod
    def standardize_publish_time(
        cls,
        raw_time_str: str,
        default_time_mode: Literal['conservative', 'aggressive', 'ultra_conservative'] = 'conservative',
        return_accuracy: bool = False
    ) -> str:
        """
        将原始时间字符串标准化为 'YYYY-MM-DD HH:MM:SS' 格式
        
        核心逻辑：防止"未来函数"泄露
        1. 如果原始数据有时分秒，保留原值
        2. 如果原始数据仅有日期，采用填充策略：
           - conservative (默认): 17:00（盘后，最安全）
           - aggressive: 08:00（盘前，可能泄露）
           - ultra_conservative: 次日 09:30（最保守）
        
        Args:
            raw_time_str: 原始时间字符串
            default_time_mode: 时间填充模式
            return_accuracy: 是否同时返回时间精度
            
        Returns:
            标准化后的时间字符串 (格式: 'YYYY-MM-DD HH:MM:SS')
            如果 return_accuracy=True, 返回 (时间字符串, 精度级别)
            
        Examples:
            >>> standardize_publish_time('2023-01-05')
            '2023-01-05 17:00:00'  # 保守模式填充为盘后
            
            >>> standardize_publish_time('2023-01-05 14:30:15')
            '2023-01-05 14:30:15'  # 已有时分秒，直接保留
            
            >>> standardize_publish_time('2023年1月5日')
            '2023-01-05 17:00:00'
        """
        if not raw_time_str:
            return ("", TimeAccuracy.DAY) if return_accuracy else ""
        
        try:
            # 检测原始时间的精度
            accuracy = cls.detect_time_accuracy(raw_time_str)
            
            # 尝试用 pandas 解析
            if pd is not None:
                try:
                    dt = pd.to_datetime(raw_time_str)
                except Exception:
                    dt = None
            else:
                dt = None
            
            # 如果 pandas 失败，尝试逐个格式解析
            if dt is None:
                for fmt in cls.DATE_FORMATS:
                    try:
                        dt = datetime.strptime(raw_time_str, fmt)
                        break
                    except ValueError:
                        continue
                
                if dt is None:
                    logger.warning(f"无法解析时间字符串: {raw_time_str}")
                    return ("", accuracy) if return_accuracy else ""
            
            # 转换为 datetime 对象
            if hasattr(dt, 'to_pydatetime'):
                dt = dt.to_pydatetime()
            
            # 关键判断：原始字符串是否包含时间部分
            # 启发式方法：检查是否匹配时间格式（HH:MM 或更精确）
            has_time_component = bool(re.search(r'\d{1,2}:\d{1,2}', str(raw_time_str)))
            
            if has_time_component:
                # 原数据有时分秒，直接保留
                result = dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                # 原数据仅有日期，执行【防泄露填充】
                if default_time_mode == 'conservative':
                    # 保守模式：盘后 17:00
                    # 逻辑：T 日仅发布日期 → 假设 17:00 发布（盘后）
                    #      → T 日盘中无法使用 → 最早 T+1 使用
                    dt = dt.replace(hour=17, minute=0, second=0)
                
                elif default_time_mode == 'aggressive':
                    # 激进模式：盘前 08:00
                    # 风险：可能在 T 日盘前就用到该信息（未来函数）
                    # 仅在确认源为"盘前专递"时使用
                    dt = dt.replace(hour=8, minute=0, second=0)
                
                elif default_time_mode == 'ultra_conservative':
                    # 超保守模式：次日盘前 09:30
                    # 最安全，但可能太保守
                    from datetime import timedelta
                    dt = dt + timedelta(days=1)
                    dt = dt.replace(hour=9, minute=30, second=0)
                
                result = dt.strftime("%Y-%m-%d %H:%M:%S")
            
            return (result, accuracy) if return_accuracy else result
        
        except Exception as e:
            logger.error(f"时间标准化失败: {raw_time_str}, 错误: {e}")
            return ("", accuracy if 'accuracy' in locals() else TimeAccuracy.DAY) if return_accuracy else ""
    
    @classmethod
    def extract_time_from_text(
        cls,
        text: str,
        max_chars: int = 300
    ) -> Optional[str]:
        """
        从正文前 N 个字符中提取发布时间戳
        
        适用场景：元数据缺失时间时，尝试从正文中找到发布时间
        例如：正文开头 "本报记者 李四 2023年5月12日 报道"
        
        Args:
            text: 要提取时间的正文文本
            max_chars: 仅从前 N 个字符中提取（防止提取错误的时间）
            
        Returns:
            提取的时间字符串，或 None 如果未找到
            
        Examples:
            >>> text = "新华社 北京 2024年1月15日 电：..."
            >>> extract_time_from_text(text)
            '2024-01-15'
        """
        if not text:
            return None
        
        # 仅处理开头部分
        text_head = text[:max_chars] if len(text) > max_chars else text
        
        # 中文日期：2023年5月12日 或 2023-05-12 或 2023/05/12
        patterns = [
            # 中文格式：2024年1月15日
            r'(\d{4})年(\d{1,2})月(\d{1,2})日',
            
            # ISO 格式：2024-01-15
            r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',
            
            # 英文月份：2024-Jan-15
            r'(\d{4})[-/]([A-Za-z]{3,})[-/](\d{1,2})',
            
            # 时间戳：2024年1月15日 14:30
            r'(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2}):(\d{1,2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_head, re.IGNORECASE)
            if match:
                # 找到第一个匹配就返回
                matched_text = match.group(0)
                # 尝试标准化
                standardized = cls.standardize_publish_time(matched_text)
                if standardized:
                    return standardized
        
        return None
    
    @staticmethod
    def normalize_date_range(
        start_date: str,
        end_date: str
    ) -> Tuple[str, str]:
        """
        标准化日期范围
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            标准化后的 (开始日期, 结束日期) 元组
        """
        if not pd:
            return start_date, end_date
        
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            return (
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d")
            )
        except Exception as e:
            logger.warning(f"日期范围标准化失败: {start_date}-{end_date}, {e}")
            return start_date, end_date


# ============================================================
# 便捷函数（供采集器直接调用）
# ============================================================

def standardize_publish_time(
    raw_time_str: str,
    mode: str = 'conservative',
    with_accuracy: bool = False
):
    """
    标准化发布时间便捷函数
    
    Args:
        raw_time_str: 原始时间字符串
        mode: 填充模式 ('conservative'|'aggressive'|'ultra_conservative')
        with_accuracy: 是否同时返回精度信息
        
    Returns:
        标准化后的时间字符串，或 (时间, 精度) 元组
    """
    return TimeNormalizer.standardize_publish_time(
        raw_time_str,
        mode,
        return_accuracy=with_accuracy
    )


def extract_time_from_text(text: str, max_chars: int = 300) -> Optional[str]:
    """
    从文本中提取发布时间
    
    Args:
        text: 正文文本
        max_chars: 搜索范围（字符数）
        
    Returns:
        提取的标准化时间字符串
    """
    return TimeNormalizer.extract_time_from_text(text, max_chars)


def is_future_data(
    publish_time: str,
    backtest_datetime: str
) -> bool:
    """
    检测是否存在"未来函数"问题
    
    检查给定的发布时间是否晚于回测时刻，如果是则存在数据泄露风险
    
    Args:
        publish_time: 发布时间字符串 (YYYY-MM-DD HH:MM:SS)
        backtest_datetime: 回测时刻 (YYYY-MM-DD HH:MM:SS)
        
    Returns:
        True 表示存在未来函数（数据泄露），False 表示安全
        
    Examples:
        >>> is_future_data('2024-01-05 17:00:00', '2024-01-05 14:30:00')
        True  # 发布时间晚于回测时刻，存在泄露
        
        >>> is_future_data('2024-01-05 09:00:00', '2024-01-05 14:30:00')
        False  # 发布时间早于回测时刻，无泄露
    """
    if not pd:
        return False
    
    try:
        pub_dt = pd.to_datetime(publish_time)
        bt_dt = pd.to_datetime(backtest_datetime)
        return pub_dt > bt_dt
    except Exception as e:
        logger.warning(f"未来函数检测失败: {e}")
        return False


if __name__ == '__main__':
    """时间标准化模块测试"""
    
    print("=" * 60)
    print("时间标准化模块测试")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        # (输入, 描述, 期望模式)
        ("2024-01-05", "仅日期（ISO 格式）", "conservative"),
        ("2024年01月05日", "仅日期（中文格式）", "conservative"),
        ("2024-01-05 14:30:45", "完整时间戳", "conservative"),
        ("2024/01/05 14:30", "斜杠格式含时间", "conservative"),
        ("2024-01-05 08:00", "盘前时间", "aggressive"),
    ]
    
    print("\n标准化测试:")
    for raw_time, desc, mode in test_cases:
        standardized, accuracy = TimeNormalizer.standardize_publish_time(
            raw_time, mode, return_accuracy=True
        )
        print(f"  [{desc}]")
        print(f"    输入: {raw_time}")
        print(f"    输出: {standardized}")
        print(f"    精度: {accuracy}")
    
    print("\n精度检测测试:")
    accuracy_tests = [
        ("2024", "年份"),
        ("2024-01", "月份"),
        ("2024-01-05", "日期"),
        ("2024-01-05 14:30:45", "秒"),
    ]
    
    for raw_time, desc in accuracy_tests:
        accuracy = TimeNormalizer.detect_time_accuracy(raw_time)
        print(f"  [{desc}] {raw_time} → {accuracy}")
    
    print("\n文本提取测试:")
    text_samples = [
        "新华社 北京 2024年1月15日 电：上证指数今日涨幅...",
        "2024-01-15 美国 Fed 主席发表讲话。根据...",
    ]
    
    for text in text_samples:
        extracted = extract_time_from_text(text)
        print(f"  原文: {text[:40]}...")
        print(f"  提取时间: {extracted}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
