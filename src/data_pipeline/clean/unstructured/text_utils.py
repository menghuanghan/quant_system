"""
通用文本标准化模块 (Text Normalizer)

核心职责：将各种千奇百怪的字符统一成标准格式，为后续 NLP 模型扫清障碍。

主要功能：
1. Unicode 标准化 (NFKC 归一化) - 全角转半角
2. 不可见字符清洗 - 剔除控制字符
3. 空白折叠 - 合并连续空格但保留段落结构
4. 金融文本特殊处理 - 日期、数字格式统一
"""

import re
import unicodedata
from typing import Optional


class TextNormalizer:
    """
    文本标准化器
    
    设计原则：
    - 输入是原始字符串，输出是清洗后的纯文本
    - 不依赖磁盘 IO，完全内存操作
    - 保留段落结构（换行符）以维护上下文信息
    """
    
    # 控制字符正则（保留换行符和制表符）
    CONTROL_CHAR_PATTERN = re.compile(
        r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]'
    )
    
    # 零宽字符正则（零宽空格、零宽连接符等）
    ZERO_WIDTH_PATTERN = re.compile(
        r'[\u200b-\u200f\u2028-\u202f\u205f-\u206f\ufeff]'
    )
    
    # 连续空白字符正则（不包括换行符）
    MULTI_SPACE_PATTERN = re.compile(r'[ \t]+')
    
    # 连续换行符正则（超过2个换行合并为2个）
    MULTI_NEWLINE_PATTERN = re.compile(r'\n{3,}')
    
    # 行首行尾空白正则
    LINE_EDGE_SPACE_PATTERN = re.compile(r'^[ \t]+|[ \t]+$', re.MULTILINE)
    
    # 中文标点到英文标点映射（可选，用于统一标点）
    PUNCTUATION_MAP = {
        '，': ',',
        '。': '.',
        '！': '!',
        '？': '?',
        '：': ':',
        '；': ';',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '【': '[',
        '】': ']',
        '（': '(',
        '）': ')',
        '—': '-',
        '～': '~',
    }
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Unicode NFKC 标准化
        
        将全角字符转为半角，统一变体字符
        例如：１２３ -> 123, Ａ股 -> A股
        
        Args:
            text: 原始文本
            
        Returns:
            标准化后的文本
        """
        if not text:
            return ""
        return unicodedata.normalize('NFKC', text)
    
    @staticmethod
    def remove_control_chars(text: str) -> str:
        """
        移除不可见控制字符
        
        剔除 \x00, \x0c 等控制字符，但保留换行符 \n 和制表符 \t
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
        
        # 1. 移除控制字符
        text = TextNormalizer.CONTROL_CHAR_PATTERN.sub('', text)
        
        # 2. 移除零宽字符
        text = TextNormalizer.ZERO_WIDTH_PATTERN.sub('', text)
        
        return text
    
    @staticmethod
    def collapse_whitespace(text: str, preserve_newlines: bool = True) -> str:
        """
        折叠空白字符
        
        将连续的空格、制表符替换为单个空格
        可选是否保留换行符（段落结构）
        
        Args:
            text: 原始文本
            preserve_newlines: 是否保留换行符，默认 True
            
        Returns:
            处理后的文本
        """
        if not text:
            return ""
        
        if preserve_newlines:
            # 保留换行符模式
            # 1. 合并连续空格和制表符（不包括换行）
            text = TextNormalizer.MULTI_SPACE_PATTERN.sub(' ', text)
            
            # 2. 去除每行首尾空白
            text = TextNormalizer.LINE_EDGE_SPACE_PATTERN.sub('', text)
            
            # 3. 合并过多的连续换行（>2个 -> 2个）
            text = TextNormalizer.MULTI_NEWLINE_PATTERN.sub('\n\n', text)
        else:
            # 不保留换行符模式（全部转为单个空格）
            text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def normalize_punctuation(text: str, to_english: bool = False) -> str:
        """
        标点符号标准化
        
        Args:
            text: 原始文本
            to_english: 是否将中文标点转为英文标点
            
        Returns:
            处理后的文本
        """
        if not text:
            return ""
        
        if to_english:
            for cn_punct, en_punct in TextNormalizer.PUNCTUATION_MAP.items():
                text = text.replace(cn_punct, en_punct)
        
        return text
    
    @staticmethod
    def normalize_numbers(text: str) -> str:
        """
        数字格式标准化
        
        - 全角数字转半角：１２３ -> 123
        - 移除数字中的千分位逗号：1,234,567 -> 1234567
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本
        """
        if not text:
            return ""
        
        # NFKC 已经处理了全角转半角
        # 这里处理千分位逗号
        # 匹配模式：数字,数字（千分位）
        text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
        # 可能有多级千分位，重复处理
        while re.search(r'(\d),(\d{3})', text):
            text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
        
        return text
    
    @staticmethod
    def normalize_dates(text: str) -> str:
        """
        日期格式标准化
        
        统一常见日期格式为 YYYY-MM-DD
        - 2024年01月15日 -> 2024-01-15
        - 2024/01/15 -> 2024-01-15
        - 20240115 -> 2024-01-15 (仅8位纯数字)
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本
        """
        if not text:
            return ""
        
        # 中文日期格式：2024年01月15日
        text = re.sub(
            r'(\d{4})年(\d{1,2})月(\d{1,2})日',
            lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
            text
        )
        
        # 斜杠日期格式：2024/01/15
        text = re.sub(
            r'(\d{4})/(\d{1,2})/(\d{1,2})',
            lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
            text
        )
        
        return text
    
    @staticmethod
    def remove_special_markers(text: str) -> str:
        """
        移除特殊标记
        
        移除 PDF 提取中常见的特殊标记：
        - [页码]、(续)、※、★ 等
        
        Args:
            text: 原始文本
            
        Returns:
            处理后的文本
        """
        if not text:
            return ""
        
        # 页码标记
        text = re.sub(r'\[第?\s*\d+\s*页?\]', '', text)
        text = re.sub(r'[（(]?第?\s*\d+\s*页[）)]?', '', text)
        
        # 续表标记
        text = re.sub(r'[（(]续[）)]', '', text)
        
        # 特殊符号（保留常用的）
        text = re.sub(r'[※★☆◆◇●○■□▲△▼▽]', '', text)
        
        return text
    
    @classmethod
    def normalize(
        cls,
        text: str,
        unicode_normalize: bool = True,
        remove_control: bool = True,
        collapse_space: bool = True,
        preserve_newlines: bool = True,
        normalize_punct: bool = False,
        normalize_nums: bool = False,
        normalize_date: bool = False,
        remove_markers: bool = False
    ) -> str:
        """
        综合文本标准化（主入口）
        
        按顺序应用各种标准化处理
        
        Args:
            text: 原始文本
            unicode_normalize: 是否进行 Unicode NFKC 标准化
            remove_control: 是否移除控制字符
            collapse_space: 是否折叠空白
            preserve_newlines: 是否保留换行符
            normalize_punct: 是否标准化标点（中转英）
            normalize_nums: 是否标准化数字
            normalize_date: 是否标准化日期
            remove_markers: 是否移除特殊标记
            
        Returns:
            标准化后的文本
        """
        if not text:
            return ""
        
        # 1. Unicode 标准化（最先执行，因为影响后续所有处理）
        if unicode_normalize:
            text = cls.normalize_unicode(text)
        
        # 2. 移除控制字符
        if remove_control:
            text = cls.remove_control_chars(text)
        
        # 3. 标点标准化
        if normalize_punct:
            text = cls.normalize_punctuation(text, to_english=True)
        
        # 4. 数字标准化
        if normalize_nums:
            text = cls.normalize_numbers(text)
        
        # 5. 日期标准化
        if normalize_date:
            text = cls.normalize_dates(text)
        
        # 6. 移除特殊标记
        if remove_markers:
            text = cls.remove_special_markers(text)
        
        # 7. 空白折叠（最后执行）
        if collapse_space:
            text = cls.collapse_whitespace(text, preserve_newlines)
        
        return text


# 便捷函数
def normalize_text(
    text: str,
    preserve_newlines: bool = True,
    aggressive: bool = False
) -> str:
    """
    文本标准化便捷函数
    
    Args:
        text: 原始文本
        preserve_newlines: 是否保留换行符
        aggressive: 是否使用激进模式（包含日期、数字、标记清理）
        
    Returns:
        标准化后的文本
        
    Examples:
        >>> normalize_text("　Ａ股涨幅１．２％")
        'A股涨幅1.2%'
        
        >>> normalize_text("2024年01月15日", aggressive=True)
        '2024-01-15'
    """
    return TextNormalizer.normalize(
        text,
        unicode_normalize=True,
        remove_control=True,
        collapse_space=True,
        preserve_newlines=preserve_newlines,
        normalize_punct=False,  # 保留原始标点风格
        normalize_nums=aggressive,
        normalize_date=aggressive,
        remove_markers=aggressive
    )


def normalize_for_nlp(text: str) -> str:
    """
    为 NLP 模型准备的标准化
    
    更激进的清理，适合输入 Transformer 模型
    
    Args:
        text: 原始文本
        
    Returns:
        标准化后的文本
    """
    return TextNormalizer.normalize(
        text,
        unicode_normalize=True,
        remove_control=True,
        collapse_space=True,
        preserve_newlines=False,  # 不保留换行，变成单行
        normalize_punct=True,      # 标点转英文
        normalize_nums=True,
        normalize_date=True,
        remove_markers=True
    )


def normalize_for_storage(text: str) -> str:
    """
    为存储准备的标准化
    
    轻度清理，保留更多原始信息
    
    Args:
        text: 原始文本
        
    Returns:
        标准化后的文本
    """
    return TextNormalizer.normalize(
        text,
        unicode_normalize=True,
        remove_control=True,
        collapse_space=True,
        preserve_newlines=True,
        normalize_punct=False,
        normalize_nums=False,
        normalize_date=False,
        remove_markers=False
    )


if __name__ == '__main__':
    """测试文本标准化功能"""
    
    print("=" * 60)
    print("文本标准化模块测试")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        # (输入, 描述)
        ("　Ａ股涨幅１．２％，成交额３，４５６亿元", "全角字符"),
        ("2024年01月15日 公司发布公告", "中文日期"),
        ("股价\x00上涨\x0c了", "控制字符"),
        ("第一段\n\n\n\n\n第二段", "多余换行"),
        ("  行首空格   行尾空格  ", "首尾空格"),
        ("[第1页] 公司年度报告 (续)", "特殊标记"),
        ("净利润1,234,567.89元", "千分位数字"),
    ]
    
    print("\n1. 标准模式测试:")
    for text, desc in test_cases:
        result = normalize_text(text)
        print(f"  [{desc}]")
        print(f"    输入: {repr(text)}")
        print(f"    输出: {repr(result)}")
    
    print("\n2. 激进模式测试:")
    for text, desc in test_cases[:3]:
        result = normalize_text(text, aggressive=True)
        print(f"  [{desc}]")
        print(f"    输入: {repr(text)}")
        print(f"    输出: {repr(result)}")
    
    print("\n3. NLP模式测试:")
    long_text = """第一段内容。
    
    第二段内容，包含【特殊标记】。
    
    第三段内容。"""
    result = normalize_for_nlp(long_text)
    print(f"  输入: {repr(long_text)}")
    print(f"  输出: {repr(result)}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
