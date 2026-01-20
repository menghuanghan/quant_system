"""
非结构化数据清洗模块 (Unstructured Data Cleaner)

核心职责：
1. 格式标准化 - 将各种格式的原始数据转为统一的纯文本
2. 内容提取 - 从 PDF/HTML 中提取正文，去除噪音
3. 元数据清洗 - 时间戳标准化，防止"未来函数"泄露

设计原则：
- 即时提取（Extract-on-the-fly）：输入是二进制流/字符串，输出是清洗后的纯文本
- 不依赖磁盘 IO：完全内存操作，支持流式处理
- 防止数据泄露：时间戳采用保守策略填充（默认盘后17:00）
- 统一接口：采集器不需要知道底层实现细节

模块组成：
- text_utils.py: 通用文本标准化（Unicode、空白、控制字符、模板移除）
- pdf_parser.py: PDF 内存解析（PyMuPDF/pdfplumber、页眉页脚过滤、扫描件检测）
- html_parser.py: HTML 网页清洗（BeautifulSoup、正文密度算法）
- time_utils.py: 时间元数据标准化（防未来函数、精度检测）

使用示例：
    >>> from src.data_pipeline.clean.unstructured import (
    ...     extract_text_from_pdf_bytes,
    ...     extract_text_from_html,
    ...     normalize_text,
    ...     standardize_publish_time
    ... )
    
    >>> # 从 PDF 提取文本
    >>> pdf_bytes = requests.get(pdf_url).content
    >>> text = extract_text_from_pdf_bytes(pdf_bytes)
    
    >>> # 从 HTML 提取文本
    >>> html = requests.get(news_url).text
    >>> text = extract_text_from_html(html)
    
    >>> # 文本标准化
    >>> text = normalize_text(raw_text)
    
    >>> # 时间标准化（防止未来函数）
    >>> publish_time = standardize_publish_time('2024-01-15')  # → '2024-01-15 17:00:00'
"""

# ============================================================
# 统一导出接口（采集器直接调用这些函数）
# ============================================================

# 文本标准化
from .text_utils import (
    normalize_text,
    normalize_for_nlp,
    normalize_for_storage,
    normalize_for_announcement,  # 新增：公告专用
    TextNormalizer
)

# PDF 解析
from .pdf_parser import (
    extract_text_from_pdf_bytes,
    extract_text_from_pdf_file,
    is_scanned_pdf,
    PDFParser,
    ScannedPDFError,  # 新增：扫描件异常
    PDFBackend,  # 新增：解析器枚举
    get_pdf_parser
)

# HTML 解析
from .html_parser import (
    extract_text_from_html,
    extract_article_info,
    clean_html_tags,
    HTMLParser,
    HTMLCleanConfig,
    get_html_parser
)

# 时间元数据标准化（防止未来函数）
from .time_utils import (
    standardize_publish_time,
    extract_time_from_text,
    is_future_data,
    TimeNormalizer,
    TimeMode,
    TimeAccuracy
)


# ============================================================
# 高级封装函数（组合多个清洗步骤）
# ============================================================

def extract_and_clean_pdf(
    pdf_bytes: bytes,
    max_pages: int = None,
    aggressive: bool = False
) -> str:
    """
    从 PDF 提取并清洗文本（一站式函数）
    
    组合 PDF 提取 + 文本标准化的完整流程。
    
    Args:
        pdf_bytes: PDF 二进制内容
        max_pages: 最大提取页数
        aggressive: 是否使用激进清洗（日期、数字标准化）
        
    Returns:
        清洗后的纯文本
        
    Examples:
        >>> response = requests.get(pdf_url)
        >>> text = extract_and_clean_pdf(response.content)
    """
    # 1. 提取文本
    text = extract_text_from_pdf_bytes(
        pdf_bytes,
        max_pages=max_pages,
        normalize=False  # 不使用内置标准化，使用我们的
    )
    
    if not text:
        return ""
    
    # 2. 应用标准化
    if aggressive:
        return normalize_for_nlp(text)
    else:
        return normalize_for_storage(text)


def extract_and_clean_html(
    html: str,
    site_type: str = None,
    aggressive: bool = False
) -> str:
    """
    从 HTML 提取并清洗文本（一站式函数）
    
    组合 HTML 提取 + 文本标准化的完整流程。
    
    Args:
        html: HTML 源码
        site_type: 网站类型 ('xueqiu', 'eastmoney', 'guba')
        aggressive: 是否使用激进清洗
        
    Returns:
        清洗后的纯文本
        
    Examples:
        >>> html = requests.get(news_url).text
        >>> text = extract_and_clean_html(html, site_type='xueqiu')
    """
    # 1. 提取文本
    text = extract_text_from_html(
        html,
        site_type=site_type,
        normalize=False  # 不使用内置标准化
    )
    
    if not text:
        return ""
    
    # 2. 应用标准化
    if aggressive:
        return normalize_for_nlp(text)
    else:
        return normalize_for_storage(text)


def detect_content_type(content: bytes) -> str:
    """
    检测内容类型
    
    根据内容的魔术字节判断是 PDF 还是 HTML。
    
    Args:
        content: 二进制内容
        
    Returns:
        'pdf', 'html', 'text', 'unknown'
    """
    if not content:
        return 'unknown'
    
    # PDF 魔术字节
    if content[:5] == b'%PDF-':
        return 'pdf'
    
    # HTML 特征
    try:
        text = content[:500].decode('utf-8', errors='ignore').lower()
        if '<!doctype html' in text or '<html' in text:
            return 'html'
        if '<head' in text or '<body' in text:
            return 'html'
    except:
        pass
    
    # 尝试作为纯文本
    try:
        content.decode('utf-8')
        return 'text'
    except:
        pass
    
    return 'unknown'


def auto_extract_text(
    content: bytes,
    content_type: str = None,
    **kwargs
) -> str:
    """
    自动检测内容类型并提取文本
    
    Args:
        content: 二进制内容
        content_type: 内容类型（None 自动检测）
        **kwargs: 传递给具体提取函数的参数
        
    Returns:
        提取的纯文本
    """
    if not content:
        return ""
    
    if content_type is None:
        content_type = detect_content_type(content)
    
    if content_type == 'pdf':
        return extract_and_clean_pdf(content, **kwargs)
    elif content_type == 'html':
        try:
            html = content.decode('utf-8', errors='ignore')
        except:
            html = content.decode('gbk', errors='ignore')
        return extract_and_clean_html(html, **kwargs)
    elif content_type == 'text':
        try:
            text = content.decode('utf-8')
        except:
            text = content.decode('gbk', errors='ignore')
        return normalize_for_storage(text)
    else:
        return ""


# ============================================================
# 版本信息
# ============================================================

__version__ = '1.0.0'
__author__ = 'Quant System Team'

__all__ = [
    # 核心函数
    'normalize_text',
    'normalize_for_nlp',
    'normalize_for_storage',
    'extract_text_from_pdf_bytes',
    'extract_text_from_pdf_file',
    'extract_text_from_html',
    'extract_article_info',
    'clean_html_tags',
    'is_scanned_pdf',
    
    # 高级封装
    'extract_and_clean_pdf',
    'extract_and_clean_html',
    'detect_content_type',
    'auto_extract_text',
    
    # 类
    'TextNormalizer',
    'PDFParser',
    'HTMLParser',
    'HTMLCleanConfig',
    
    # 单例获取
    'get_pdf_parser',
    'get_html_parser',
]