"""
PDF 内存解析模块 (PDF Parser)

核心职责：接收内存中的 PDF 二进制流（Bytes），输出纯文本。

主要功能：
1. 内存流包装 - 使用 BytesIO 避免磁盘 IO
2. 文本流提取 - 优先使用 PyMuPDF (10-20x 速度)，pdfplumber 作为备选
3. 页眉页脚去除 - 基于垂直坐标自动过滤（默认 8% top/bottom）
4. 异常处理 - 捕获加密/损坏 PDF 异常
5. 扫描件检测 - 文本密度检查（文件 > 500KB 但字符 < 50）

解析器选择：
- 'pymupdf' (默认): PyMuPDF/fitz，速度极快（10-20x），适合纯文本提取
- 'pdfplumber': 适合复杂表格场景（但速度慢）
- 'auto': 自动降级（PyMuPDF → pdfplumber）

设计原则：即时提取（Extract-on-the-fly），不依赖磁盘存储
"""

import io
import re
import logging
from typing import Optional, List, Tuple, Union, Literal
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class PDFBackend(str, Enum):
    """PDF 解析器后端枚举"""
    PYMUPDF = 'pymupdf'      # PyMuPDF (fitz) - 快速文本提取
    PDFPLUMBER = 'pdfplumber'  # pdfplumber - 表格场景
    AUTO = 'auto'            # 自动降级


class ScannedPDFError(Exception):
    """扫描件 PDF 异常（需要 OCR 处理）"""
    pass


class PDFParser:
    """
    PDF 内存解析器
    
    支持从二进制流或文件路径提取文本，
    具备页眉页脚过滤和表格处理能力。
    """
    
    # 默认页眉页脚区域占比（页面高度的百分比）
    DEFAULT_HEADER_RATIO = 0.08  # 顶部 8%
    DEFAULT_FOOTER_RATIO = 0.08  # 底部 8%
    
    # 常见页眉页脚关键词（用于辅助判断）
    HEADER_FOOTER_KEYWORDS = [
        r'第\s*\d+\s*页',           # 第X页
        r'共\s*\d+\s*页',           # 共X页
        r'Page\s*\d+',              # Page X
        r'\d+\s*/\s*\d+',           # X/Y
        r'^\d{4}年.*报告$',         # 2024年年度报告
        r'^\d{4}年度报告$',
        r'^年度报告$',
        r'^季度报告$',
        r'^半年度报告$',
    ]
    
    def __init__(
        self,
        backend: str = 'pymupdf',
        header_ratio: float = DEFAULT_HEADER_RATIO,
        footer_ratio: float = DEFAULT_FOOTER_RATIO,
        remove_header_footer: bool = True,
        check_scanned: bool = True,
        scanned_threshold_kb: int = 500,
        scanned_min_chars: int = 50
    ):
        """
        初始化 PDF 解析器
        
        Args:
            backend: 解析器后端 ('pymupdf', 'pdfplumber', 'auto')
            header_ratio: 页眉区域占页面高度的比例
            footer_ratio: 页脚区域占页面高度的比例
            remove_header_footer: 是否移除页眉页脚
            check_scanned: 是否检测扫描件 PDF
            scanned_threshold_kb: 扫描件检测的文件大小阈值（KB）
            scanned_min_chars: 扫描件检测的最小字符数阈值
        """
        self.backend = backend
        self.header_ratio = header_ratio
        self.footer_ratio = footer_ratio
        self.remove_header_footer = remove_header_footer
        self.check_scanned = check_scanned
        self.scanned_threshold_kb = scanned_threshold_kb
        self.scanned_min_chars = scanned_min_chars
        
        # 编译正则
        self._header_footer_patterns = [
            re.compile(pattern) for pattern in self.HEADER_FOOTER_KEYWORDS
        ]
    
    def extract_from_bytes(
        self,
        pdf_bytes: bytes,
        max_pages: Optional[int] = None,
        normalize: bool = True
    ) -> str:
        """
        从 PDF 二进制数据提取文本（核心方法）
        
        Args:
            pdf_bytes: PDF 文件的二进制内容
            max_pages: 最大提取页数（None 表示全部）
            normalize: 是否进行文本标准化
            
        Returns:
            提取的纯文本
            
        Raises:
            ScannedPDFError: 检测到扫描件 PDF（需 OCR 处理）
            
        Examples:
            >>> with open('report.pdf', 'rb') as f:
            ...     pdf_bytes = f.read()
            >>> text = parser.extract_from_bytes(pdf_bytes)
        """
        if not pdf_bytes:
            logger.warning("PDF 内容为空")
            return ""
        
        # 根据 backend 选择解析方法
        if self.backend == 'pymupdf':
            text = self._extract_with_pymupdf(pdf_bytes, max_pages)
        elif self.backend == 'pdfplumber':
            text = self._extract_with_pdfplumber(pdf_bytes, max_pages)
        elif self.backend == 'auto':
            # 自动降级：PyMuPDF → pdfplumber
            text = self._extract_with_pymupdf(pdf_bytes, max_pages)
            if not text:
                logger.info("PyMuPDF 提取失败，降级到 pdfplumber")
                text = self._extract_with_pdfplumber(pdf_bytes, max_pages)
        else:
            logger.error(f"不支持的解析器后端: {self.backend}")
            return ""
        
        # 扫描件检测（文本密度检查）
        if self.check_scanned and text is not None:
            file_size_kb = len(pdf_bytes) / 1024
            text_length = len(text.strip())
            
            # 判断：文件 > 500KB 但提取字符 < 50 → 扫描件
            if file_size_kb > self.scanned_threshold_kb and text_length < self.scanned_min_chars:
                error_msg = (
                    f"检测到扫描件 PDF: 文件大小 {file_size_kb:.1f} KB，"
                    f"但仅提取 {text_length} 个字符。需要 OCR 处理。"
                )
                logger.warning(error_msg)
                raise ScannedPDFError(error_msg)
        
        # 可选：应用文本标准化
        if normalize and text:
            from .text_utils import normalize_for_storage
            text = normalize_for_storage(text)
        
        return text
    
    def extract_from_file(
        self,
        file_path: Union[str, Path],
        max_pages: Optional[int] = None,
        normalize: bool = True
    ) -> str:
        """
        从 PDF 文件路径提取文本
        
        Args:
            file_path: PDF 文件路径
            max_pages: 最大提取页数
            normalize: 是否进行文本标准化
            
        Returns:
            提取的纯文本
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"PDF 文件不存在: {file_path}")
            return ""
        
        try:
            with open(file_path, 'rb') as f:
                return self.extract_from_bytes(f.read(), max_pages, normalize)
        except Exception as e:
            logger.warning(f"读取 PDF 文件失败: {file_path}, {e}")
            return ""
    
    def _extract_with_pymupdf(
        self,
        pdf_bytes: bytes,
        max_pages: Optional[int] = None
    ) -> str:
        """
        使用 PyMuPDF (fitz) 提取文本（速度快 10-20x）
        
        Args:
            pdf_bytes: PDF 二进制数据
            max_pages: 最大提取页数
            
        Returns:
            提取的纯文本
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF 未安装，请执行: pip install pymupdf")
            return ""
        
        try:
            # 从内存流打开 PDF
            pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            all_text = []
            total_pages = len(pdf)
            pages_to_process = min(total_pages, max_pages) if max_pages else total_pages
            
            for page_num in range(pages_to_process):
                page = pdf[page_num]
                
                if self.remove_header_footer:
                    # 获取页面尺寸
                    rect = page.rect
                    page_height = rect.height
                    
                    # 计算裁剪区域（排除页眉页脚）
                    header_y = page_height * self.header_ratio
                    footer_y = page_height * (1 - self.footer_ratio)
                    
                    # 裁剪矩形（left, top, right, bottom）
                    clip_rect = fitz.Rect(0, header_y, rect.width, footer_y)
                    
                    # 提取裁剪区域的文本
                    page_text = page.get_text(clip=clip_rect)
                else:
                    # 提取整页文本
                    page_text = page.get_text()
                
                # 额外过滤页眉页脚关键词
                page_text = self._filter_header_footer_text(page_text)
                
                if page_text and page_text.strip():
                    all_text.append(page_text)
            
            pdf.close()
            
            return '\n\n'.join(all_text)
            
        except Exception as e:
            logger.warning(f"PyMuPDF 解析失败: {type(e).__name__}: {e}")
            return ""
    
    def _extract_with_pdfplumber(
        self,
        pdf_bytes: bytes,
        max_pages: Optional[int] = None
    ) -> str:
        """
        使用 pdfplumber 提取文本（适合表格场景）
        
        Args:
            pdf_bytes: PDF 二进制数据
            max_pages: 最大提取页数
            
        Returns:
            提取的纯文本
        """
        try:
            import pdfplumber
        except ImportError:
            logger.error("pdfplumber 未安装，请执行: pip install pdfplumber")
            return ""
        
        # 使用 BytesIO 包装二进制数据，避免磁盘 IO
        pdf_stream = io.BytesIO(pdf_bytes)
        
        try:
            with pdfplumber.open(pdf_stream) as pdf:
                pages_to_process = pdf.pages
                if max_pages:
                    pages_to_process = pages_to_process[:max_pages]
                
                all_text = []
                
                for page_num, page in enumerate(pages_to_process, 1):
                    try:
                        page_text = self._extract_page_text_pdfplumber(page)
                        if page_text:
                            all_text.append(page_text)
                    except Exception as e:
                        logger.debug(f"第 {page_num} 页提取失败: {e}")
                        continue
                
                return '\n\n'.join(all_text)
                
        except Exception as e:
            logger.warning(f"pdfplumber 解析失败: {type(e).__name__}: {e}")
            return ""
    
    def _extract_page_text_pdfplumber(self, page) -> str:
        """
        使用 pdfplumber 提取单页文本，支持页眉页脚过滤
        
        Args:
            page: pdfplumber.Page 对象
            
        Returns:
            页面文本
        """
        if not self.remove_header_footer:
            # 不过滤页眉页脚，直接提取
            return page.extract_text() or ""
        
        # 获取页面尺寸
        page_height = page.height
        page_width = page.width
        
        # 计算有效区域（排除页眉页脚）
        header_y = page_height * self.header_ratio
        footer_y = page_height * (1 - self.footer_ratio)
        
        # 裁剪页面，只保留正文区域
        # pdfplumber 坐标系：左上角为原点，y 向下增加
        cropped_page = page.within_bbox((0, header_y, page_width, footer_y))
        
        text = cropped_page.extract_text() or ""
        
        # 额外过滤：基于关键词移除残留的页眉页脚文本
        text = self._filter_header_footer_text(text)
        
        return text
    
    def _filter_header_footer_text(self, text: str) -> str:
        """
        基于关键词过滤页眉页脚残留文本
        
        Args:
            text: 页面文本
            
        Returns:
            过滤后的文本
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # 跳过空行
            if not line_stripped:
                filtered_lines.append(line)
                continue
            
            # 检查是否匹配页眉页脚关键词
            is_header_footer = False
            for pattern in self._header_footer_patterns:
                if pattern.search(line_stripped):
                    is_header_footer = True
                    break
            
            # 短行（少于10个字符）且是纯数字，可能是页码
            if not is_header_footer and len(line_stripped) < 10:
                if line_stripped.isdigit():
                    is_header_footer = True
            
            if not is_header_footer:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def get_metadata(self, pdf_bytes: bytes) -> dict:
        """
        获取 PDF 元数据
        
        Args:
            pdf_bytes: PDF 二进制内容
            
        Returns:
            包含页数、标题等信息的字典
        """
        if not pdf_bytes:
            return {}
        
        try:
            import pdfplumber
        except ImportError:
            return {}
        
        pdf_stream = io.BytesIO(pdf_bytes)
        
        try:
            with pdfplumber.open(pdf_stream) as pdf:
                metadata = {
                    'page_count': len(pdf.pages),
                    'metadata': pdf.metadata or {}
                }
                
                # 尝试获取第一页尺寸
                if pdf.pages:
                    first_page = pdf.pages[0]
                    metadata['page_size'] = {
                        'width': first_page.width,
                        'height': first_page.height
                    }
                
                return metadata
                
        except Exception as e:
            logger.warning(f"获取 PDF 元数据失败: {e}")
            return {}
    
    def is_scanned_pdf(self, pdf_bytes: bytes, sample_pages: int = 3) -> bool:
        """
        检测 PDF 是否为扫描件（图片型 PDF）
        
        扫描件的特征：文本提取结果极少或为空
        
        Args:
            pdf_bytes: PDF 二进制内容
            sample_pages: 采样检测的页数
            
        Returns:
            True 表示可能是扫描件
        """
        if not pdf_bytes:
            return False
        
        # 提取前几页文本
        text = self.extract_from_bytes(pdf_bytes, max_pages=sample_pages, normalize=False)
        
        # 统计有效字符数（排除空白）
        text_length = len(text.replace(' ', '').replace('\n', ''))
        
        # 经验值：如果前3页的文本少于100个字符，很可能是扫描件
        threshold = 100 * sample_pages / 3
        
        return text_length < threshold


class PDFExtractorFactory:
    """
    PDF 提取器工厂
    
    提供不同解析引擎的选择
    """
    
    @staticmethod
    def create(engine: str = 'pdfplumber', **kwargs) -> PDFParser:
        """
        创建 PDF 解析器
        
        Args:
            engine: 解析引擎名称 ('pdfplumber', 'pymupdf', 'pypdf2')
            **kwargs: 传递给解析器的参数
            
        Returns:
            PDFParser 实例
        """
        # 目前只实现 pdfplumber，后续可扩展
        if engine == 'pdfplumber':
            return PDFParser(**kwargs)
        else:
            logger.warning(f"不支持的引擎 {engine}，使用默认 pdfplumber")
            return PDFParser(**kwargs)


# 全局单例（使用 PyMuPDF 作为默认）
_default_parser: Optional[PDFParser] = None


def get_pdf_parser(backend: str = 'pymupdf') -> PDFParser:
    """
    获取全局 PDF 解析器单例
    
    Args:
        backend: 解析器后端 ('pymupdf', 'pdfplumber', 'auto')
    """
    global _default_parser
    if _default_parser is None or _default_parser.backend != backend:
        _default_parser = PDFParser(backend=backend)
    return _default_parser


# 便捷函数
def extract_text_from_pdf_bytes(
    pdf_bytes: bytes,
    max_pages: Optional[int] = None,
    normalize: bool = True,
    remove_header_footer: bool = True,
    backend: str = 'pymupdf',
    check_scanned: bool = True
) -> str:
    """
    从 PDF 二进制数据提取文本（便捷函数）
    
    这是采集器调用的主要接口，不需要知道底层实现。
    
    Args:
        pdf_bytes: PDF 文件的二进制内容
        max_pages: 最大提取页数
        normalize: 是否进行文本标准化
        remove_header_footer: 是否移除页眉页脚
        backend: 解析器后端 ('pymupdf', 'pdfplumber', 'auto')
        check_scanned: 是否检测扫描件
        
    Returns:
        提取的纯文本
        
    Raises:
        ScannedPDFError: 检测到扫描件 PDF（需 OCR 处理）
        
    Examples:
        >>> response = requests.get(pdf_url)
        >>> text = extract_text_from_pdf_bytes(response.content)
    """
    parser = PDFParser(
        backend=backend,
        remove_header_footer=remove_header_footer,
        check_scanned=check_scanned
    )
    return parser.extract_from_bytes(pdf_bytes, max_pages, normalize)


def extract_text_from_pdf_file(
    file_path: Union[str, Path],
    max_pages: Optional[int] = None,
    normalize: bool = True
) -> str:
    """
    从 PDF 文件路径提取文本（便捷函数）
    
    Args:
        file_path: PDF 文件路径
        max_pages: 最大提取页数
        normalize: 是否进行文本标准化
        
    Returns:
        提取的纯文本
    """
    return get_pdf_parser().extract_from_file(file_path, max_pages, normalize)


def is_scanned_pdf(pdf_bytes: bytes) -> bool:
    """
    检测 PDF 是否为扫描件
    
    Args:
        pdf_bytes: PDF 二进制内容
        
    Returns:
        True 表示可能是扫描件，需要 OCR 处理
    """
    return get_pdf_parser().is_scanned_pdf(pdf_bytes)


if __name__ == '__main__':
    """测试 PDF 解析功能"""
    import sys
    
    print("=" * 60)
    print("PDF 内存解析模块测试")
    print("=" * 60)
    
    # 检查 pdfplumber 是否安装
    try:
        import pdfplumber
        print(f"✓ pdfplumber 版本: {pdfplumber.__version__}")
    except ImportError:
        print("✗ pdfplumber 未安装，请执行: pip install pdfplumber")
        sys.exit(1)
    
    # 测试解析器初始化
    parser = PDFParser()
    print(f"✓ PDFParser 初始化成功")
    print(f"  - 页眉区域: {parser.header_ratio * 100}%")
    print(f"  - 页脚区域: {parser.footer_ratio * 100}%")
    
    # 测试空数据处理
    result = extract_text_from_pdf_bytes(b'')
    assert result == "", "空数据应返回空字符串"
    print("✓ 空数据处理正常")
    
    # 测试无效 PDF 处理
    result = extract_text_from_pdf_bytes(b'not a pdf')
    assert result == "", "无效 PDF 应返回空字符串"
    print("✓ 无效 PDF 处理正常")
    
    # 测试真实 PDF（如果有的话）
    test_pdf_path = Path("data/raw/unstructured/events/penalty")
    if test_pdf_path.exists():
        pdf_files = list(test_pdf_path.rglob("*.pdf"))
        if pdf_files:
            pdf_file = pdf_files[0]
            print(f"\n测试真实 PDF: {pdf_file}")
            
            with open(pdf_file, 'rb') as f:
                pdf_bytes = f.read()
            
            # 获取元数据
            metadata = parser.get_metadata(pdf_bytes)
            print(f"  - 页数: {metadata.get('page_count', 'N/A')}")
            
            # 检测是否扫描件
            scanned = is_scanned_pdf(pdf_bytes)
            print(f"  - 扫描件: {scanned}")
            
            # 提取文本
            text = extract_text_from_pdf_bytes(pdf_bytes, max_pages=2)
            print(f"  - 提取文本长度: {len(text)} 字符")
            if text:
                print(f"  - 前100字符: {text[:100]}...")
    else:
        print("\n暂无测试 PDF 文件")
    
    print("\n" + "=" * 60)
    print("测试完成!")
