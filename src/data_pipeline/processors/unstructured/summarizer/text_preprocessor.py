"""
文本预处理模块

负责在发送给LLM之前对文本进行清洗和处理：
1. 去除冗余信息（页眉页脚、免责声明等）
2. 规范化格式（空白、标点等）
3. 长文本分块（超过模型上下文限制时）
4. 提取关键段落（智能截断）

支持GPU加速（cuDF）进行批量处理。
"""

import re
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .base import SummarizerConfig, TextCleaningMixin

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """文本块数据结构"""
    content: str
    start_pos: int          # 原文中的起始位置
    end_pos: int            # 原文中的结束位置
    chunk_index: int        # 块索引
    total_chunks: int       # 总块数
    
    @property
    def length(self) -> int:
        return len(self.content)


class TextPreprocessor(TextCleaningMixin):
    """
    文本预处理器
    
    在将content_text发送给LLM之前进行预处理，包括：
    1. 清洗：去除无关内容（页眉、免责声明等）
    2. 规范化：统一格式
    3. 分块：处理超长文本
    4. 截断：智能选择最相关的部分
    
    使用示例：
    ```python
    preprocessor = TextPreprocessor()
    
    # 简单清洗
    cleaned = preprocessor.clean(raw_text)
    
    # 清洗并分块
    chunks = preprocessor.preprocess(long_text)
    for chunk in chunks:
        print(f"块{chunk.chunk_index}: {chunk.length}字符")
    ```
    """
    
    # ==================== 正则模式 ====================
    
    # 免责声明模式
    DISCLAIMER_PATTERNS = [
        r'免责声明[：:].{0,500}',
        r'风险提示[：:].{0,300}',
        r'本报告.{0,100}(不构成|仅供).{0,200}',
        r'投资者应.{0,200}(自行判断|独立判断).{0,100}',
        r'本公司.{0,100}不对.{0,200}(承担|负责).{0,100}',
        r'以上信息.{0,100}(仅供参考|不作为).{0,100}',
    ]
    
    # 页眉页脚模式
    HEADER_FOOTER_PATTERNS = [
        r'^第\s*\d+\s*页.*?共\s*\d+\s*页',
        r'^\d+\s*/\s*\d+$',
        r'^-\s*\d+\s*-$',
        r'请务必阅读正文之后的免责条款',
        r'证券研究报告',
        r'[A-Z]+证券',
    ]
    
    # 公告套话模式
    BOILERPLATE_PATTERNS = [
        r'特此公告[。\s]*',
        r'敬请投资者注意投资风险[。\s]*',
        r'本公司董事会保证.{0,200}',
        r'根据《中华人民共和国公司法》.{0,200}',
        r'根据《上海证券交易所股票上市规则》.{0,200}',
        r'根据《深圳证券交易所股票上市规则》.{0,200}',
    ]
    
    # 联系方式模式
    CONTACT_PATTERNS = [
        r'联系方式[：:].{0,200}',
        r'联系电话[：:].{0,50}',
        r'传\s*真[：:].{0,50}',
        r'电子邮箱[：:].{0,100}',
        r'公司地址[：:].{0,200}',
    ]
    
    def __init__(self, config: Optional[SummarizerConfig] = None):
        """
        初始化预处理器
        
        Args:
            config: 配置对象
        """
        self.config = config or SummarizerConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 编译正则表达式
        self._compile_patterns()
    
    def _compile_patterns(self):
        """编译正则表达式"""
        self._disclaimer_re = [
            re.compile(p, re.IGNORECASE | re.DOTALL)
            for p in self.DISCLAIMER_PATTERNS
        ]
        self._header_footer_re = [
            re.compile(p, re.MULTILINE)
            for p in self.HEADER_FOOTER_PATTERNS
        ]
        self._boilerplate_re = [
            re.compile(p, re.IGNORECASE | re.DOTALL)
            for p in self.BOILERPLATE_PATTERNS
        ]
        self._contact_re = [
            re.compile(p, re.IGNORECASE)
            for p in self.CONTACT_PATTERNS
        ]
    
    def clean(self, text: str) -> str:
        """
        清洗文本
        
        移除冗余信息，规范化格式。
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清洗后的文本
        """
        if not text:
            return ""
        
        # 基础清洗
        text = self.clean_text_basic(text)
        
        # 移除页眉页脚
        text = self._remove_headers_footers(text)
        
        # 移除免责声明
        text = self._remove_disclaimers(text)
        
        # 移除套话
        text = self._remove_boilerplate(text)
        
        # 移除联系方式
        text = self._remove_contacts(text)
        
        # 移除多余空行
        text = self._normalize_whitespace(text)
        
        return text.strip()
    
    def _remove_headers_footers(self, text: str) -> str:
        """移除页眉页脚"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            is_header_footer = False
            for pattern in self._header_footer_re:
                if pattern.search(line.strip()):
                    is_header_footer = True
                    break
            if not is_header_footer:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _remove_disclaimers(self, text: str) -> str:
        """移除免责声明"""
        for pattern in self._disclaimer_re:
            text = pattern.sub('', text)
        return text
    
    def _remove_boilerplate(self, text: str) -> str:
        """移除公告套话"""
        for pattern in self._boilerplate_re:
            text = pattern.sub('', text)
        return text
    
    def _remove_contacts(self, text: str) -> str:
        """移除联系方式"""
        for pattern in self._contact_re:
            text = pattern.sub('', text)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白字符"""
        # 多个空行合并为一个
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # 多个空格合并为一个
        text = re.sub(r'[ \t]+', ' ', text)
        return text
    
    def preprocess(
        self,
        text: str,
        chunk_if_needed: bool = True
    ) -> List[TextChunk]:
        """
        预处理文本
        
        清洗后，如果文本超过最大长度则分块。
        
        Args:
            text: 原始文本
            chunk_if_needed: 是否在需要时分块
            
        Returns:
            List[TextChunk]: 文本块列表
        """
        # 先清洗
        cleaned = self.clean(text)
        
        if not cleaned:
            return []
        
        # 检查是否需要分块
        if len(cleaned) <= self.config.max_input_chars or not chunk_if_needed:
            return [TextChunk(
                content=cleaned,
                start_pos=0,
                end_pos=len(cleaned),
                chunk_index=0,
                total_chunks=1,
            )]
        
        # 需要分块
        return self._chunk_text(cleaned)
    
    def _chunk_text(self, text: str) -> List[TextChunk]:
        """
        将长文本分块
        
        使用滑动窗口策略，保持段落完整性。
        """
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        # 尝试按段落分割
        paragraphs = self._split_paragraphs(text)
        
        current_chunk = ""
        current_start = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 新块从当前段落开始，可能带一些重叠
                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + para + "\n\n"
                else:
                    current_chunk = para + "\n\n"
        
        # 最后一块
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 转换为TextChunk对象
        total = len(chunks)
        result = []
        pos = 0
        
        for i, chunk in enumerate(chunks):
            result.append(TextChunk(
                content=chunk,
                start_pos=pos,
                end_pos=pos + len(chunk),
                chunk_index=i,
                total_chunks=total,
            ))
            pos += len(chunk) - overlap if overlap > 0 else len(chunk)
        
        return result
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """按段落分割文本"""
        # 按双换行分割
        paragraphs = re.split(r'\n\s*\n', text)
        # 过滤空段落
        return [p.strip() for p in paragraphs if p.strip()]
    
    def truncate_smart(
        self,
        text: str,
        max_chars: int,
        keep_start: float = 0.7,
        keep_end: float = 0.3
    ) -> str:
        """
        智能截断文本
        
        保留开头和结尾的重要部分，中间用省略号连接。
        
        Args:
            text: 原始文本
            max_chars: 最大字符数
            keep_start: 保留开头的比例
            keep_end: 保留结尾的比例
            
        Returns:
            str: 截断后的文本
        """
        if len(text) <= max_chars:
            return text
        
        # 预留省略号空间
        available = max_chars - 10
        start_len = int(available * keep_start)
        end_len = int(available * keep_end)
        
        start_text = text[:start_len]
        end_text = text[-end_len:] if end_len > 0 else ""
        
        return f"{start_text}\n\n... [省略中间部分] ...\n\n{end_text}"
    
    def extract_key_sentences(
        self,
        text: str,
        max_sentences: int = 10
    ) -> str:
        """
        提取关键句子
        
        使用简单的启发式规则提取可能重要的句子：
        - 包含数字的句子
        - 包含关键词的句子
        - 段落首句
        
        Args:
            text: 原始文本
            max_sentences: 最大句子数
            
        Returns:
            str: 提取的关键句子
        """
        # 分句
        sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return text[:self.config.max_input_chars]
        
        # 评分
        scored = []
        for sent in sentences:
            score = 0
            # 包含数字（可能是业绩数据）
            if re.search(r'\d+\.?\d*[%亿万元]', sent):
                score += 3
            # 包含关键词
            if re.search(r'(增长|下降|同比|环比|预计|业绩|利润|营收)', sent):
                score += 2
            # 包含金额
            if re.search(r'\d+\.?\d*亿|\d+\.?\d*万元', sent):
                score += 2
            # 句子长度适中
            if 20 <= len(sent) <= 100:
                score += 1
            
            scored.append((sent, score))
        
        # 按分数排序，取top
        scored.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored[:max_sentences]]
        
        # 按原文顺序排列
        ordered = []
        for sent in sentences:
            if sent in top_sentences:
                ordered.append(sent)
        
        return '。'.join(ordered) + '。'
    
    def preprocess_batch(
        self,
        texts: List[str],
        use_gpu: bool = True
    ) -> List[str]:
        """
        批量预处理文本
        
        Args:
            texts: 文本列表
            use_gpu: 是否使用GPU加速（cuDF）
            
        Returns:
            List[str]: 处理后的文本列表
        """
        if use_gpu and self._check_cudf() and len(texts) >= 10:
            return self._preprocess_batch_gpu(texts)
        
        return [self.clean(t) for t in texts]
    
    def _preprocess_batch_gpu(self, texts: List[str]) -> List[str]:
        """GPU加速的批量预处理"""
        try:
            import cudf
            
            # 先用cuDF做基础清洗
            gs = cudf.Series(texts)
            gs = gs.str.normalize_spaces()
            gs = gs.str.strip()
            
            # 转回CPU做详细清洗
            cpu_texts = gs.to_pandas().tolist()
            
            # 详细清洗（正则等cuDF不支持的操作）
            return [self._clean_detailed(t) for t in cpu_texts]
            
        except Exception as e:
            self.logger.warning(f"GPU批量预处理失败，回退到CPU: {e}")
            return [self.clean(t) for t in texts]
    
    def _clean_detailed(self, text: str) -> str:
        """详细清洗（CPU）"""
        if not text:
            return ""
        
        # 移除免责声明
        for pattern in self._disclaimer_re:
            text = pattern.sub('', text)
        
        # 移除套话
        for pattern in self._boilerplate_re:
            text = pattern.sub('', text)
        
        # 规范化空白
        text = self._normalize_whitespace(text)
        
        return text.strip()


class FinancialTextExtractor:
    """
    金融文本关键信息提取器
    
    专门针对金融文本，提取最相关的内容段落。
    """
    
    # 关键信息标题模式
    KEY_SECTION_PATTERNS = [
        r'(一|二|三|四|五|六|七|八|九|十)[、.]\s*(主要财务数据|经营情况|业绩预告|投资建议|风险提示|核心观点)',
        r'(摘要|概要|要点|核心结论|投资逻辑)',
        r'\d+[、.]\s*(业绩|营收|利润|增长|展望)',
    ]
    
    def __init__(self):
        self._section_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.KEY_SECTION_PATTERNS
        ]
    
    def extract_key_sections(self, text: str) -> str:
        """
        提取关键章节
        
        识别并提取公告/研报中的关键章节。
        """
        lines = text.split('\n')
        result = []
        in_key_section = False
        section_lines = []
        
        for line in lines:
            # 检查是否是关键章节标题
            is_key_title = any(p.search(line) for p in self._section_patterns)
            
            if is_key_title:
                # 保存之前的关键章节
                if section_lines:
                    result.extend(section_lines)
                    section_lines = []
                in_key_section = True
                section_lines.append(line)
            elif in_key_section:
                # 检查是否到了新章节（非关键）
                if re.match(r'^[一二三四五六七八九十][、.]', line):
                    in_key_section = False
                    result.extend(section_lines)
                    section_lines = []
                else:
                    section_lines.append(line)
        
        # 最后的章节
        if section_lines:
            result.extend(section_lines)
        
        return '\n'.join(result) if result else text
