"""
内容去重模块 (Content Deduplication)

基于 SimHash + MD5 指纹 + Bloom Filter 实现高效去重：
- MD5: 精确去重，用于完全相同的内容
- SimHash: 模糊去重，用于高度相似的内容（如通稿改写）
- Bloom Filter: 高效的存在性检测，节省内存

预估收益：减少 30%-40% 的重复新闻存储
"""

import hashlib
import logging
import pickle
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Tuple, List
from threading import Lock
import re

logger = logging.getLogger(__name__)


# ============== SimHash 实现 ==============

class SimHash:
    """
    SimHash 算法实现
    
    用于计算文本的局部敏感哈希，相似文本的 SimHash 值也相似。
    可用于检测"换汤不换药"的改写稿件。
    
    原理：
    1. 对文本分词
    2. 对每个词计算哈希，并根据词频加权
    3. 合并成一个固定长度的指纹
    
    Example:
        >>> sh = SimHash()
        >>> hash1 = sh.compute("今日股市大涨，沪指突破3000点")
        >>> hash2 = sh.compute("今日A股大涨，上证指数突破3000点")
        >>> distance = sh.hamming_distance(hash1, hash2)
        >>> print(f"海明距离: {distance}")  # 距离越小越相似
    """
    
    def __init__(self, hash_bits: int = 64):
        """
        Args:
            hash_bits: 哈希位数，越大越精确但占用更多空间
        """
        self.hash_bits = hash_bits
        # 中文分词的简单实现（按字符N-gram）
        self.ngram_size = 3
    
    def _tokenize(self, text: str) -> List[str]:
        """
        文本分词（简单实现，生产环境可用 jieba）
        
        使用字符级 N-gram 作为特征，避免分词依赖
        """
        # 移除标点和空白
        text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
        
        if len(text) < self.ngram_size:
            return [text] if text else []
        
        # 生成 N-gram
        tokens = []
        for i in range(len(text) - self.ngram_size + 1):
            tokens.append(text[i:i + self.ngram_size])
        
        return tokens
    
    def _hash_token(self, token: str) -> int:
        """计算单个 token 的哈希值"""
        return int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
    
    def compute(self, text: str) -> int:
        """
        计算文本的 SimHash 值
        
        Args:
            text: 输入文本
        
        Returns:
            64位整数形式的 SimHash
        """
        if not text:
            return 0
        
        tokens = self._tokenize(text)
        if not tokens:
            return 0
        
        # 初始化向量
        v = [0] * self.hash_bits
        
        # 计算加权向量
        for token in tokens:
            token_hash = self._hash_token(token)
            for i in range(self.hash_bits):
                bit = (token_hash >> i) & 1
                if bit:
                    v[i] += 1
                else:
                    v[i] -= 1
        
        # 生成最终哈希
        fingerprint = 0
        for i in range(self.hash_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        
        return fingerprint
    
    def hamming_distance(self, hash1: int, hash2: int) -> int:
        """
        计算两个 SimHash 的海明距离
        
        距离越小，文本越相似：
        - 0-3: 高度相似（可能是同一篇稿件的改写）
        - 4-10: 中度相似
        - >10: 不相似
        """
        xor = hash1 ^ hash2
        distance = 0
        while xor:
            distance += 1
            xor &= xor - 1
        return distance
    
    def is_similar(self, hash1: int, hash2: int, threshold: int = 3) -> bool:
        """判断两个哈希是否相似"""
        return self.hamming_distance(hash1, hash2) <= threshold


# ============== Bloom Filter 实现 ==============

class BloomFilter:
    """
    布隆过滤器实现
    
    高效的概率性数据结构，用于检测元素是否存在于集合中：
    - 如果返回 False，元素一定不存在
    - 如果返回 True，元素可能存在（有小概率误判）
    
    适用场景：快速过滤明显不重复的内容，减少精确比较的次数
    
    Example:
        >>> bf = BloomFilter(expected_items=1000000)
        >>> bf.add("some_hash_value")
        >>> bf.contains("some_hash_value")  # True
        >>> bf.contains("other_value")  # False
    """
    
    def __init__(
        self,
        expected_items: int = 1000000,
        false_positive_rate: float = 0.01
    ):
        """
        Args:
            expected_items: 预期存储的元素数量
            false_positive_rate: 期望的误报率（越小需要越多空间）
        """
        # 计算最优参数
        self.size = self._optimal_size(expected_items, false_positive_rate)
        self.hash_count = self._optimal_hash_count(self.size, expected_items)
        
        # 位数组（使用 bytearray 节省内存）
        self.bit_array = bytearray((self.size + 7) // 8)
        self.count = 0
        
        logger.debug(
            f"BloomFilter 初始化: size={self.size}, "
            f"hash_count={self.hash_count}, "
            f"memory={len(self.bit_array) / 1024:.1f}KB"
        )
    
    def _optimal_size(self, n: int, p: float) -> int:
        """计算最优位数组大小"""
        import math
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)
    
    def _optimal_hash_count(self, m: int, n: int) -> int:
        """计算最优哈希函数数量"""
        import math
        k = (m / n) * math.log(2)
        return max(1, int(k))
    
    def _get_hash_values(self, item: str) -> List[int]:
        """生成多个哈希值"""
        # 使用双哈希技术减少计算
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)
        
        return [(h1 + i * h2) % self.size for i in range(self.hash_count)]
    
    def _set_bit(self, index: int):
        """设置位"""
        byte_index = index // 8
        bit_index = index % 8
        self.bit_array[byte_index] |= (1 << bit_index)
    
    def _get_bit(self, index: int) -> bool:
        """获取位"""
        byte_index = index // 8
        bit_index = index % 8
        return bool(self.bit_array[byte_index] & (1 << bit_index))
    
    def add(self, item: str):
        """添加元素"""
        for index in self._get_hash_values(item):
            self._set_bit(index)
        self.count += 1
    
    def contains(self, item: str) -> bool:
        """检查元素是否可能存在"""
        return all(self._get_bit(index) for index in self._get_hash_values(item))
    
    def __contains__(self, item: str) -> bool:
        return self.contains(item)
    
    def __len__(self) -> int:
        return self.count


# ============== 去重管理器 ==============

@dataclass
class DeduplicationResult:
    """去重检查结果"""
    is_duplicate: bool           # 是否重复
    duplicate_type: Optional[str] = None  # 重复类型: 'exact', 'similar', None
    original_id: Optional[str] = None     # 原始记录ID（如果有）
    similarity_score: Optional[float] = None  # 相似度分数


class ContentDeduplicator:
    """
    内容去重管理器
    
    组合 MD5（精确）+ SimHash（模糊）+ BloomFilter（快速过滤）实现高效去重。
    
    工作流程：
    1. 计算 MD5 → BloomFilter 快速检查
    2. 如果可能存在 → 查询精确索引确认
    3. 如果不是精确重复 → 计算 SimHash 检查相似
    
    Example:
        >>> dedup = ContentDeduplicator()
        >>> 
        >>> # 检查是否重复
        >>> result = dedup.check("今日股市大涨...", source_id="news_001")
        >>> if result.is_duplicate:
        ...     print(f"重复类型: {result.duplicate_type}")
        ... else:
        ...     # 添加到索引
        ...     dedup.add("今日股市大涨...", source_id="news_001")
    """
    
    # 默认持久化路径
    DEFAULT_INDEX_DIR = Path("data/state/dedup_index")
    
    def __init__(
        self,
        index_dir: Optional[Path] = None,
        expected_items: int = 5000000,  # 500万条
        simhash_threshold: int = 3,     # SimHash 相似阈值
        auto_persist: bool = True,
        persist_interval: int = 1000    # 每 N 次添加持久化一次
    ):
        """
        Args:
            index_dir: 索引存储目录
            expected_items: 预期存储的元素数量
            simhash_threshold: SimHash 相似判定阈值
            auto_persist: 是否自动持久化
            persist_interval: 持久化间隔
        """
        self.index_dir = index_dir or self.DEFAULT_INDEX_DIR
        self.simhash_threshold = simhash_threshold
        self.auto_persist = auto_persist
        self.persist_interval = persist_interval
        
        # 初始化组件
        self.simhash = SimHash()
        self.bloom_filter = BloomFilter(expected_items=expected_items)
        
        # 精确 MD5 索引: {md5: source_id}
        self.md5_index: dict = {}
        
        # SimHash 索引: {simhash: [(source_id, md5), ...]}
        # 用于快速查找相似内容
        self.simhash_index: dict = {}
        
        # 统计
        self.stats = {
            'total_checked': 0,
            'exact_duplicates': 0,
            'similar_duplicates': 0,
            'unique': 0
        }
        
        self._lock = Lock()
        self._add_count = 0
        
        # 确保目录存在
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # 尝试加载已有索引
        self._load_index()
    
    def _compute_md5(self, text: str) -> str:
        """计算文本 MD5"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _normalize_text(self, text: str) -> str:
        """
        文本标准化（去重前的预处理）
        
        移除不影响语义的差异：
        - 多余空白
        - 来源标注（如"据XX报道"）
        - 时间戳
        """
        if not text:
            return ""
        
        # 移除常见来源前缀
        text = re.sub(r'^(据)?[新华社|央视|财联社|证券时报|中证报|上证报]\S{0,10}(报道|讯|消息)?[,，：:]?\s*', '', text)
        
        # 移除时间戳格式
        text = re.sub(r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?\s*\d{0,2}[时:：]?\d{0,2}[分]?\d{0,2}[秒]?', '', text)
        
        # 统一空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def check(
        self,
        text: str,
        source_id: Optional[str] = None,
        normalize: bool = True
    ) -> DeduplicationResult:
        """
        检查内容是否重复
        
        Args:
            text: 待检查的文本
            source_id: 来源标识（用于跟踪）
            normalize: 是否标准化文本
        
        Returns:
            DeduplicationResult 包含去重结果
        """
        if not text:
            return DeduplicationResult(is_duplicate=False)
        
        with self._lock:
            self.stats['total_checked'] += 1
            
            # 标准化
            if normalize:
                text = self._normalize_text(text)
            
            # 计算 MD5
            md5 = self._compute_md5(text)
            
            # Step 1: BloomFilter 快速检查
            if md5 not in self.bloom_filter:
                return DeduplicationResult(is_duplicate=False)
            
            # Step 2: 精确 MD5 匹配
            if md5 in self.md5_index:
                self.stats['exact_duplicates'] += 1
                return DeduplicationResult(
                    is_duplicate=True,
                    duplicate_type='exact',
                    original_id=self.md5_index[md5],
                    similarity_score=1.0
                )
            
            # Step 3: SimHash 相似检查
            text_simhash = self.simhash.compute(text)
            
            for stored_simhash, items in self.simhash_index.items():
                distance = self.simhash.hamming_distance(text_simhash, stored_simhash)
                if distance <= self.simhash_threshold:
                    self.stats['similar_duplicates'] += 1
                    # 返回第一个相似项
                    original_id = items[0][0] if items else None
                    similarity = 1.0 - (distance / 64.0)
                    return DeduplicationResult(
                        is_duplicate=True,
                        duplicate_type='similar',
                        original_id=original_id,
                        similarity_score=similarity
                    )
            
            return DeduplicationResult(is_duplicate=False)
    
    def add(
        self,
        text: str,
        source_id: str,
        normalize: bool = True
    ):
        """
        添加内容到去重索引
        
        Args:
            text: 文本内容
            source_id: 唯一标识符
            normalize: 是否标准化文本
        """
        if not text:
            return
        
        with self._lock:
            if normalize:
                text = self._normalize_text(text)
            
            md5 = self._compute_md5(text)
            text_simhash = self.simhash.compute(text)
            
            # 添加到 BloomFilter
            self.bloom_filter.add(md5)
            
            # 添加到 MD5 索引
            self.md5_index[md5] = source_id
            
            # 添加到 SimHash 索引
            if text_simhash not in self.simhash_index:
                self.simhash_index[text_simhash] = []
            self.simhash_index[text_simhash].append((source_id, md5))
            
            self.stats['unique'] += 1
            self._add_count += 1
            
            # 自动持久化
            if self.auto_persist and self._add_count % self.persist_interval == 0:
                self._save_index()
    
    def check_and_add(
        self,
        text: str,
        source_id: str,
        normalize: bool = True
    ) -> DeduplicationResult:
        """
        检查并添加（如果不重复）
        
        这是最常用的方法，组合了 check 和 add。
        """
        result = self.check(text, source_id, normalize)
        if not result.is_duplicate:
            self.add(text, source_id, normalize)
        return result
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            stats = self.stats.copy()
            stats['index_size'] = len(self.md5_index)
            stats['bloom_filter_count'] = len(self.bloom_filter)
            
            # 计算去重率
            total = stats['total_checked']
            if total > 0:
                stats['dedup_rate'] = (
                    (stats['exact_duplicates'] + stats['similar_duplicates']) 
                    / total * 100
                )
            else:
                stats['dedup_rate'] = 0.0
            
            return stats
    
    def _save_index(self):
        """保存索引到磁盘"""
        try:
            # 保存 MD5 索引
            md5_path = self.index_dir / "md5_index.pkl"
            with open(md5_path, 'wb') as f:
                pickle.dump(self.md5_index, f)
            
            # 保存 SimHash 索引
            simhash_path = self.index_dir / "simhash_index.pkl"
            with open(simhash_path, 'wb') as f:
                pickle.dump(self.simhash_index, f)
            
            # 保存 BloomFilter
            bloom_path = self.index_dir / "bloom_filter.pkl"
            with open(bloom_path, 'wb') as f:
                pickle.dump({
                    'bit_array': bytes(self.bloom_filter.bit_array),
                    'size': self.bloom_filter.size,
                    'hash_count': self.bloom_filter.hash_count,
                    'count': self.bloom_filter.count
                }, f)
            
            # 保存统计
            stats_path = self.index_dir / "stats.pkl"
            with open(stats_path, 'wb') as f:
                pickle.dump(self.stats, f)
            
            logger.debug(f"去重索引已保存: {len(self.md5_index)} 条记录")
            
        except Exception as e:
            logger.error(f"保存去重索引失败: {e}")
    
    def _load_index(self):
        """从磁盘加载索引"""
        try:
            # 加载 MD5 索引
            md5_path = self.index_dir / "md5_index.pkl"
            if md5_path.exists():
                with open(md5_path, 'rb') as f:
                    self.md5_index = pickle.load(f)
            
            # 加载 SimHash 索引
            simhash_path = self.index_dir / "simhash_index.pkl"
            if simhash_path.exists():
                with open(simhash_path, 'rb') as f:
                    self.simhash_index = pickle.load(f)
            
            # 加载 BloomFilter
            bloom_path = self.index_dir / "bloom_filter.pkl"
            if bloom_path.exists():
                with open(bloom_path, 'rb') as f:
                    data = pickle.load(f)
                    self.bloom_filter.bit_array = bytearray(data['bit_array'])
                    self.bloom_filter.size = data['size']
                    self.bloom_filter.hash_count = data['hash_count']
                    self.bloom_filter.count = data['count']
            
            # 加载统计
            stats_path = self.index_dir / "stats.pkl"
            if stats_path.exists():
                with open(stats_path, 'rb') as f:
                    self.stats = pickle.load(f)
            
            if self.md5_index:
                logger.info(f"已加载去重索引: {len(self.md5_index)} 条记录")
                
        except Exception as e:
            logger.warning(f"加载去重索引失败: {e}，使用空索引")
    
    def persist(self):
        """手动持久化"""
        with self._lock:
            self._save_index()
    
    def clear(self):
        """清空索引"""
        with self._lock:
            self.md5_index.clear()
            self.simhash_index.clear()
            self.bloom_filter = BloomFilter(expected_items=5000000)
            self.stats = {
                'total_checked': 0,
                'exact_duplicates': 0,
                'similar_duplicates': 0,
                'unique': 0
            }
            self._add_count = 0


# ============== 全局单例 ==============

_global_deduplicator: Optional[ContentDeduplicator] = None


def get_deduplicator() -> ContentDeduplicator:
    """获取全局去重器"""
    global _global_deduplicator
    if _global_deduplicator is None:
        _global_deduplicator = ContentDeduplicator()
    return _global_deduplicator


def check_duplicate(text: str, source_id: Optional[str] = None) -> DeduplicationResult:
    """便捷函数：检查是否重复"""
    return get_deduplicator().check(text, source_id)


def add_to_index(text: str, source_id: str):
    """便捷函数：添加到索引"""
    get_deduplicator().add(text, source_id)


def check_and_add(text: str, source_id: str) -> DeduplicationResult:
    """便捷函数：检查并添加"""
    return get_deduplicator().check_and_add(text, source_id)
