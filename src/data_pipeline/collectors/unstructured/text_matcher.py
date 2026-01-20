"""
文本相似度匹配工具

用于改进事件驱动模块的对齐算法
"""

from typing import List, Tuple, Optional
import re
from difflib import SequenceMatcher


class TextMatcher:
    """文本匹配器"""
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        计算Levenshtein距离（编辑距离）
        
        Args:
            s1: 字符串1
            s2: 字符串2
            
        Returns:
            编辑距离（越小越相似）
        """
        if len(s1) < len(s2):
            return TextMatcher.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # 插入、删除、替换的成本
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def levenshtein_similarity(s1: str, s2: str) -> float:
        """
        基于Levenshtein距离的相似度
        
        Returns:
            相似度 (0-1，越大越相似)
        """
        if not s1 or not s2:
            return 0.0
        
        distance = TextMatcher.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1 - (distance / max_len) if max_len > 0 else 0.0
    
    @staticmethod
    def jaccard_similarity(s1: str, s2: str, ngram: int = 2) -> float:
        """
        Jaccard相似度（基于n-gram）
        
        Args:
            s1: 字符串1
            s2: 字符串2
            ngram: n-gram大小
            
        Returns:
            相似度 (0-1)
        """
        def get_ngrams(text: str, n: int) -> set:
            """提取n-gram集合"""
            return set([text[i:i+n] for i in range(len(text) - n + 1)])
        
        if not s1 or not s2:
            return 0.0
        
        ngrams1 = get_ngrams(s1, ngram)
        ngrams2 = get_ngrams(s2, ngram)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2
        
        return len(intersection) / len(union) if union else 0.0
    
    @staticmethod
    def sequence_similarity(s1: str, s2: str) -> float:
        """
        序列相似度（使用Python内置SequenceMatcher）
        
        Returns:
            相似度 (0-1)
        """
        if not s1 or not s2:
            return 0.0
        
        return SequenceMatcher(None, s1, s2).ratio()
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        文本标准化
        
        - 转小写
        - 移除多余空白
        - 移除特殊符号
        """
        # 转小写
        text = text.lower()
        # 移除特殊符号（保留中文、英文、数字）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        # 移除多余空白
        text = ' '.join(text.split())
        return text
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """
        提取关键词（简单版：基于词频）
        
        Args:
            text: 文本
            top_n: 返回top N关键词
            
        Returns:
            关键词列表
        """
        # 简单分词（中文按字符，英文按空格）
        tokens = []
        
        # 提取中文词（2-4字）
        chinese_words = re.findall(r'[\u4e00-\u9fa5]{2,4}', text)
        tokens.extend(chinese_words)
        
        # 提取英文词
        english_words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        tokens.extend(english_words)
        
        # 统计词频
        from collections import Counter
        word_counts = Counter(tokens)
        
        # 过滤停用词（简化版）
        stopwords = {'关于', '公告', '通知', '公司', '股份', '有限', '的', '了', '在', '是'}
        filtered = [(w, c) for w, c in word_counts.items() if w not in stopwords]
        
        # 返回top N
        return [w for w, c in sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n]]
    
    @staticmethod
    def keyword_overlap_similarity(s1: str, s2: str, top_n: int = 10) -> float:
        """
        基于关键词重叠的相似度
        
        Returns:
            相似度 (0-1)
        """
        keywords1 = set(TextMatcher.extract_keywords(s1, top_n))
        keywords2 = set(TextMatcher.extract_keywords(s2, top_n))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1 & keywords2
        union = keywords1 | keywords2
        
        return len(intersection) / len(union) if union else 0.0
    
    @staticmethod
    def hybrid_similarity(
        s1: str,
        s2: str,
        weights: Optional[Tuple[float, float, float, float]] = None
    ) -> float:
        """
        混合相似度（综合多种算法）
        
        Args:
            s1: 字符串1
            s2: 字符串2
            weights: 权重 (levenshtein, jaccard, sequence, keyword)
            
        Returns:
            综合相似度 (0-1)
        """
        if weights is None:
            weights = (0.25, 0.25, 0.25, 0.25)  # 默认平均权重
        
        # 标准化文本
        s1_norm = TextMatcher.normalize_text(s1)
        s2_norm = TextMatcher.normalize_text(s2)
        
        # 计算各种相似度
        lev_sim = TextMatcher.levenshtein_similarity(s1_norm, s2_norm)
        jac_sim = TextMatcher.jaccard_similarity(s1_norm, s2_norm, ngram=2)
        seq_sim = TextMatcher.sequence_similarity(s1_norm, s2_norm)
        kw_sim = TextMatcher.keyword_overlap_similarity(s1, s2)  # 原始文本提取关键词
        
        # 加权平均
        score = (
            weights[0] * lev_sim +
            weights[1] * jac_sim +
            weights[2] * seq_sim +
            weights[3] * kw_sim
        )
        
        return score
    
    @staticmethod
    def find_best_match(
        query: str,
        candidates: List[str],
        threshold: float = 0.5,
        method: str = 'hybrid'
    ) -> Tuple[Optional[int], float]:
        """
        从候选列表中找到最佳匹配
        
        Args:
            query: 查询字符串
            candidates: 候选字符串列表
            threshold: 最低相似度阈值
            method: 匹配方法 ('levenshtein', 'jaccard', 'sequence', 'keyword', 'hybrid')
            
        Returns:
            (最佳匹配的索引, 相似度分数)，无匹配时返回 (None, 0.0)
        """
        if not candidates:
            return None, 0.0
        
        # 选择匹配方法
        similarity_func = {
            'levenshtein': TextMatcher.levenshtein_similarity,
            'jaccard': TextMatcher.jaccard_similarity,
            'sequence': TextMatcher.sequence_similarity,
            'keyword': TextMatcher.keyword_overlap_similarity,
            'hybrid': TextMatcher.hybrid_similarity
        }.get(method, TextMatcher.hybrid_similarity)
        
        # 计算所有候选的相似度
        scores = []
        for candidate in candidates:
            if method == 'jaccard':
                score = similarity_func(query, candidate, ngram=2)
            else:
                score = similarity_func(query, candidate)
            scores.append(score)
        
        # 找到最高分
        max_score = max(scores)
        max_idx = scores.index(max_score)
        
        # 检查阈值
        if max_score < threshold:
            return None, max_score
        
        return max_idx, max_score


def align_with_similarity(
    df_source: 'pd.DataFrame',
    df_target: 'pd.DataFrame',
    source_text_col: str = 'title',
    target_text_col: str = 'title',
    threshold: float = 0.6,
    method: str = 'hybrid'
) -> 'pd.DataFrame':
    """
    使用文本相似度对齐两个DataFrame
    
    Args:
        df_source: 源DataFrame（如东财数据）
        df_target: 目标DataFrame（如巨潮数据）
        source_text_col: 源文本列名
        target_text_col: 目标文本列名
        threshold: 相似度阈值
        method: 匹配方法
        
    Returns:
        对齐后的DataFrame（包含两边的列）
    """
    import pandas as pd
    
    aligned_rows = []
    
    for idx_source, row_source in df_source.iterrows():
        query_text = str(row_source[source_text_col])
        candidates = df_target[target_text_col].tolist()
        
        match_idx, score = TextMatcher.find_best_match(
            query=query_text,
            candidates=candidates,
            threshold=threshold,
            method=method
        )
        
        if match_idx is not None:
            row_target = df_target.iloc[match_idx]
            # 合并两行数据
            merged_row = {
                **row_source.to_dict(),
                **{f"{k}_target": v for k, v in row_target.to_dict().items()},
                'match_score': score
            }
            aligned_rows.append(merged_row)
    
    return pd.DataFrame(aligned_rows)
