"""
国务院政策采集器（搜索API版）

采集范围：
- 国务院政策文件

数据源: sousuo.www.gov.cn 搜索API（返回结构化JSON数据）
"""

import time
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from .base_policy import (
    BasePolicyCollector,
    PolicyDocument,
    PolicySource,
)
from ..request_utils import safe_request

logger = logging.getLogger(__name__)


class GovCouncilCollector(BasePolicyCollector):
    """国务院政策采集器（使用搜索API）"""
    
    SOURCE = PolicySource.GOV
    BASE_URL = "https://www.gov.cn"
    SEARCH_API = "https://sousuo.www.gov.cn/search-gov/data"
    
    # 每页数据量（API最大支持n=50）
    PAGE_SIZE = 50
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        max_results: int = 2000,
        **kwargs
    ) -> 'pd.DataFrame':
        """
        采集国务院政策
        
        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            max_results: 最大采集数量
        """
        import pandas as pd
        
        # 标准化日期格式
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        # 搜索API需要 YYYY-MM-DD 格式
        min_time = start_dt.strftime('%Y-%m-%d')
        max_time = end_dt.strftime('%Y-%m-%d')
        
        logger.info(f"采集国务院政策文件 ({min_time} 至 {max_time})，使用搜索API...")
        
        all_documents = []
        seen_ids = set()
        page = 1  # API页码从1开始（p=0和p=1都返回第一页）
        
        while len(all_documents) < max_results:
            # 调用搜索API
            items, total_count = self._search_api(
                min_time=min_time,
                max_time=max_time,
                page=page,
                page_size=self.PAGE_SIZE,
            )
            
            if not items:
                logger.debug(f"第 {page} 页无数据，停止翻页")
                break
            
            if page == 0:
                logger.info(f"搜索API返回: 共 {total_count} 条政策文件")
            
            for item in items:
                if len(all_documents) >= max_results:
                    break
                
                doc = self._parse_search_item(item)
                if doc and doc.id not in seen_ids:
                    all_documents.append(doc)
                    seen_ids.add(doc.id)
            
            # 检查是否还有更多数据
            # API分页: page从1开始, 每页page_size条
            fetched_so_far = page * self.PAGE_SIZE
            if fetched_so_far >= total_count:
                break
            
            page += 1
            time.sleep(0.3)
        
        logger.info(f"国务院政策采集完成，共 {len(all_documents)} 条")
        return self.to_dataframe(all_documents)
    
    def _search_api(
        self,
        min_time: str,
        max_time: str,
        page: int = 0,
        page_size: int = 20,
    ) -> tuple:
        """
        调用国务院政策文件库搜索API
        
        Args:
            min_time: 最早时间 (YYYY-MM-DD)
            max_time: 最晚时间 (YYYY-MM-DD)
            page: 页码（从0开始）
            page_size: 每页数量
            
        Returns:
            (items_list, total_count) 元组
        """
        params = {
            't': 'zhengcelibrary_gw',
            'q': '',
            'timetype': 'timezd',  # timezd=指定时间范围过滤, timeqb=全部时间
            'mintime': min_time,
            'maxtime': max_time,
            'sort': 'pubtime',
            'sortType': 1,
            'searchfield': 'title',
            'pcodeJig498': '',
            'childtype': '',
            'subchildtype': '',
            'tsbq': '',
            'puborg': '',
            'puborgsearchinfo': '',
            'searchType': 0,
            'pcodeYear': '',
            'pcodeNum': '',
            'filetype': '',
            'p': page,
            'n': page_size,
            'inpro': '',
        }
        
        try:
            response = safe_request(
                self.SEARCH_API,
                method='GET',
                params=params,
                timeout=15,
            )
            
            if not response or response.status_code != 200:
                logger.warning(f"搜索API请求失败: status={getattr(response, 'status_code', 'N/A')}")
                return [], 0
            
            data = response.json()
            
            if data.get('code') != 200:
                logger.warning(f"搜索API返回错误: {data.get('msg', 'unknown')}")
                return [], 0
            
            search_vo = data.get('searchVO', {})
            total_count = search_vo.get('totalCount', 0)
            list_vo = search_vo.get('listVO', [])
            
            logger.debug(f"搜索API第 {page} 页: 获取 {len(list_vo)} 条，共 {total_count} 条")
            
            return list_vo, total_count
            
        except Exception as e:
            logger.warning(f"搜索API调用失败: {e}")
            return [], 0
    
    def _parse_search_item(self, item: Dict[str, Any]) -> Optional[PolicyDocument]:
        """
        解析搜索API返回的单条数据
        
        API返回的字段包括:
        - title: 标题
        - url: 政策文件URL
        - pcode: 发文字号（如 国办发〔2026〕6号）
        - puborg: 发布机构（如 国务院办公厅）
        - pubtimeStr: 发布时间字符串（如 2026.03.06）
        - childtype: 分类（如 公安、安全、司法\公安）
        - summary: 摘要
        - index: 索引号
        """
        try:
            title = item.get('title', '').strip()
            url = item.get('url', '').strip()
            pcode = item.get('pcode', '').strip()
            puborg = item.get('puborg', '').strip() or '国务院'
            pub_time_str = item.get('pubtimeStr', '').strip()
            child_type = item.get('childtype', '').strip()
            summary = item.get('summary', '').strip()
            index_no = item.get('index', '').strip()
            
            if not title:
                return None
            
            # 解析发布日期: "2020.11.05" -> "2020-11-05"
            pub_date = ''
            if pub_time_str:
                pub_date = pub_time_str.replace('.', '-')
            
            # 发文字号
            doc_no = pcode if pcode else self._extract_doc_no(title)
            
            # 分类
            policy_category = self._classify_from_childtype(child_type, title)
            
            # 标签
            tags = self._extract_tags(title, summary)
            
            # 生成唯一ID
            doc_id = self._generate_id(doc_no, title, self.SOURCE.value)
            
            return PolicyDocument(
                id=doc_id,
                source_dept=puborg,
                doc_no=doc_no,
                title=title,
                date=pub_date,
                source=self.SOURCE.value,
                category=policy_category,
                tags=tags,
                url=url,
            )
            
        except Exception as e:
            logger.debug(f"解析搜索结果项失败: {e}")
            return None
    
    def _classify_from_childtype(self, child_type: str, title: str) -> str:
        """根据API返回的childtype字段进行分类"""
        if not child_type:
            return self._classify_policy(title, '')
        
        # 从childtype字段提取主分类
        mapping = {
            '财政': 'macro',
            '金融': 'macro',
            '审计': 'macro',
            '税务': 'macro',
            '证券': 'stock',
            '银行': 'macro',
            '货币': 'macro',
            '保险': 'macro',
            '基金': 'fund',
            '期货': 'futures',
        }
        
        for keyword, category in mapping.items():
            if keyword in child_type:
                return category
        
        # 使用标题作为fallback
        return self._classify_policy(title, '')


def get_gov_policies(start_date: str, end_date: str, **kwargs) -> 'pd.DataFrame':
    """获取国务院政策"""
    collector = GovCouncilCollector()
    return collector.collect(start_date, end_date, **kwargs)
