"""
东方财富政策中心采集器

聚合来源，用于历史数据回补

数据源: https://data.eastmoney.com/zcfb/

特点：
- 政策已按类别整理（宏观、股市、行业等）
- 更新频率高
- 适合做补充数据源
"""

import re
import time
import json
import logging
from typing import Optional, List
from datetime import datetime
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base_policy import (
    BasePolicyCollector,
    PolicyDocument,
    PolicySource,
    PolicyCategory
)
from ..request_utils import safe_request

logger = logging.getLogger(__name__)


class EastMoneyPolicyCollector(BasePolicyCollector):
    """
    东方财富政策中心采集器
    
    特点：
    1. 政策已分类整理
    2. 提供API接口，无需复杂爬取
    3. 更新及时，覆盖全面
    """
    
    SOURCE = PolicySource.EASTMONEY
    BASE_URL = "https://data.eastmoney.com"
    
    # API接口
    API_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    
    # 政策类别映射
    CATEGORY_MAP = {
        'macro': '宏观政策',
        'stock': '股市政策',
        'bond': '债市政策',
        'fund': '基金政策',
        'futures': '期货政策',
        'bank': '银行政策',
        'insurance': '保险政策',
        'forex': '外汇政策',
    }
    
    # 报表名称映射 - 使用通用政策发布API
    REPORT_NAME_MAP = {
        'macro': 'RPT_ECONOMIC_POLICY',       # 宏观经济政策
        'stock': 'RPT_ECONOMIC_POLICY',       # 股票市场政策（使用通用接口）
        'bond': 'RPT_ECONOMIC_POLICY',        # 债券市场政策
        'fund': 'RPT_ECONOMIC_POLICY',        # 基金市场政策
        'futures': 'RPT_ECONOMIC_POLICY',     # 期货期权政策
    }
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        categories: Optional[List[str]] = None,
        max_pages: int = 20,
        download_files: bool = False,
        **kwargs
    ) -> 'pd.DataFrame':
        """
        采集东方财富政策中心数据
        
        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            categories: 政策类别 ['macro', 'stock', 'bond', 'fund', 'futures']
            max_pages: 每个类别最大采集页数
            download_files: 是否下载附件（东方财富一般不提供）
        """
        import pandas as pd
        
        # 转换日期格式
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        start_date_fmt = start_dt.strftime('%Y-%m-%d')
        end_date_fmt = end_dt.strftime('%Y-%m-%d')
        
        existing_ids = self._load_existing_ids()
        logger.info(f"已有 {len(existing_ids)} 条东方财富政策记录")
        
        if categories is None:
            categories = list(self.REPORT_NAME_MAP.keys())
        
        all_documents = []
        
        for category in categories:
            if category not in self.REPORT_NAME_MAP:
                logger.warning(f"未知类别: {category}")
                continue
            
            logger.info(f"采集东方财富{self.CATEGORY_MAP.get(category, category)}...")
            docs = self._collect_category(
                category=category,
                start_date=start_date_fmt,
                end_date=end_date_fmt,
                max_pages=max_pages,
                existing_ids=existing_ids
            )
            all_documents.extend(docs)
            
            if docs:
                self._save_metadata(docs)
            
            time.sleep(0.3)
        
        logger.info(f"东方财富政策采集完成，共 {len(all_documents)} 条")
        return self.to_dataframe(all_documents)
    
    def _collect_category(
        self,
        category: str,
        start_date: str,
        end_date: str,
        max_pages: int,
        existing_ids: set
    ) -> List[PolicyDocument]:
        """通过API采集单个类别的政策"""
        documents = []
        
        # 先尝试API方式
        api_docs = self._collect_via_api(category, start_date, end_date, max_pages, existing_ids)
        if api_docs:
            return api_docs
        
        # API失败，尝试网页爬取
        logger.info(f"API采集失败，尝试网页爬取...")
        return self._collect_via_scraping(category, start_date, end_date, max_pages, existing_ids)
    
    def _collect_via_api(
        self,
        category: str,
        start_date: str,
        end_date: str,
        max_pages: int,
        existing_ids: set
    ) -> List[PolicyDocument]:
        """通过API采集政策"""
        documents = []
        report_name = self.REPORT_NAME_MAP[category]
        
        for page in range(1, max_pages + 1):
            logger.debug(f"采集 {category} 第 {page} 页...")
            
            # 构建API请求
            params = {
                'reportName': report_name,
                'columns': 'ALL',
                'pageNumber': page,
                'pageSize': 50,
                'sortColumns': 'NOTICE_DATE',
                'sortTypes': '-1',
                'filter': f'(NOTICE_DATE>="{start_date}")(NOTICE_DATE<="{end_date}")',
            }
            
            try:
                response = safe_request(
                    self.API_URL,
                    params=params,
                    timeout=15
                )
                
                if not response or response.status_code != 200:
                    logger.warning(f"API请求失败: {response.status_code if response else 'None'}")
                    break
                
                data = response.json()
                
                if not data.get('success'):
                    logger.warning(f"API返回错误: {data.get('message')}")
                    break
                
                result = data.get('result', {})
                items = result.get('data', [])
                
                if not items:
                    logger.info(f"第{page}页无数据，停止采集")
                    break
                
                # 处理每条政策
                for item in items:
                    doc = self._parse_api_item(item, category, existing_ids)
                    if doc:
                        documents.append(doc)
                        existing_ids.add(doc.id)
                        if doc.doc_no:
                            existing_ids.add(doc.doc_no)
                
                # 检查是否还有更多页
                total_pages = result.get('pages', 1)
                if page >= total_pages:
                    break
                
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"API采集失败: {e}")
                break
        
        return documents
    
    def _parse_api_item(
        self,
        item: dict,
        category: str,
        existing_ids: set
    ) -> Optional[PolicyDocument]:
        """解析API返回的数据项"""
        try:
            # 提取字段
            title = item.get('TITLE', '') or item.get('NOTICE_TITLE', '')
            if not title:
                return None
            
            # 发布日期
            pub_date_raw = item.get('NOTICE_DATE', '') or item.get('PUBLISH_DATE', '')
            pub_date = ''
            if pub_date_raw:
                try:
                    if 'T' in str(pub_date_raw):
                        pub_date = pub_date_raw.split('T')[0]
                    else:
                        pub_date = self._extract_publish_date(str(pub_date_raw))
                except:
                    pass
            
            # 发文机构
            dept = item.get('DEPARTMENT', '') or item.get('ORG_NAME', '') or '东方财富'
            
            # 发文字号
            doc_no = item.get('DOC_NO', '') or item.get('DOCUMENT_NO', '')
            if not doc_no:
                doc_no = self._extract_doc_no(title)
            
            # 内容摘要
            content = item.get('CONTENT', '') or item.get('ABSTRACT', '') or ''
            
            # 原文链接
            url = item.get('URL', '') or item.get('DETAIL_URL', '')
            if not url:
                # 构建默认详情页URL
                info_code = item.get('INFO_CODE', '')
                if info_code:
                    url = f"https://data.eastmoney.com/zcfb/detail/{info_code}.html"
            
            # 生成ID
            doc_id = self._generate_id(doc_no, title, self.SOURCE.value)
            
            if doc_id in existing_ids:
                return None
            
            # 分类和标签
            policy_category = category
            tags = self._extract_tags(title, content)
            
            # 添加来源标签
            if dept:
                if '证监会' in dept:
                    tags.append('证监会')
                elif '央行' in dept or '人民银行' in dept:
                    tags.append('央行')
                elif '财政部' in dept:
                    tags.append('财政部')
                elif '发改委' in dept:
                    tags.append('发改委')
            
            return PolicyDocument(
                id=doc_id,
                source_dept=dept,
                doc_no=doc_no,
                title=title,
                publish_date=pub_date,
                source=self.SOURCE.value,
                category=policy_category,
                tags=tags,
                file_type='html',
                url=url,
                local_path='',
                content_text=content[:5000] if content else '',
                summary=content[:500] if content else '',
                effective_date='',
                status='active',
                create_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
        except Exception as e:
            logger.debug(f"解析API数据失败: {e}")
            return None
    
    def _collect_via_scraping(
        self,
        category: str,
        start_date: str,
        end_date: str,
        max_pages: int,
        existing_ids: set
    ) -> List[PolicyDocument]:
        """
        通过网页爬取采集政策
        
        备选方案：当API不可用时使用
        """
        documents = []
        
        # 东方财富财经新闻页面（包含政策新闻）
        news_url = "https://finance.eastmoney.com/a/czcsj.html"
        
        try:
            response = safe_request(news_url, timeout=15)
            if not response or response.status_code != 200:
                logger.warning(f"网页请求失败")
                return documents
            
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 解析新闻列表
            for li in soup.select('.articleList li, .news_list li, ul.newList li'):
                a_tag = li.find('a')
                if not a_tag:
                    continue
                
                title = a_tag.get_text(strip=True)
                href = a_tag.get('href', '')
                
                if not title or not href:
                    continue
                
                # 提取日期
                pub_date = ''
                date_span = li.find('span', class_='time') or li.find('span')
                if date_span:
                    pub_date = self._extract_publish_date(date_span.get_text())
                
                # 日期过滤
                if pub_date:
                    try:
                        item_dt = datetime.strptime(pub_date, '%Y-%m-%d')
                        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                        if item_dt < start_dt or item_dt > end_dt:
                            continue
                    except:
                        pass
                
                # 生成ID并去重
                doc_no = self._extract_doc_no(title)
                doc_id = self._generate_id(doc_no, title, self.SOURCE.value)
                
                if doc_id in existing_ids:
                    continue
                
                # 分类和标签
                tags = self._extract_tags(title, '')
                
                doc = PolicyDocument(
                    id=doc_id,
                    source_dept="东方财富",
                    doc_no=doc_no,
                    title=title,
                    publish_date=pub_date,
                    source=self.SOURCE.value,
                    category=category,
                    tags=tags,
                    file_type='html',
                    url=href,
                    local_path='',
                    content_text='',
                    summary='',
                    effective_date='',
                    status='active',
                    create_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
                
                documents.append(doc)
                existing_ids.add(doc.id)
                
                if len(documents) >= 50:  # 限制数量
                    break
            
            logger.info(f"网页爬取获得 {len(documents)} 条政策")
            
        except Exception as e:
            logger.error(f"网页爬取失败: {e}")
        
        return documents
    
    def collect_by_keyword(
        self,
        keyword: str,
        start_date: str,
        end_date: str,
        max_pages: int = 10
    ) -> 'pd.DataFrame':
        """
        按关键词搜索政策
        
        Args:
            keyword: 搜索关键词
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            max_pages: 最大页数
        """
        import pandas as pd
        
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        documents = []
        existing_ids = self._load_existing_ids()
        
        logger.info(f"按关键词搜索政策: {keyword}")
        
        for page in range(1, max_pages + 1):
            params = {
                'reportName': 'RPT_ZCFB_SEARCH',
                'columns': 'ALL',
                'pageNumber': page,
                'pageSize': 50,
                'sortColumns': 'NOTICE_DATE',
                'sortTypes': '-1',
                'filter': f'(KEYWORD="{keyword}")',
            }
            
            try:
                response = safe_request(self.API_URL, params=params, timeout=15)
                
                if not response or response.status_code != 200:
                    break
                
                data = response.json()
                items = data.get('result', {}).get('data', [])
                
                if not items:
                    break
                
                for item in items:
                    # 日期过滤
                    pub_date = item.get('NOTICE_DATE', '')
                    if pub_date:
                        try:
                            item_dt = datetime.strptime(pub_date.split('T')[0], '%Y-%m-%d')
                            if item_dt < start_dt or item_dt > end_dt:
                                continue
                        except:
                            pass
                    
                    doc = self._parse_api_item(item, 'search', existing_ids)
                    if doc:
                        documents.append(doc)
                        existing_ids.add(doc.id)
                
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"关键词搜索失败: {e}")
                break
        
        if documents:
            self._save_metadata(documents, f"eastmoney_search_{keyword}_{start_date}.jsonl")
        
        logger.info(f"关键词搜索完成，共 {len(documents)} 条")
        return self.to_dataframe(documents)


def get_eastmoney_policy(
    start_date: str,
    end_date: str,
    categories: Optional[List[str]] = None,
    max_pages: int = 20
) -> 'pd.DataFrame':
    """
    采集东方财富政策中心数据
    
    Args:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        categories: ['macro', 'stock', 'bond', 'fund', 'futures']
        max_pages: 每个类别最大采集页数
        
    Returns:
        政策元数据DataFrame
    """
    collector = EastMoneyPolicyCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        categories=categories,
        max_pages=max_pages
    )


def search_policy(
    keyword: str,
    start_date: str,
    end_date: str,
    max_pages: int = 10
) -> 'pd.DataFrame':
    """
    按关键词搜索政策
    
    Args:
        keyword: 搜索关键词
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        max_pages: 最大页数
    """
    collector = EastMoneyPolicyCollector()
    return collector.collect_by_keyword(
        keyword=keyword,
        start_date=start_date,
        end_date=end_date,
        max_pages=max_pages
    )
