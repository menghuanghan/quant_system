"""
国家发改委政策采集器（简化版）

采集范围：
- 通知
- 政策/规划文本
- 解读

数据源: https://www.ndrc.gov.cn
"""

import re
import time
import logging
from typing import Optional, List
from datetime import datetime
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base_policy import (
    BasePolicyCollector,
    PolicyDocument,
    PolicySource,
)
from ..request_utils import safe_request

logger = logging.getLogger(__name__)


class NDRCCollector(BasePolicyCollector):
    """国家发改委政策采集器"""
    
    SOURCE = PolicySource.NDRC
    BASE_URL = "https://www.ndrc.gov.cn"
    
    # 政策列表页（更新为实际存在的URL）
    LIST_URLS = {
        'notice': '/xxgk/zcfb/tz/',      # 通知
        'policy': '/xxgk/zcfb/ghwb/',    # 规划文本
        'normative': '/xxgk/zcfb/ghxwj/', # 规范性文件
        'announcement': '/xxgk/zcfb/gg/', # 公告
    }
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        categories: Optional[List[str]] = None,
        max_pages: int = 30,
        **kwargs
    ) -> 'pd.DataFrame':
        """
        采集发改委政策
        
        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            categories: 政策类别列表
            max_pages: 每个类别最大采集页数
        """
        import pandas as pd
        
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        if categories is None:
            categories = list(self.LIST_URLS.keys())
        
        all_documents = []
        seen_ids = set()
        
        for category in categories:
            if category not in self.LIST_URLS:
                logger.warning(f"未知类别: {category}")
                continue
            
            logger.info(f"采集发改委{category}类政策...")
            docs = self._collect_category(
                category=category,
                start_dt=start_dt,
                end_dt=end_dt,
                max_pages=max_pages,
                seen_ids=seen_ids
            )
            all_documents.extend(docs)
        
        logger.info(f"发改委政策采集完成，共 {len(all_documents)} 条")
        return self.to_dataframe(all_documents)
    
    def _collect_category(
        self,
        category: str,
        start_dt: datetime,
        end_dt: datetime,
        max_pages: int,
        seen_ids: set
    ) -> List[PolicyDocument]:
        """采集单个类别的政策"""
        documents = []
        base_url = self.BASE_URL + self.LIST_URLS[category]
        
        for page in range(1, max_pages + 1):
            if page == 1:
                page_url = base_url + 'index.html'
            else:
                page_url = base_url + f'index_{page-1}.html'
            
            logger.debug(f"采集页面: {page_url}")
            
            items = self._parse_list_page(page_url)
            
            if not items:
                break
            
            reach_start_date = False
            for item in items:
                pub_date = item.get('date', '')
                
                if not pub_date:
                    continue
                
                try:
                    item_dt = datetime.strptime(pub_date, '%Y-%m-%d')
                    if item_dt < start_dt:
                        reach_start_date = True
                        continue
                    if item_dt > end_dt:
                        continue
                except:
                    continue
                
                doc_id = self._generate_id(
                    item.get('doc_no', ''),
                    item.get('title', ''),
                    self.SOURCE.value
                )
                
                if doc_id in seen_ids:
                    continue
                
                doc = self._fetch_detail(item, category)
                if doc:
                    documents.append(doc)
                    seen_ids.add(doc.id)
                
                time.sleep(0.2)
            
            if reach_start_date:
                break
            
            time.sleep(0.3)
        
        return documents
    
    def _parse_list_page(self, url: str) -> List[dict]:
        """解析政策列表页"""
        items = []
        
        try:
            response = safe_request(url, method='GET', timeout=10)
            if not response or response.status_code != 200:
                logger.warning(f"请求失败: {url}, status={response.status_code if response else 'None'}")
                return items
            
            # 确保使用 UTF-8 编码
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 发改委新版网站结构：ul.u-list > li
            list_container = soup.select_one('ul.u-list') or soup.select_one('div.list ul')
            if not list_container:
                logger.debug(f"未找到列表容器: {url}")
                return items
            
            for li in list_container.select('li'):
                try:
                    a_tag = li.select_one('a')
                    if not a_tag:
                        continue
                    
                    title = a_tag.get_text(strip=True)
                    href = a_tag.get('href', '')
                    
                    if not title or not href or len(title) < 10:
                        continue
                    
                    # 过滤导航链接
                    if href.startswith('../') and 'jd' in href:
                        continue  # 解读链接
                    
                    # 提取日期 (格式: YYYY/MM/DD)
                    pub_date = ''
                    date_span = li.select_one('span')
                    if date_span:
                        date_text = date_span.get_text(strip=True)
                        # 转换 YYYY/MM/DD 为 YYYY-MM-DD
                        date_match = re.search(r'(\d{4})/(\d{2})/(\d{2})', date_text)
                        if date_match:
                            pub_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                        else:
                            pub_date = self._extract_publish_date(date_text)
                    
                    # 构建完整URL
                    if href.startswith('./'):
                        # 相对路径
                        full_url = urljoin(url, href)
                    elif href.startswith('/'):
                        full_url = self.BASE_URL + href
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        full_url = urljoin(url, href)
                    
                    items.append({
                        'title': title,
                        'url': full_url,
                        'date': pub_date
                    })
                    
                except Exception as e:
                    logger.debug(f"解析列表项失败: {e}")
        
        except Exception as e:
            logger.warning(f"获取列表页失败 {url}: {e}")
        
        return items
    
    def _fetch_detail(self, item: dict, category: str) -> Optional[PolicyDocument]:
        """获取政策详情（不含content）"""
        title = item.get('title', '')
        url = item.get('url', '')
        pub_date = item.get('date', '')
        
        if not url:
            return None
        
        # 提取发文字号
        doc_no = self._extract_doc_no(title)
        
        # 分类和标签（基于标题）
        policy_category = self._classify_policy(title, '')
        tags = self._extract_tags(title, '')
        
        doc_id = self._generate_id(doc_no, title, self.SOURCE.value)
        
        return PolicyDocument(
            id=doc_id,
            source_dept='发改委',
            doc_no=doc_no,
            title=title,
            date=pub_date,
            source=self.SOURCE.value,
            category=policy_category,
            tags=tags,
            url=url,
            original_category=category
        )
    
    def _fetch_content(self, url: str) -> Optional[str]:
        """获取详情页正文 - 保留方法但不使用"""
        # 不再获取content以提高采集效率
        return None


def get_ndrc_policies(start_date: str, end_date: str, **kwargs) -> 'pd.DataFrame':
    """获取发改委政策"""
    collector = NDRCCollector()
    return collector.collect(start_date, end_date, **kwargs)
