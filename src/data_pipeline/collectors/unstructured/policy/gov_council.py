"""
国务院政策采集器（简化版）

采集范围：
- 国务院政策文件

数据源: www.gov.cn
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


class GovCouncilCollector(BasePolicyCollector):
    """国务院政策采集器"""
    
    SOURCE = PolicySource.GOV
    BASE_URL = "https://www.gov.cn"
    LIST_URL = "https://www.gov.cn/zhengce/zuixin/home.htm"
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        max_results: int = 1000,
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
        
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        logger.info(f"采集国务院政策文件 ({start_dt.date()} 至 {end_dt.date()})...")
        
        all_documents = []
        seen_ids = set()
        page_num = 0
        
        while len(all_documents) < max_results:
            if page_num == 0:
                page_url = self.LIST_URL
            else:
                page_url = f"https://www.gov.cn/zhengce/zuixin/home_{page_num}.htm"
            
            items = self._parse_list_page(page_url)
            
            if not items:
                break
            
            for item in items:
                if len(all_documents) >= max_results:
                    break
                
                pub_date_str = item.get('date', '')
                if not pub_date_str:
                    continue
                
                try:
                    pub_date = datetime.strptime(pub_date_str, '%Y-%m-%d')
                    if pub_date < start_dt or pub_date > end_dt:
                        continue
                except:
                    continue
                
                doc_id = self._generate_id(
                    '',
                    item.get('title', ''),
                    self.SOURCE.value
                )
                
                if doc_id in seen_ids:
                    continue
                
                doc = self._fetch_detail(item)
                if doc:
                    all_documents.append(doc)
                    seen_ids.add(doc.id)
                
                time.sleep(0.2)
            
            page_num += 1
            time.sleep(0.3)
        
        logger.info(f"国务院政策采集完成，共 {len(all_documents)} 条")
        return self.to_dataframe(all_documents)
    
    def _parse_list_page(self, url: str) -> List[dict]:
        """解析政策列表页"""
        items = []
        
        try:
            response = safe_request(url, method='GET', timeout=10)
            if not response or response.status_code != 200:
                return items
            
            # 确保使用UTF-8编码
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 国务院网站结构（多种选择器尝试）
            list_items = (
                soup.select('ul.list li h4 a') or 
                soup.select('ul.list li a') or
                soup.select('.news_box a') or
                soup.select('.list_main a')
            )
            
            for a_tag in list_items:
                try:
                    title = a_tag.get_text(strip=True)
                    href = a_tag.get('href', '')
                    
                    if not title or not href or len(title) < 5:
                        continue
                    
                    # 提取日期
                    pub_date = ''
                    parent = a_tag.parent
                    if parent:
                        parent = parent.parent  # li
                        if parent:
                            date_elem = parent.find('span', class_='date')
                            if date_elem:
                                date_text = date_elem.get_text(strip=True)
                                pub_date = self._extract_publish_date(date_text)
                    
                    # 构建URL
                    if href.startswith('/'):
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
    
    def _fetch_detail(self, item: dict) -> Optional[PolicyDocument]:
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
            source_dept='国务院',
            doc_no=doc_no,
            title=title,
            date=pub_date,
            source=self.SOURCE.value,
            category=policy_category,
            tags=tags,
            url=url
        )
    
    def _fetch_content(self, url: str) -> Optional[str]:
        """获取详情页正文 - 保留方法但不使用"""
        # 不再获取content以提高采集效率
        return None


def get_gov_policies(start_date: str, end_date: str, **kwargs) -> 'pd.DataFrame':
    """获取国务院政策"""
    collector = GovCouncilCollector()
    return collector.collect(start_date, end_date, **kwargs)
