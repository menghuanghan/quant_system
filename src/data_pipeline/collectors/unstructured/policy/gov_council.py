"""
国务院/发改委政策采集器

采集范围：
- 国务院政策文件: https://www.gov.cn/zhengce/
- 发改委政策: https://www.ndrc.gov.cn/

特点：
- 网页结构相对规范
- 部分需要Playwright渲染
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
    PolicyCategory
)
from ..request_utils import safe_request

logger = logging.getLogger(__name__)


class GovCouncilCollector(BasePolicyCollector):
    """
    国务院政策采集器
    
    数据源: https://www.gov.cn/zhengce/
    
    采集范围：
    - 最新政策
    - 国务院公报
    - 政策解读
    """
    
    SOURCE = PolicySource.GOV
    BASE_URL = "https://www.gov.cn"
    
    # 政策列表页
    LIST_URLS = {
        'latest': '/zhengce/xxgk/index.htm',        # 政策公开
        'gazette': '/gongbao/guowuyuan/index.htm',  # 国务院公报
        'interpretation': '/zhengce/jiedu.htm',     # 政策解读
    }
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        categories: Optional[List[str]] = None,
        max_pages: int = 10,
        download_files: bool = True,
        **kwargs
    ) -> 'pd.DataFrame':
        """
        采集国务院政策
        
        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            categories: 政策类别 ['latest', 'gazette', 'interpretation']
            max_pages: 每个类别最大采集页数
            download_files: 是否下载附件
        """
        import pandas as pd
        
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        existing_ids = self._load_existing_ids()
        logger.info(f"已有 {len(existing_ids)} 条国务院政策记录")
        
        if categories is None:
            categories = list(self.LIST_URLS.keys())
        
        all_documents = []
        
        for category in categories:
            if category not in self.LIST_URLS:
                logger.warning(f"未知类别: {category}")
                continue
            
            logger.info(f"采集国务院{category}类政策...")
            docs = self._collect_category(
                category=category,
                start_dt=start_dt,
                end_dt=end_dt,
                max_pages=max_pages,
                download_files=download_files,
                existing_ids=existing_ids
            )
            all_documents.extend(docs)
            
            if docs:
                self._save_metadata(docs)
        
        self._close_browser()
        
        logger.info(f"国务院政策采集完成，共 {len(all_documents)} 条")
        return self.to_dataframe(all_documents)
    
    def _collect_category(
        self,
        category: str,
        start_dt: datetime,
        end_dt: datetime,
        max_pages: int,
        download_files: bool,
        existing_ids: set
    ) -> List[PolicyDocument]:
        """采集单个类别的政策"""
        documents = []
        list_url = self.BASE_URL + self.LIST_URLS[category]
        
        for page in range(1, max_pages + 1):
            # 构建分页URL
            if page == 1:
                page_url = list_url
            else:
                # 国务院网站分页格式
                page_url = list_url.replace('.htm', f'_{page}.htm')
            
            logger.debug(f"采集页面: {page_url}")
            
            items = self._parse_list_page(page_url, category)
            
            if not items:
                logger.info(f"第{page}页无数据，停止采集")
                break
            
            reach_start_date = False
            for item in items:
                pub_date = item.get('publish_date', '')
                
                if pub_date:
                    try:
                        item_dt = datetime.strptime(pub_date, '%Y-%m-%d')
                        if item_dt < start_dt:
                            reach_start_date = True
                            continue
                        if item_dt > end_dt:
                            continue
                    except:
                        pass
                
                doc_id = self._generate_id(
                    item.get('doc_no', ''),
                    item.get('title', ''),
                    self.SOURCE.value
                )
                
                if doc_id in existing_ids:
                    continue
                
                doc = self._fetch_detail(item, category, download_files)
                if doc:
                    documents.append(doc)
                    existing_ids.add(doc.id)
                    if doc.doc_no:
                        existing_ids.add(doc.doc_no)
                
                time.sleep(0.2)
            
            if reach_start_date:
                logger.info(f"已到达开始日期，停止采集")
                break
            
            time.sleep(0.3)
        
        return documents
    
    def _parse_list_page(self, url: str, category: str) -> List[dict]:
        """解析政策列表页"""
        items = []
        
        try:
            response = safe_request(url, timeout=15)
            if not response or response.status_code != 200:
                return items
            
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 不同类别的列表格式略有不同
            if category == 'latest':
                # 最新政策列表
                for li in soup.select('.news_box li, .list li'):
                    a_tag = li.find('a')
                    if not a_tag:
                        continue
                    
                    title = a_tag.get_text(strip=True)
                    href = a_tag.get('href', '')
                    
                    # 提取日期
                    date_span = li.find('span')
                    pub_date = ''
                    if date_span:
                        pub_date = self._extract_publish_date(date_span.get_text())
                    
                    if title and href:
                        full_url = urljoin(url, href)
                        items.append({
                            'title': title,
                            'url': full_url,
                            'publish_date': pub_date
                        })
            else:
                # 通用列表格式
                for li in soup.select('ul.list li, .news_list li, .listCont li'):
                    a_tag = li.find('a')
                    if not a_tag:
                        continue
                    
                    title = a_tag.get_text(strip=True)
                    href = a_tag.get('href', '')
                    
                    # 提取日期
                    pub_date = ''
                    date_elem = li.find('span') or li.find('em')
                    if date_elem:
                        pub_date = self._extract_publish_date(date_elem.get_text())
                    
                    if title and href:
                        full_url = urljoin(url, href)
                        items.append({
                            'title': title,
                            'url': full_url,
                            'publish_date': pub_date
                        })
            
        except Exception as e:
            logger.warning(f"解析列表页失败: {url}, {e}")
        
        logger.debug(f"解析到 {len(items)} 条列表项")
        return items
    
    def _fetch_detail(
        self,
        item: dict,
        category: str,
        download_files: bool
    ) -> Optional[PolicyDocument]:
        """获取政策详情"""
        url = item.get('url', '')
        if not url:
            return None
        
        try:
            response = safe_request(url, timeout=15)
            if not response or response.status_code != 200:
                return None
            
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取标题
            title = item.get('title', '')
            title_tag = soup.select_one('h1, .article_title, .pages-title')
            if title_tag:
                title = title_tag.get_text(strip=True) or title
            
            # 提取发文字号
            doc_no = ''
            info_div = soup.select_one('.pages-date, .article-info, .source')
            if info_div:
                doc_no = self._extract_doc_no(info_div.get_text())
            if not doc_no:
                doc_no = self._extract_doc_no(title)
            
            # 提取正文
            content = ''
            content_div = soup.select_one('.pages_content, .article-con, .TRS_Editor')
            if content_div:
                content = content_div.get_text(strip=True)
            
            # 发布日期
            pub_date = item.get('publish_date', '')
            if not pub_date:
                date_div = soup.select_one('.pages-date, .article-date, .time')
                if date_div:
                    pub_date = self._extract_publish_date(date_div.get_text())
            
            # 查找附件
            attachments = []
            for a in soup.select('a[href$=".pdf"], a[href$=".doc"], a[href$=".docx"]'):
                href = a.get('href', '')
                if href:
                    attachments.append(urljoin(url, href))
            
            file_type = 'html'
            local_path = ''
            
            if attachments and download_files:
                attach_url = attachments[0]
                file_type = 'pdf' if '.pdf' in attach_url.lower() else 'doc'
                local_path = str(self._get_file_path(doc_no, title, pub_date, file_type))
                self._download_file(attach_url, local_path)
            
            doc_id = self._generate_id(doc_no, title, self.SOURCE.value)
            policy_category = self._classify_policy(title, content)
            tags = self._extract_tags(title, content)
            
            return PolicyDocument(
                id=doc_id,
                source_dept="国务院",
                doc_no=doc_no,
                title=title,
                publish_date=pub_date,
                source=self.SOURCE.value,
                category=policy_category,
                tags=tags,
                file_type=file_type,
                url=url,
                local_path=local_path,
                content_text=content[:5000] if content else '',
                summary='',
                effective_date='',
                status='active',
                create_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
        except Exception as e:
            logger.error(f"获取详情失败: {url}, 错误: {e}")
            return None


class NDRCCollector(BasePolicyCollector):
    """
    发改委政策采集器
    
    数据源: https://www.ndrc.gov.cn/xxgk/zcfb/
    
    采集范围：
    - 规章
    - 规范性文件
    - 通知公告
    """
    
    SOURCE = PolicySource.NDRC
    BASE_URL = "https://www.ndrc.gov.cn"
    
    LIST_URLS = {
        'rules': '/xxgk/zcfb/ghxwj/index.html',     # 规划文件
        'normative': '/xxgk/zcfb/fzggwl/index.html',  # 发展改革委令
        'notice': '/xxgk/zcfb/tz/index.html',       # 通知
    }
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        categories: Optional[List[str]] = None,
        max_pages: int = 10,
        download_files: bool = True,
        **kwargs
    ) -> 'pd.DataFrame':
        """采集发改委政策"""
        import pandas as pd
        
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        existing_ids = self._load_existing_ids()
        logger.info(f"已有 {len(existing_ids)} 条发改委政策记录")
        
        if categories is None:
            categories = list(self.LIST_URLS.keys())
        
        all_documents = []
        
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
                download_files=download_files,
                existing_ids=existing_ids
            )
            all_documents.extend(docs)
            
            if docs:
                self._save_metadata(docs)
        
        self._close_browser()
        
        logger.info(f"发改委政策采集完成，共 {len(all_documents)} 条")
        return self.to_dataframe(all_documents)
    
    def _collect_category(
        self,
        category: str,
        start_dt: datetime,
        end_dt: datetime,
        max_pages: int,
        download_files: bool,
        existing_ids: set
    ) -> List[PolicyDocument]:
        """采集单个类别的政策"""
        documents = []
        list_url = self.BASE_URL + self.LIST_URLS[category]
        
        for page in range(1, max_pages + 1):
            if page == 1:
                page_url = list_url
            else:
                page_url = list_url.replace('.html', f'_{page}.html')
            
            logger.debug(f"采集页面: {page_url}")
            
            items = self._parse_list_page(page_url)
            
            if not items:
                logger.info(f"第{page}页无数据，停止采集")
                break
            
            reach_start_date = False
            for item in items:
                pub_date = item.get('publish_date', '')
                
                if pub_date:
                    try:
                        item_dt = datetime.strptime(pub_date, '%Y-%m-%d')
                        if item_dt < start_dt:
                            reach_start_date = True
                            continue
                        if item_dt > end_dt:
                            continue
                    except:
                        pass
                
                doc_id = self._generate_id(
                    item.get('doc_no', ''),
                    item.get('title', ''),
                    self.SOURCE.value
                )
                
                if doc_id in existing_ids:
                    continue
                
                doc = self._fetch_detail(item, download_files)
                if doc:
                    documents.append(doc)
                    existing_ids.add(doc.id)
                    if doc.doc_no:
                        existing_ids.add(doc.doc_no)
                
                time.sleep(0.2)
            
            if reach_start_date:
                break
            
            time.sleep(0.3)
        
        return documents
    
    def _parse_list_page(self, url: str) -> List[dict]:
        """解析政策列表页"""
        items = []
        
        try:
            response = safe_request(url, timeout=15)
            if not response or response.status_code != 200:
                return items
            
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for li in soup.select('ul.list li, .u-list li, .news_list li'):
                a_tag = li.find('a')
                if not a_tag:
                    continue
                
                title = a_tag.get_text(strip=True)
                href = a_tag.get('href', '')
                
                pub_date = ''
                date_span = li.find('span')
                if date_span:
                    pub_date = self._extract_publish_date(date_span.get_text())
                
                if title and href:
                    full_url = urljoin(url, href)
                    items.append({
                        'title': title,
                        'url': full_url,
                        'publish_date': pub_date
                    })
            
        except Exception as e:
            logger.warning(f"解析列表页失败: {url}, {e}")
        
        return items
    
    def _fetch_detail(
        self,
        item: dict,
        download_files: bool
    ) -> Optional[PolicyDocument]:
        """获取政策详情"""
        url = item.get('url', '')
        if not url:
            return None
        
        try:
            response = safe_request(url, timeout=15)
            if not response or response.status_code != 200:
                return None
            
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title = item.get('title', '')
            title_tag = soup.select_one('h1, .article-title, .news_title')
            if title_tag:
                title = title_tag.get_text(strip=True) or title
            
            doc_no = ''
            info_div = soup.select_one('.article-info, .news_info, .source')
            if info_div:
                doc_no = self._extract_doc_no(info_div.get_text())
            if not doc_no:
                doc_no = self._extract_doc_no(title)
            
            content = ''
            content_div = soup.select_one('.article-content, .TRS_Editor, .news_content')
            if content_div:
                content = content_div.get_text(strip=True)
            
            pub_date = item.get('publish_date', '')
            
            attachments = []
            for a in soup.select('a[href$=".pdf"], a[href$=".doc"], a[href$=".docx"]'):
                href = a.get('href', '')
                if href:
                    attachments.append(urljoin(url, href))
            
            file_type = 'html'
            local_path = ''
            
            if attachments and download_files:
                attach_url = attachments[0]
                file_type = 'pdf' if '.pdf' in attach_url.lower() else 'doc'
                local_path = str(self._get_file_path(doc_no, title, pub_date, file_type))
                self._download_file(attach_url, local_path)
            
            doc_id = self._generate_id(doc_no, title, self.SOURCE.value)
            policy_category = self._classify_policy(title, content)
            tags = self._extract_tags(title, content)
            
            return PolicyDocument(
                id=doc_id,
                source_dept="国家发改委",
                doc_no=doc_no,
                title=title,
                publish_date=pub_date,
                source=self.SOURCE.value,
                category=policy_category,
                tags=tags,
                file_type=file_type,
                url=url,
                local_path=local_path,
                content_text=content[:5000] if content else '',
                summary='',
                effective_date='',
                status='active',
                create_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
        except Exception as e:
            logger.error(f"获取详情失败: {url}, 错误: {e}")
            return None


def get_gov_policy(
    start_date: str,
    end_date: str,
    categories: Optional[List[str]] = None,
    max_pages: int = 10,
    download_files: bool = True
) -> 'pd.DataFrame':
    """
    采集国务院政策
    
    Args:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        categories: ['latest', 'gazette', 'interpretation']
        max_pages: 每个类别最大采集页数
        download_files: 是否下载附件
    """
    collector = GovCouncilCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        categories=categories,
        max_pages=max_pages,
        download_files=download_files
    )


def get_ndrc_policy(
    start_date: str,
    end_date: str,
    categories: Optional[List[str]] = None,
    max_pages: int = 10,
    download_files: bool = True
) -> 'pd.DataFrame':
    """
    采集发改委政策
    
    Args:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        categories: ['rules', 'normative', 'notice']
        max_pages: 每个类别最大采集页数
        download_files: 是否下载附件
    """
    collector = NDRCCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        categories=categories,
        max_pages=max_pages,
        download_files=download_files
    )
