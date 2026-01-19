"""
工信部政策采集器

采集范围：
- 政策文件
- 部门文件
- 通知公告
- 规章制度

数据源: https://www.miit.gov.cn/
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


class MIITCollector(BasePolicyCollector):
    """
    工信部政策采集器
    
    采集策略：
    1. 使用Playwright渲染列表页（JavaScript动态加载）
    2. 解析政策列表提取标题、日期、链接
    3. 访问详情页提取发文字号和正文
    4. 下载PDF/Word附件
    """
    
    SOURCE = PolicySource.MIIT
    BASE_URL = "https://www.miit.gov.cn"
    
    # 政策列表页
    LIST_URLS = {
        'policy': '/zwgk/zcwj/wjfb/index.html',           # 政策文件
        'department': '/zwgk/zcwj/bmgz/index.html',      # 部门规章
        'notice': '/zwgk/zcwj/wjfb/tz/index.html',       # 通知公告
        'regulation': '/zwgk/zcwj/bmgz/gzxwj/index.html' # 规章制度
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
        采集工信部政策
        
        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            categories: 政策类别
            max_pages: 每个类别最大采集页数
            download_files: 是否下载附件
        """
        import pandas as pd
        
        # 转换日期格式
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        # 加载已有ID去重
        existing_ids = self._load_existing_ids()
        logger.info(f"已有 {len(existing_ids)} 条工信部政策记录")
        
        # 确定采集类别
        if categories is None:
            categories = list(self.LIST_URLS.keys())
        
        all_documents = []
        
        for category in categories:
            if category not in self.LIST_URLS:
                logger.warning(f"未知类别: {category}")
                continue
            
            logger.info(f"采集工信部{category}类政策...")
            docs = self._collect_category(
                category=category,
                start_dt=start_dt,
                end_dt=end_dt,
                max_pages=max_pages,
                download_files=download_files,
                existing_ids=existing_ids
            )
            all_documents.extend(docs)
            
            # 保存当前类别的元数据
            if docs:
                self._save_metadata(docs)
        
        self._close_browser()
        
        logger.info(f"工信部政策采集完成，共 {len(all_documents)} 条")
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
                # 工信部分页格式: index_1.html
                page_url = list_url.replace('index.html', f'index_{page-1}.html')
            
            logger.debug(f"采集页面: {page_url}")
            
            # 获取列表页内容
            items = self._parse_list_page(page_url)
            
            if not items:
                logger.info(f"第{page}页无数据，停止采集")
                break
            
            # 处理每条政策
            reach_start_date = False
            for item in items:
                pub_date = item.get('publish_date', '')
                
                # 日期过滤
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
                
                # 去重检查
                doc_id = self._generate_id(
                    item.get('doc_no', ''),
                    item.get('title', ''),
                    self.SOURCE.value
                )
                
                if doc_id in existing_ids or item.get('doc_no') in existing_ids:
                    continue
                
                # 获取详情
                doc = self._fetch_detail(item, category, download_files)
                if doc:
                    documents.append(doc)
                    existing_ids.add(doc.id)
                    if doc.doc_no:
                        existing_ids.add(doc.doc_no)
                
                time.sleep(0.2)  # 控制请求频率
            
            if reach_start_date:
                logger.info(f"已到达开始日期，停止采集")
                break
            
            time.sleep(0.3)
        
        return documents
    
    def _parse_list_page(self, url: str) -> List[dict]:
        """
        解析政策列表页（使用Playwright渲染）
        
        返回列表：[{title, url, publish_date}, ...]
        """
        items = []
        
        # 工信部网站需要JavaScript渲染
        browser = self._get_playwright_browser()
        html = None
        
        if browser:
            try:
                html = browser.get_with_wait(url, wait_selector='li, .list-item', timeout=10000)
                logger.debug(f"使用Playwright成功获取页面")
            except Exception as e:
                logger.warning(f"Playwright获取失败: {e}，尝试备选方案")
        
        # 备选：尝试requests
        if not html:
            try:
                import requests
                response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                if response.status_code == 200:
                    response.encoding = 'utf-8'
                    html = response.text
            except:
                pass
        
        if not html:
            return items
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # 解析列表项 - 适配工信部网站结构
        exclude_keywords = ['年度报表', '公开指南', '网站地图', '联系方式']
        
        # 多种可能的列表选择器
        list_items = soup.select('.list-item, .zwgk_list li, ul.news_list li')
        
        for item in list_items:
            try:
                # 提取链接和标题
                a_tag = item.select_one('a')
                if not a_tag:
                    continue
                
                title = a_tag.get_text(strip=True)
                href = a_tag.get('href', '')
                
                if not title or not href:
                    continue
                
                # 跳过太短的标题
                if len(title) < 10:
                    continue
                
                # 过滤无关内容
                if any(kw in title for kw in exclude_keywords):
                    continue
                
                # 提取日期
                pub_date = ''
                date_span = item.select_one('.date, .time, span')
                if date_span:
                    date_text = date_span.get_text(strip=True)
                    # 查找日期格式
                    date_match = re.search(r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})', date_text)
                    if date_match:
                        date_str = date_match.group(1)
                        # 标准化日期格式
                        date_str = date_str.replace('年', '-').replace('月', '-').replace('日', '')
                        pub_date = date_str
                
                # 构建完整URL
                if href.startswith('/'):
                    full_url = self.BASE_URL + href
                elif href.startswith('http'):
                    full_url = href
                else:
                    full_url = urljoin(url, href)
                
                items.append({
                    'title': title,
                    'url': full_url,
                    'publish_date': pub_date
                })
                
            except Exception as e:
                logger.debug(f"解析列表项失败: {e}")
                continue
        
        logger.debug(f"解析到 {len(items)} 条列表项")
        return items
    
    def _fetch_detail(
        self,
        item: dict,
        category: str,
        download_files: bool
    ) -> Optional[PolicyDocument]:
        """
        获取政策详情
        
        Args:
            item: 列表项信息 {title, url, publish_date}
            category: 政策类别
            download_files: 是否下载附件
        """
        url = item.get('url', '')
        if not url:
            return None
        
        try:
            # 获取详情页
            response = safe_request(url, timeout=15)
            if not response or response.status_code != 200:
                return None
            
            response.encoding = response.apparent_encoding or 'utf-8'
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            
            # 提取标题
            title = item.get('title', '')
            title_tag = soup.select_one('h1, .article-title, .detail-title')
            if title_tag:
                title = title_tag.get_text(strip=True) or title
            
            # 提取发文字号
            doc_no = ''
            # 查找发文字号区域
            for selector in ['.article-info, .info, .source']:
                info_div = soup.select_one(selector)
                if info_div:
                    info_text = info_div.get_text()
                    doc_no = self._extract_doc_no(info_text)
                    if doc_no:
                        break
            
            # 从标题中提取
            if not doc_no:
                doc_no = self._extract_doc_no(title)
            
            # 提取正文
            content = ''
            content_div = soup.select_one('.article-content, .content, .TRS_Editor')
            if content_div:
                content = content_div.get_text(strip=True)
            
            # 发布日期
            pub_date = item.get('publish_date', '')
            if not pub_date:
                for selector in ['.date, .time, .publish-time']:
                    date_div = soup.select_one(selector)
                    if date_div:
                        pub_date = self._extract_publish_date(date_div.get_text())
                        if pub_date:
                            break
            
            # 查找附件
            attachments = []
            for a in soup.select('a[href$=".pdf"], a[href$=".doc"], a[href$=".docx"]'):
                href = a.get('href', '')
                if href:
                    if href.startswith('/'):
                        href = self.BASE_URL + href
                    elif not href.startswith('http'):
                        href = urljoin(url, href)
                    attachments.append(href)
            
            # 确定文件类型
            file_type = 'html'
            local_path = ''
            
            if attachments and download_files:
                # 下载第一个附件
                attach_url = attachments[0]
                file_type = 'pdf' if '.pdf' in attach_url.lower() else 'doc'
                local_path = str(self._get_file_path(doc_no, title, pub_date, file_type))
                self._download_file(attach_url, local_path)
            
            # 生成唯一ID
            doc_id = self._generate_id(doc_no, title, self.SOURCE.value)
            
            # 分类和标签
            policy_category = self._classify_policy(title, content)
            tags = self._extract_tags(title, content)
            
            return PolicyDocument(
                id=doc_id,
                source_dept="工业和信息化部",
                doc_no=doc_no,
                title=title,
                publish_date=pub_date,
                source=self.SOURCE.value,
                category=policy_category,
                tags=tags,
                file_type=file_type,
                url=url,
                local_path=local_path,
                content_text=content[:5000] if content else '',  # 限制长度
                summary='',
                effective_date='',
                status='active',
                create_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
        except Exception as e:
            logger.error(f"获取详情失败: {url}, 错误: {e}")
            return None


def get_miit_policy(
    start_date: str,
    end_date: str,
    categories: Optional[List[str]] = None,
    max_pages: int = 10,
    download_files: bool = True
) -> 'pd.DataFrame':
    """
    采集工信部政策
    
    Args:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
        categories: 政策类别
        max_pages: 每个类别最大采集页数
        download_files: 是否下载附件
        
    Returns:
        政策元数据DataFrame
        
    Example:
        >>> df = get_miit_policy('20240101', '20241231')
    """
    collector = MIITCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        categories=categories,
        max_pages=max_pages,
        download_files=download_files
    )
