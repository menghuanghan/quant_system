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
import os
import time
import logging
import json
import pandas as pd
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
    """
    
    SOURCE = PolicySource.MIIT
    BASE_URL = "https://www.miit.gov.cn"
    
    # 政策列表页 - 真实的搜索入口
    LIST_URLS = {
        'policy': '/search/wjfb.html?websiteid=110000000000000&pg=10&p=1&tpl=14&category=51&q=',
        'department': '/search/wjfb.html?websiteid=110000000000000&pg=10&p=1&tpl=14&category=52&q=',
        'notice': '/search/wjfb.html?websiteid=110000000000000&pg=10&p=1&tpl=14&category=53&q=',
        'regulation': '/search/wjfb.html?websiteid=110000000000000&pg=10&p=1&tpl=14&category=653&q='
    }
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        categories: Optional[List[str]] = None,
        max_pages: int = 10,
        download_files: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """采集工信部政策"""
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        existing_ids = self._load_existing_ids()
        
        if categories is None:
            categories = list(self.LIST_URLS.keys())
        
        all_documents = []
        for category in categories:
            if category not in self.LIST_URLS: continue
            
            logger.info(f"正在进行工信部【{category}】类政策补采...")
            docs = self._collect_category(category, start_dt, end_dt, max_pages, download_files, existing_ids)
            all_documents.extend(docs)
            
            if docs:
                self._save_metadata(docs)
        
        self._close_browser()
        logger.info(f"工信部采集完成，共新增 {len(all_documents)} 条记录")
        return self.to_dataframe(all_documents)
    
    def _collect_category(self, category, start_dt, end_dt, max_pages, download_files, existing_ids):
        documents = []
        list_url = self.BASE_URL + self.LIST_URLS[category]
        
        for page in range(1, max_pages + 1):
            page_url = f"{list_url}&p={page}"
            logger.debug(f"扫描页面: {page_url}")
            
            items = self._parse_list_page(page_url)
            if not items: break
            
            reach_start_date = False
            for item in items:
                pub_date = item.get('publish_date', '')
                if pub_date:
                    try:
                        item_dt = datetime.strptime(pub_date, '%Y-%m-%d')
                        if item_dt < start_dt:
                            reach_start_date = True; continue
                        if item_dt > end_dt: continue
                    except: pass
                
                doc_id = self._generate_id(item.get('doc_no', ''), item.get('title', ''), self.SOURCE.value)
                if doc_id in existing_ids: continue
                
                doc = self._fetch_detail(item, category, download_files)
                if doc:
                    documents.append(doc)
                    existing_ids.add(doc.id)
                time.sleep(0.5)
                
            if reach_start_date: break
        return documents

    def _parse_list_page(self, url: str) -> List[dict]:
        """使用浏览器上下文注入执行 API 抓取"""
        items = []
        browser = self._get_playwright_browser()
        if not browser: return items
        
        try:
            # 1. 映射分类标签
            category_id = url.split('category=')[1].split('&')[0] if 'category=' in url else '51'
            type_name_map = {'51': '', '52': '部门规章', '53': '通知', '653': '公告'}
            target_type = type_name_map.get(category_id, '')

            # 2. 提取页码
            page_num = 1
            if '&p=' in url:
                try: page_num = int(url.split('&p=')[1].split('&')[0])
                except: pass

            # 3. 访问并注入
            browser.get(url)
            time.sleep(2)
            
            params = {
                'websiteid': '110000000000000',
                'pg': '15',
                'cateid': '57',
                'p': str(page_num),
                '_cus_eq_typename': target_type,
                'selectFields': 'title,url,deploytime,filenumbername'
            }
            
            js_code = f"""
            (async () => {{
                async function fetchMiit() {{
                    const p = {json.dumps(params)};
                    const url = "/search-front-server/api/search/info?" + new URLSearchParams(p).toString();
                    const res = await fetch(url);
                    return await res.json();
                }}
                return await fetchMiit();
            }})();
            """
            result = browser._page.evaluate(js_code)
            
            if result and result.get('success'):
                data_results = result.get('data', {}).get('searchResult', {}).get('dataResults', [])
                for row in data_results:
                    d = row.get('data', {})
                    pub_date = ''
                    if d.get('deploytime'):
                        try:
                            # 强制转换为 float 防止字符串除法错误
                            ts = float(d['deploytime'])
                            pub_date = datetime.fromtimestamp(ts/1000).strftime('%Y-%m-%d')
                        except:
                            pass
                    
                    items.append({
                        'title': d.get('title', ''),
                        'url': self._format_url(self.BASE_URL, d.get('url', '')),
                        'publish_date': pub_date,
                        'doc_no': d.get('filenumbername', '')
                    })
                logger.info(f"API 获取到 {len(items)} 条数据 (类型: {target_type or '全部'})")
        except Exception as e:
            logger.error(f"Playwright 注入抓取异常: {e}")
            
        return items

    def _fetch_detail(self, item: dict, category: str, download_files: bool) -> Optional[PolicyDocument]:
        url = item.get('url', '')
        if not url: return None
        
        try:
            response = safe_request(url, timeout=15)
            if not response or response.status_code != 200: return None
            
            response.encoding = response.apparent_encoding or 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 正文提取
            content_div = soup.select_one('.article-content, .content, #content')
            content = content_div.get_text(strip=True) if content_div else ""
            
            # 附件
            attachments = []
            for a in soup.select('a[href$=".pdf"], a[href$=".doc"], a[href$=".docx"]'):
                href = urljoin(url, a.get('href', ''))
                attachments.append(href)
            
            local_path = ""
            if attachments and download_files:
                local_path = str(self._download_file(attachments[0], item.get('publish_date', '')))

            doc_id = self._generate_id(item.get('doc_no', ''), item.get('title', ''), self.SOURCE.value)
            
            return PolicyDocument(
                id=doc_id,
                source_dept="工业和信息化部",
                doc_no=item.get('doc_no', ''),
                title=item.get('title', ''),
                publish_date=item.get('publish_date', ''),
                source=self.SOURCE.value,
                category=self._classify_policy(item.get('title', ''), content),
                tags=self._extract_tags(item.get('title', ''), content),
                file_type='pdf' if '.pdf' in local_path else 'html',
                url=url,
                local_path=local_path,
                content_text=content[:5000],
                status='active',
                create_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
        except Exception as e:
            logger.error(f"详情抓取异常: {url}, {e}")
            return None

    def _format_url(self, base_url: str, href: str) -> str:
        return urljoin(base_url, href)


def get_miit_policy(
    start_date: str,
    end_date: str,
    categories: Optional[List[str]] = None,
    max_pages: int = 10,
    download_files: bool = True
) -> pd.DataFrame:
    """采集工信部政策的便捷函数"""
    collector = MIITCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        categories=categories,
        max_pages=max_pages,
        download_files=download_files
    )
