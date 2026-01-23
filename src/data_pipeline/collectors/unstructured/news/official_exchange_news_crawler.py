"""
上交所/深交所官方公告采集器

直接从交易所官方网站API获取监管措施、公告等数据
"""

import logging
import hashlib
import json
import time
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..base import UnstructuredCollector
from ..request_utils import safe_request
from ..cleaning_adapter import CleaningMixin

logger = logging.getLogger(__name__)


class OfficialExchangeNewsCrawler(UnstructuredCollector, CleaningMixin):
    """
    上交所/深交所官方公告采集器
    
    数据来源:
    - 上交所: query.sse.com.cn API
    - 深交所: www.szse.cn API
    """
    
    STANDARD_FIELDS = [
        'news_id',
        'title',
        'content',
        'date',
        'source',
        'category',
        'url',
        'related_stocks',
        'stock_code',
        'stock_name',
        'keywords',
    ]
    
    # 上交所API配置
    SSE_API_URL = "https://query.sse.com.cn/commonSoaQuery.do"
    SSE_BASE_URL = "http://www.sse.com.cn"
    
    # 深交所API配置
    SZSE_API_URL = "https://www.szse.cn/api/report/ShowReport/data"
    SZSE_BASE_URL = "http://www.szse.cn"
    
    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        })
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        exchanges: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集交易所公告
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            exchanges: 交易所列表 ['sse', 'szse'], 默认全部
        
        Returns:
            公告数据DataFrame
        """
        if exchanges is None:
            exchanges = ['sse', 'szse']
        
        all_data = []
        
        if 'sse' in exchanges:
            logger.info(f"开始采集上交所公告: {start_date} ~ {end_date}")
            sse_data = self._collect_sse(start_date, end_date)
            if not sse_data.empty:
                all_data.append(sse_data)
                logger.info(f"上交所采集完成: {len(sse_data)} 条")
        
        if 'szse' in exchanges:
            logger.info(f"开始采集深交所公告: {start_date} ~ {end_date}")
            szse_data = self._collect_szse(start_date, end_date)
            if not szse_data.empty:
                all_data.append(szse_data)
                logger.info(f"深交所采集完成: {len(szse_data)} 条")
        
        if not all_data:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=['news_id'], keep='first')
        
        return self._standardize_output(result)
    
    def _collect_sse(self, start_date: str, end_date: str) -> pd.DataFrame:
        """采集上交所监管措施数据"""
        all_records = []
        
        # 转换日期格式
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        start_time_str = start_dt.strftime('%Y-%m-%d 00:00:00')
        end_time_str = end_dt.strftime('%Y-%m-%d 23:59:59')
        
        # 分别采集主板和科创板
        for board_type in ['4', '0']:  # 4=主板, 0=科创板
            page_no = 1
            page_size = 25
            
            while True:
                params = {
                    'isPagination': 'true',
                    'pageHelp.pageSize': str(page_size),
                    'pageHelp.pageNo': str(page_no),
                    'pageHelp.beginPage': str(page_no),
                    'pageHelp.cacheSize': '1',
                    'pageHelp.endPage': str(page_no),
                    'sqlId': 'BS_KCB_GGLL_NEW',
                    'siteId': '28',
                    'channelId': '10007,10008,10009,10010',  # 所有监管类型
                    'type': board_type,
                    'stockcode': '',
                    'extTeacher': '',
                    'extWTFL': '',
                    'createTime': start_time_str,
                    'createTimeEnd': end_time_str,
                    'order': 'createTime|desc,stockcode|asc',
                    '_': str(int(time.time() * 1000))
                }
                
                try:
                    self.session.headers['Referer'] = 'http://www.sse.com.cn/disclosure/credibility/supervision/measures/'
                    response = self.session.get(self.SSE_API_URL, params=params, timeout=30)
                    
                    if response.status_code != 200:
                        logger.warning(f"上交所API返回状态码: {response.status_code}")
                        break
                    
                    # 解析JSONP响应
                    text = response.text
                    # 移除JSONP包装
                    json_str = re.sub(r'^\w+\((.*)\)$', r'\1', text, flags=re.DOTALL)
                    data = json.loads(json_str)
                    
                    if 'result' not in data or not data['result']:
                        break
                    
                    records = data['result']
                    if not records:
                        break
                    
                    for record in records:
                        all_records.append(self._parse_sse_record(record))
                    
                    # 检查是否还有更多页
                    total_pages = data.get('pageHelp', {}).get('pageCount', 0)
                    if page_no >= total_pages:
                        break
                    
                    page_no += 1
                    time.sleep(0.5)  # 避免请求过快
                    
                except Exception as e:
                    logger.error(f"采集上交所第{page_no}页失败: {e}")
                    break
        
        if not all_records:
            return pd.DataFrame()
        
        return pd.DataFrame(all_records)
    
    def _parse_sse_record(self, record: Dict) -> Dict:
        """解析上交所单条记录"""
        stock_code = record.get('extSECURITY_CODE', '')
        stock_name = record.get('extGSJC', '')
        title = record.get('docTitle', '')
        doc_url = record.get('docURL', '')
        create_time = record.get('createTime', '')
        measure_type = record.get('extWTFL', '')
        
        # 处理URL
        if doc_url and not doc_url.startswith('http'):
            doc_url = self.SSE_BASE_URL + doc_url
        
        # 生成ID
        news_id = self._generate_id(f"sse_{stock_code}_{title}_{create_time}")
        
        # 解析日期
        pub_date = ''
        pub_time = ''
        if create_time:
            try:
                dt = datetime.strptime(create_time, '%Y-%m-%d %H:%M:%S')
                pub_date = dt.strftime('%Y-%m-%d')
                pub_time = create_time
            except:
                pub_date = create_time[:10] if len(create_time) >= 10 else create_time
                pub_time = create_time
        
        return {
            'news_id': news_id,
            'title': title,
            'content': '',  # 需要下载PDF才能获取
            'date': pub_date,
            'source': 'sse',
            'category': measure_type or '监管措施',
            'url': doc_url,
            'related_stocks': f"{stock_code}.{stock_name}" if stock_code else '',
            'stock_code': stock_code,
            'stock_name': stock_name,
            'keywords': measure_type,
        }
    
    def _collect_szse(self, start_date: str, end_date: str) -> pd.DataFrame:
        """采集深交所监管措施数据"""
        all_records = []
        
        page_no = 1
        page_size = 30
        
        while True:
            params = {
                'SHOWTYPE': 'JSON',
                'CATALOGID': '1800_jgxxgk',
                'TABKEY': 'tab1',
                'selectBkmc': '0',
                'txtDate': start_date,
                'txtEnd': end_date,
                'PAGENO': str(page_no),
            }
            
            try:
                self.session.headers['Referer'] = 'http://www.szse.cn/disclosure/supervision/measure/index.html'
                response = self.session.get(self.SZSE_API_URL, params=params, timeout=30)
                
                if response.status_code != 200:
                    logger.warning(f"深交所API返回状态码: {response.status_code}")
                    break
                
                data = response.json()
                
                if not data or len(data) == 0:
                    break
                
                # 深交所API返回的是数组,第一个元素包含实际数据
                if 'data' not in data[0]:
                    break
                
                records = data[0]['data']
                if not records:
                    break
                
                for record in records:
                    all_records.append(self._parse_szse_record(record))
                
                # 检查是否还有更多数据
                if len(records) < page_size:
                    break
                
                page_no += 1
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"采集深交所第{page_no}页失败: {e}")
                break
        
        if not all_records:
            return pd.DataFrame()
        
        return pd.DataFrame(all_records)
    
    def _parse_szse_record(self, record: Dict) -> Dict:
        """解析深交所单条记录"""
        stock_code = record.get('gkxx_gsdm', '')
        stock_name = record.get('gkxx_gsjc', '')
        measure = record.get('gkxx_jgcs', '')
        pub_date = record.get('gkxx_gdrq', '')
        
        # 从hjnr字段提取PDF链接
        hjnr = record.get('hjnr', '')
        doc_url = ''
        if hjnr:
            # hjnr包含HTML,需要解析
            soup = BeautifulSoup(hjnr, 'html.parser')
            link = soup.find('a')
            if link:
                # 深交所的PDF链接在encode-open属性中
                pdf_path = link.get('encode-open', '')
                if pdf_path:
                    if not pdf_path.startswith('http'):
                        doc_url = self.SZSE_BASE_URL + pdf_path
                    else:
                        doc_url = pdf_path
        
        # 生成标题
        title = f"{stock_name}({stock_code}) - {measure}"
        
        # 生成ID
        news_id = self._generate_id(f"szse_{stock_code}_{measure}_{pub_date}")
        
        return {
            'news_id': news_id,
            'title': title,
            'content': '',  # 需要下载PDF才能获取
            'date': pub_date,
            'source': 'szse',
            'category': '监管措施',
            'url': doc_url,
            'related_stocks': f"{stock_code}.{stock_name}" if stock_code else '',
            'stock_code': stock_code,
            'stock_name': stock_name,
            'keywords': measure_type,
        }
    
    def _generate_id(self, content: str) -> str:
        """生成唯一ID"""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _standardize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化输出"""
        for col in self.STANDARD_FIELDS:
            if col not in df.columns:
                df[col] = ''
        
        return df[self.STANDARD_FIELDS]


# 便捷函数

def get_official_exchange_news(
    start_date: str,
    end_date: str,
    exchanges: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    获取交易所官方公告
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        exchanges: 交易所列表
    
    Returns:
        公告DataFrame
    """
    crawler = OfficialExchangeNewsCrawler()
    return crawler.collect(
        start_date=start_date,
        end_date=end_date,
        exchanges=exchanges
    )
