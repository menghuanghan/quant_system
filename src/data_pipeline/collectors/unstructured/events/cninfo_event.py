"""
巨潮资讯事件采集器

核心数据源，专门采集事件驱动型公告：
- 并购重组
- 违规处罚
- 实控人变更
- 重大合同

采集策略：通过精准的"分类ID"定向请求，避免全量爬取
"""

import time
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from urllib.parse import urljoin

import pandas as pd

from .base_event import (
    BaseEventCollector,
    EventDocument,
    EventType,
    EventSource
)
from ..request_utils import safe_request, RequestDisguiser

logger = logging.getLogger(__name__)


# 巨潮资讯事件分类字典（核心配置）
EVENT_CATEGORIES = {
    # === 极高价值 (Alpha 核心) ===
    EventType.MERGER.value: {
        'id': 'category_scgk_szsh;category_bcgz_szsh;',  # 收购兼并 + 并购重组
        'keywords': ['预案', '草案', '报告书', '审核', '回复', '重组', '收购', '合并'],
        'name': '并购重组'
    },
    EventType.PENALTY.value: {
        'id': 'category_jggz_szsh;',  # 监管关注（包含处罚、警示函等）
        'keywords': [],  # 处罚类全要
        'name': '违规处罚'
    },
    EventType.EQUITY_CHANGE.value: {
        'id': 'category_gqbd_szsh;',  # 股权变动
        'keywords': ['控制人', '举牌', '要约', '权益变动'],
        'name': '权益变动'
    },
    EventType.CONTROL_CHANGE.value: {
        'id': 'category_gqbd_szsh;',  # 股权变动（子集）
        'keywords': ['实际控制人', '控股股东变更', '控制权'],
        'name': '实控人变更'
    },
    
    # === 中等价值 ===
    EventType.MAJOR_CONTRACT.value: {
        'id': 'category_rcjy_szsh;',  # 日常经营
        'keywords': ['重大合同', '中标', '战略合作', '订单', '框架协议'],
        'name': '重大合同'
    },
    EventType.LITIGATION.value: {
        'id': 'category_sszdsx_szsh;',  # 诉讼仲裁
        'keywords': ['诉讼', '仲裁', '起诉', '索赔'],
        'name': '诉讼仲裁'
    },
}


class CninfoEventCollector(BaseEventCollector):
    """
    巨潮资讯事件采集器
    
    核心逻辑：
    1. 使用精准的category ID定向请求
    2. 通过关键词二次过滤
    3. 下载PDF原文到本地
    4. 支持历史回溯（3-5年）
    """
    
    SOURCE = EventSource.CNINFO
    
    # API 端点
    BASE_URL = "http://www.cninfo.com.cn"
    SEARCH_API = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    PDF_BASE_URL = "http://static.cninfo.com.cn/"
    
    def __init__(self, use_proxy: bool = False):
        super().__init__()
        self._use_proxy = use_proxy
        self._disguiser = RequestDisguiser()
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        event_types: Optional[List[str]] = None,
        stock_codes: Optional[List[str]] = None,
        download_pdf: bool = True,
        max_pages: int = 50,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集事件数据
        
        Args:
            start_date: 开始日期 (YYYYMMDD 或 YYYY-MM-DD)
            end_date: 结束日期
            event_types: 事件类型列表，默认全部
            stock_codes: 股票代码列表（可选，留空则全市场）
            download_pdf: 是否下载PDF
            max_pages: 每个类别最大采集页数
            
        Returns:
            事件数据DataFrame
        """
        # 标准化日期格式
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)
        
        # 确定采集的事件类型
        if event_types is None:
            event_types = [
                EventType.MERGER.value,
                EventType.PENALTY.value,
                EventType.CONTROL_CHANGE.value,
                EventType.MAJOR_CONTRACT.value,
            ]
        
        # 加载已有ID去重
        existing_ids = self._load_existing_ids()
        logger.info(f"已有 {len(existing_ids)} 条事件记录")
        
        all_events = []
        
        for event_type in event_types:
            if event_type not in EVENT_CATEGORIES:
                logger.warning(f"未配置的事件类型: {event_type}")
                continue
            
            config = EVENT_CATEGORIES[event_type]
            logger.info(f"\n采集{config['name']}事件...")
            
            events = self._collect_event_type(
                event_type=event_type,
                category_id=config['id'],
                keywords=config['keywords'],
                start_date=start_date,
                end_date=end_date,
                stock_codes=stock_codes,
                download_pdf=download_pdf,
                max_pages=max_pages,
                existing_ids=existing_ids
            )
            
            if events:
                all_events.extend(events)
                # 保存元数据
                self._save_metadata(events, f"{event_type}_{datetime.now().strftime('%Y%m%d')}.jsonl")
                logger.info(f"{config['name']}: 采集到 {len(events)} 条")
        
        logger.info(f"\n事件采集完成，共 {len(all_events)} 条")
        return self.to_dataframe(all_events)
    
    def _collect_event_type(
        self,
        event_type: str,
        category_id: str,
        keywords: List[str],
        start_date: str,
        end_date: str,
        stock_codes: Optional[List[str]],
        download_pdf: bool,
        max_pages: int,
        existing_ids: set
    ) -> List[EventDocument]:
        """采集单个事件类型"""
        events = []
        page_num = 1
        
        while page_num <= max_pages:
            # 构建请求参数
            params = self._build_params(
                category=category_id,
                start_date=start_date,
                end_date=end_date,
                stock=stock_codes[0] if stock_codes and len(stock_codes) == 1 else None,
                page_num=page_num
            )
            
            # 发送请求
            response = safe_request(
                self.SEARCH_API,
                method='POST',
                data=params,
                headers=self._get_headers(),
                use_proxy=self._use_proxy,
                rate_limit=True
            )
            
            if response is None:
                logger.warning(f"请求失败，页码: {page_num}")
                break
            
            try:
                data = response.json()
            except Exception as e:
                logger.warning(f"解析JSON失败: {e}")
                break
            
            announcements = data.get('announcements', [])
            if not announcements:
                break
            
            # 处理每条公告
            for ann in announcements:
                event = self._parse_announcement(
                    ann,
                    event_type=event_type,
                    keywords=keywords,
                    existing_ids=existing_ids,
                    download_pdf=download_pdf
                )
                
                if event:
                    events.append(event)
                    existing_ids.add(event.id)
            
            # 检查是否还有下一页
            has_more = data.get('hasMore', False)
            if not has_more:
                break
            
            page_num += 1
            time.sleep(0.3)  # 控制请求频率
        
        return events
    
    def _build_params(
        self,
        category: str,
        start_date: str,
        end_date: str,
        stock: Optional[str] = None,
        page_num: int = 1,
        page_size: int = 30
    ) -> Dict[str, Any]:
        """构建查询参数"""
        params = {
            'pageNum': page_num,
            'pageSize': page_size,
            'tabName': 'fulltext',
            'isHLtitle': 'true',
            'category': category,  # 核心：注入分类ID
            'seDate': f"{start_date}~{end_date}",
        }
        
        if stock:
            # 清理代码格式
            if '.' in stock:
                stock = stock.split('.')[0]
            params['stock'] = stock.zfill(6)
        
        return params
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = self._disguiser.get_json_headers({
            'Referer': 'http://www.cninfo.com.cn/new/commonUrl/pageOfSearch',
            'Origin': 'http://www.cninfo.com.cn',
        })
        return headers
    
    def _parse_announcement(
        self,
        ann: Dict,
        event_type: str,
        keywords: List[str],
        existing_ids: set,
        download_pdf: bool
    ) -> Optional[EventDocument]:
        """解析单条公告"""
        try:
            title = ann.get('announcementTitle', '')
            ann_date = ann.get('announcementTime', '')
            sec_code = ann.get('secCode', '')
            sec_name = ann.get('secName', '')
            pdf_url = ann.get('adjunctUrl', '')
            ann_id = ann.get('announcementId', '')
            
            # 转换日期
            if isinstance(ann_date, (int, float)):
                ann_date = datetime.fromtimestamp(ann_date / 1000).strftime('%Y-%m-%d')
            
            # 关键词过滤
            if keywords:
                if not any(kw in title for kw in keywords):
                    return None
            
            # 生成ts_code
            ts_code = self._code_to_tscode(sec_code)
            
            # 生成唯一ID
            event_id = self._generate_id(ts_code, title, ann_date)
            
            # 去重检查
            if event_id in existing_ids:
                return None
            
            # 构建完整PDF URL
            if pdf_url and not pdf_url.startswith('http'):
                pdf_url = urljoin(self.PDF_BASE_URL, pdf_url)
            
            # 子类型和更正检测
            event_subtype = self._extract_event_subtype(title, event_type)
            is_correction = self._detect_correction(title)
            
            # PDF路径
            local_path = ""
            if download_pdf and pdf_url:
                save_path = self._get_pdf_path(event_type, ts_code, title, ann_date)
                if self._download_pdf(pdf_url, save_path):
                    local_path = str(save_path)
            
            return EventDocument(
                id=event_id,
                ts_code=ts_code,
                stock_name=sec_name,
                event_type=event_type,
                event_subtype=event_subtype,
                title=title,
                summary='',  # 摘要可后续从PDF提取
                ann_date=ann_date,
                effective_date='',
                source=self.SOURCE.value,
                url=f"{self.BASE_URL}/new/disclosure/detail?announcementId={ann_id}",
                pdf_url=pdf_url,
                local_path=local_path,
                labels={},
                is_correction=is_correction,
                original_id=str(ann_id),
                create_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
        except Exception as e:
            logger.debug(f"解析公告失败: {e}")
            return None
    
    @staticmethod
    def _code_to_tscode(code: str) -> str:
        """股票代码转换"""
        code = str(code).zfill(6)
        if code.startswith(('0', '2', '3')):
            return f"{code}.SZ"
        elif code.startswith(('6', '9')):
            return f"{code}.SH"
        elif code.startswith(('4', '8')):
            return f"{code}.BJ"
        return f"{code}.SZ"
    
    @staticmethod
    def _normalize_date(date_str: str) -> str:
        """标准化日期格式为 YYYY-MM-DD"""
        date_str = date_str.replace('-', '').replace('/', '')
        if len(date_str) == 8:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return date_str


def get_cninfo_events(
    start_date: str,
    end_date: str,
    event_types: Optional[List[str]] = None,
    stock_codes: Optional[List[str]] = None,
    download_pdf: bool = True,
    max_pages: int = 50
) -> pd.DataFrame:
    """
    采集巨潮事件数据
    
    Args:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期
        event_types: 事件类型列表，可选：
            - 'merger': 并购重组
            - 'penalty': 违规处罚
            - 'control_change': 实控人变更
            - 'contract': 重大合同
        stock_codes: 股票代码列表（可选）
        download_pdf: 是否下载PDF原文
        max_pages: 每个类别最大采集页数
        
    Returns:
        事件数据DataFrame
        
    Example:
        >>> # 采集近一年的并购重组事件
        >>> df = get_cninfo_events('20240101', '20241231', event_types=['merger'])
        
        >>> # 采集特定股票的所有事件
        >>> df = get_cninfo_events('20240101', '20241231', stock_codes=['000001'])
    """
    collector = CninfoEventCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        event_types=event_types,
        stock_codes=stock_codes,
        download_pdf=download_pdf,
        max_pages=max_pages
    )
