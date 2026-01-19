"""
东方财富事件数据采集器

辅助数据源，用于获取结构化标签：
- 并购重组详情（交易金额、标的资产等）
- 违规处罚详情（处罚原因、罚款金额等）
- 大宗交易/龙虎榜数据

用途：
1. 作为训练模型的Label（标签）
2. 与巨潮PDF通过 股票代码+日期 对齐
"""

import re
import time
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

from .base_event import (
    BaseEventCollector,
    EventDocument,
    EventType,
    EventSource
)
from ..request_utils import safe_request, RequestDisguiser

logger = logging.getLogger(__name__)


# 东方财富数据中心API配置
EASTMONEY_APIS = {
    # 并购重组
    'merger': {
        'url': 'https://datacenter-web.eastmoney.com/api/data/v1/get',
        'params': {
            'reportName': 'RPT_MERGER_ACQUISITION',
            'columns': 'ALL',
            'source': 'WEB',
            'client': 'WEB',
            'pageSize': 50,
        },
        'name': '并购重组'
    },
    # 违规处罚
    'penalty': {
        'url': 'https://datacenter-web.eastmoney.com/api/data/v1/get',
        'params': {
            'reportName': 'RPT_VIOLATION_PUNISHMENT',
            'columns': 'ALL',
            'source': 'WEB',
            'client': 'WEB',
            'pageSize': 50,
        },
        'name': '违规处罚'
    },
    # 股权质押
    'pledge': {
        'url': 'https://datacenter-web.eastmoney.com/api/data/v1/get',
        'params': {
            'reportName': 'RPT_PLEDGE_DETAIL',
            'columns': 'ALL',
            'source': 'WEB',
            'client': 'WEB',
            'pageSize': 50,
        },
        'name': '股权质押'
    },
    # 高管增减持
    'executive_trade': {
        'url': 'https://datacenter-web.eastmoney.com/api/data/v1/get',
        'params': {
            'reportName': 'RPT_EXECUTIVE_TRADE',
            'columns': 'ALL',
            'source': 'WEB',
            'client': 'WEB',
            'pageSize': 50,
        },
        'name': '高管增减持'
    },
}

# 东方财富网页爬取配置（备选）
EASTMONEY_WEB = {
    'merger': {
        'url': 'https://data.eastmoney.com/bgcz/czbg.html',
        'name': '并购重组'
    },
    'penalty': {
        'url': 'https://data.eastmoney.com/sifa/wgcf.html',
        'name': '违规处罚'
    },
}


class EastMoneyEventCollector(BaseEventCollector):
    """
    东方财富事件采集器
    
    核心功能：
    1. 获取结构化事件数据（表格形式）
    2. 提取关键标签（金额、原因等）
    3. 生成metadata.csv用于与PDF对齐
    """
    
    SOURCE = EventSource.EASTMONEY
    
    def __init__(self):
        super().__init__()
        self._disguiser = RequestDisguiser()
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        event_types: Optional[List[str]] = None,
        stock_codes: Optional[List[str]] = None,
        download_pdf: bool = False,  # 东财不下载PDF
        max_pages: int = 20,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集东方财富事件数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            event_types: 事件类型列表
            stock_codes: 股票代码列表（可选）
            max_pages: 每个类别最大页数
            
        Returns:
            结构化事件数据DataFrame
        """
        # 标准化日期
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)
        
        # 确定事件类型
        if event_types is None:
            event_types = ['merger', 'penalty']
        
        all_events = []
        
        for event_type in event_types:
            logger.info(f"\n采集东财{event_type}数据...")
            
            # 优先使用API
            events = self._collect_via_api(
                event_type=event_type,
                start_date=start_date,
                end_date=end_date,
                stock_codes=stock_codes,
                max_pages=max_pages
            )
            
            # 如果API失败，尝试网页爬取
            if not events:
                events = self._collect_via_web(
                    event_type=event_type,
                    start_date=start_date,
                    end_date=end_date,
                    max_pages=max_pages
                )
            
            if events:
                all_events.extend(events)
                logger.info(f"东财{event_type}: 采集到 {len(events)} 条")
        
        # 保存元数据
        if all_events:
            self._save_metadata(all_events, f"eastmoney_{datetime.now().strftime('%Y%m%d')}.jsonl")
        
        return self.to_dataframe(all_events)
    
    def _collect_via_api(
        self,
        event_type: str,
        start_date: str,
        end_date: str,
        stock_codes: Optional[List[str]],
        max_pages: int
    ) -> List[EventDocument]:
        """通过API采集"""
        events = []
        
        if event_type not in EASTMONEY_APIS:
            return events
        
        config = EASTMONEY_APIS[event_type]
        
        for page in range(1, max_pages + 1):
            params = config['params'].copy()
            params['pageNumber'] = page
            
            # 添加日期过滤
            params['filter'] = f"(NOTICE_DATE>='{start_date}')(NOTICE_DATE<='{end_date}')"
            
            # 股票代码过滤
            if stock_codes:
                codes_filter = ','.join([f"'{c}'" for c in stock_codes])
                params['filter'] += f"(SECURITY_CODE in ({codes_filter}))"
            
            try:
                response = safe_request(
                    config['url'],
                    params=params,
                    headers=self._disguiser.get_json_headers(),
                    timeout=15
                )
                
                if not response:
                    break
                
                data = response.json()
                
                if not data.get('success'):
                    logger.warning(f"API返回失败: {data.get('message')}")
                    break
                
                result = data.get('result', {})
                records = result.get('data', [])
                
                if not records:
                    break
                
                for record in records:
                    event = self._parse_api_record(record, event_type)
                    if event:
                        events.append(event)
                
                # 检查是否还有下一页
                total_pages = result.get('pages', 1)
                if page >= total_pages:
                    break
                
                time.sleep(0.3)
                
            except Exception as e:
                logger.warning(f"API请求失败: {e}")
                break
        
        return events
    
    def _parse_api_record(self, record: Dict, event_type: str) -> Optional[EventDocument]:
        """解析API返回的记录"""
        try:
            # 通用字段映射
            ts_code = record.get('SECURITY_CODE', '') or record.get('SECUCODE', '')
            stock_name = record.get('SECURITY_NAME_ABBR', '') or record.get('SECURITY_NAME', '')
            ann_date = record.get('NOTICE_DATE', '') or record.get('ANN_DATE', '')
            title = record.get('TITLE', '') or record.get('PLAN_NAME', '') or ''
            
            # 标准化ts_code
            if ts_code and '.' not in ts_code:
                ts_code = self._code_to_tscode(ts_code)
            
            # 标准化日期
            if ann_date:
                if 'T' in ann_date:
                    ann_date = ann_date.split('T')[0]
                elif ' ' in ann_date:
                    ann_date = ann_date.split(' ')[0]
            
            # 提取结构化标签
            labels = self._extract_labels(record, event_type)
            
            # 生成摘要
            summary = self._generate_summary(record, event_type)
            
            # 生成ID
            event_id = self._generate_id(ts_code, title or summary, ann_date)
            
            return EventDocument(
                id=event_id,
                ts_code=ts_code,
                stock_name=stock_name,
                event_type=event_type,
                event_subtype=self._extract_event_subtype(title, event_type),
                title=title or summary,
                summary=summary,
                ann_date=ann_date,
                effective_date='',
                source=self.SOURCE.value,
                url='',
                pdf_url='',
                local_path='',
                labels=labels,
                is_correction=False,
                original_id='',
                create_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
        except Exception as e:
            logger.debug(f"解析记录失败: {e}")
            return None
    
    def _extract_labels(self, record: Dict, event_type: str) -> Dict[str, Any]:
        """提取结构化标签"""
        labels = {}
        
        if event_type == 'merger':
            # 并购重组标签
            labels['deal_amount'] = record.get('DEAL_AMOUNT', '')  # 交易金额
            labels['target_asset'] = record.get('TARGET_ASSET', '')  # 标的资产
            labels['acquirer'] = record.get('ACQUIRER', '')  # 收购方
            labels['deal_type'] = record.get('DEAL_TYPE', '')  # 交易类型
            labels['progress'] = record.get('PROGRESS', '')  # 进度
            labels['pay_method'] = record.get('PAY_METHOD', '')  # 支付方式
            
        elif event_type == 'penalty':
            # 违规处罚标签
            labels['penalty_amount'] = record.get('PENALTY_AMOUNT', '')  # 罚款金额
            labels['penalty_reason'] = record.get('PENALTY_REASON', '')  # 处罚原因
            labels['penalty_type'] = record.get('PENALTY_TYPE', '')  # 处罚类型
            labels['regulator'] = record.get('REGULATOR', '')  # 监管机构
            labels['related_person'] = record.get('RELATED_PERSON', '')  # 涉及人员
            
        elif event_type == 'pledge':
            # 股权质押标签
            labels['pledge_ratio'] = record.get('PLEDGE_RATIO', '')  # 质押比例
            labels['pledge_shares'] = record.get('PLEDGE_SHARES', '')  # 质押股数
            labels['pledger'] = record.get('PLEDGER', '')  # 质押人
            labels['pledgee'] = record.get('PLEDGEE', '')  # 质权人
            
        # 清理空值
        labels = {k: v for k, v in labels.items() if v}
        
        return labels
    
    def _generate_summary(self, record: Dict, event_type: str) -> str:
        """生成事件摘要"""
        parts = []
        
        stock_name = record.get('SECURITY_NAME_ABBR', '') or record.get('SECURITY_NAME', '')
        
        if event_type == 'merger':
            deal_type = record.get('DEAL_TYPE', '并购重组')
            amount = record.get('DEAL_AMOUNT', '')
            if amount:
                parts.append(f"{stock_name}{deal_type}，交易金额{amount}")
            else:
                parts.append(f"{stock_name}{deal_type}")
                
        elif event_type == 'penalty':
            penalty_type = record.get('PENALTY_TYPE', '监管处罚')
            amount = record.get('PENALTY_AMOUNT', '')
            if amount:
                parts.append(f"{stock_name}收到{penalty_type}，罚款{amount}")
            else:
                parts.append(f"{stock_name}收到{penalty_type}")
        
        return ''.join(parts) or stock_name
    
    def _collect_via_web(
        self,
        event_type: str,
        start_date: str,
        end_date: str,
        max_pages: int
    ) -> List[EventDocument]:
        """通过网页爬取（备选方案）"""
        events = []
        
        if event_type not in EASTMONEY_WEB:
            return events
        
        config = EASTMONEY_WEB[event_type]
        logger.info(f"使用网页爬取方式采集: {config['name']}")
        
        try:
            response = safe_request(
                config['url'],
                headers=self._disguiser.get_headers(),
                timeout=15
            )
            
            if not response:
                return events
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 解析表格数据
            table = soup.select_one('table.default_table, table#dt_1')
            if not table:
                return events
            
            rows = table.select('tbody tr')
            
            for row in rows[:100]:  # 限制数量
                cells = row.select('td')
                if len(cells) < 5:
                    continue
                
                try:
                    # 根据事件类型解析
                    if event_type == 'merger':
                        ts_code = cells[1].get_text(strip=True)
                        stock_name = cells[2].get_text(strip=True)
                        title = cells[3].get_text(strip=True)
                        ann_date = cells[4].get_text(strip=True)
                    elif event_type == 'penalty':
                        ts_code = cells[0].get_text(strip=True)
                        stock_name = cells[1].get_text(strip=True)
                        title = cells[2].get_text(strip=True)
                        ann_date = cells[3].get_text(strip=True)
                    else:
                        continue
                    
                    # 日期过滤
                    if ann_date < start_date or ann_date > end_date:
                        continue
                    
                    if '.' not in ts_code:
                        ts_code = self._code_to_tscode(ts_code)
                    
                    event_id = self._generate_id(ts_code, title, ann_date)
                    
                    events.append(EventDocument(
                        id=event_id,
                        ts_code=ts_code,
                        stock_name=stock_name,
                        event_type=event_type,
                        event_subtype='',
                        title=title,
                        summary=title,
                        ann_date=ann_date,
                        effective_date='',
                        source=self.SOURCE.value,
                        url=config['url'],
                        pdf_url='',
                        local_path='',
                        labels={},
                        is_correction=False,
                        original_id='',
                        create_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ))
                    
                except Exception as e:
                    logger.debug(f"解析行失败: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"网页爬取失败: {e}")
        
        return events
    
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
        """标准化日期格式"""
        date_str = date_str.replace('-', '').replace('/', '')
        if len(date_str) == 8:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return date_str


def get_eastmoney_events(
    start_date: str,
    end_date: str,
    event_types: Optional[List[str]] = None,
    stock_codes: Optional[List[str]] = None,
    max_pages: int = 20
) -> pd.DataFrame:
    """
    采集东方财富事件数据
    
    Args:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期
        event_types: 事件类型列表 ('merger', 'penalty', 'pledge')
        stock_codes: 股票代码列表（可选）
        max_pages: 每个类别最大页数
        
    Returns:
        结构化事件数据DataFrame
        
    Example:
        >>> # 采集并购重组数据（含结构化标签）
        >>> df = get_eastmoney_events('20240101', '20241231', event_types=['merger'])
        >>> print(df[['ts_code', 'title', 'labels']])
    """
    collector = EastMoneyEventCollector()
    return collector.collect(
        start_date=start_date,
        end_date=end_date,
        event_types=event_types,
        stock_codes=stock_codes,
        max_pages=max_pages
    )


def align_events_with_pdf(
    eastmoney_df: pd.DataFrame,
    cninfo_df: pd.DataFrame,
    tolerance_days: int = 3
) -> pd.DataFrame:
    """
    将东财结构化数据与巨潮PDF对齐
    
    通过 股票代码 + 日期（容差范围内） 进行匹配
    
    Args:
        eastmoney_df: 东财数据
        cninfo_df: 巨潮数据
        tolerance_days: 日期容差天数
        
    Returns:
        对齐后的数据（包含labels和local_path）
    """
    if eastmoney_df.empty or cninfo_df.empty:
        return eastmoney_df
    
    # 转换日期格式
    eastmoney_df['ann_date'] = pd.to_datetime(eastmoney_df['ann_date'])
    cninfo_df['ann_date'] = pd.to_datetime(cninfo_df['ann_date'])
    
    aligned_records = []
    
    for _, em_row in eastmoney_df.iterrows():
        ts_code = em_row['ts_code']
        ann_date = em_row['ann_date']
        
        # 查找匹配的巨潮记录
        mask = (
            (cninfo_df['ts_code'] == ts_code) &
            (abs((cninfo_df['ann_date'] - ann_date).dt.days) <= tolerance_days)
        )
        
        matches = cninfo_df[mask]
        
        if not matches.empty:
            # 取最近的匹配
            best_match = matches.iloc[0]
            em_row['local_path'] = best_match['local_path']
            em_row['pdf_url'] = best_match['pdf_url']
        
        aligned_records.append(em_row)
    
    return pd.DataFrame(aligned_records)
