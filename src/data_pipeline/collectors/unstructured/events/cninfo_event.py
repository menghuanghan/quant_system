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
        'id': 'category_scgk_szsh;category_bcgz_szsh;',
        'keywords': [
            '收购', '出售', '重组', '合并', '吸收', '置换', '购买资产',
            '交易预案', '交易草案', '交易报告书', '重组预案', '重组草案', '重组报告书',
            '发行股份', '资产置换', '要约收购', '股权转让', '标的资产', '过户'
        ],
        'excludes': [
            '取消', '终止', '说明会', '核查意见', '法律意见', '回复', '反馈', '摘要'
        ],
        'name': '并购重组'
    },
    EventType.PENALTY.value: {
        'id': 'category_jggz_szsh;',
        'keywords': [
            '行政处罚', '立案调查', '立案告知', '市场禁入', '公开谴责', '通报批评',
            '罚款', '违规行为', '纪律处分', '监管措施', '刑事强制措施', '逮捕', '拘留',
            '认定为不适当人选'
        ],
        'excludes': [
            '监管函', '关注函', '问询函', '回复', '核查意见', '整改报告', 
            '风险提示', '说明', '意见书', '复核'
        ],
        'name': '违规处罚'
    },
    EventType.EQUITY_CHANGE.value: {
        'id': 'category_gqbd_szsh;',
        'keywords': [
            '权益变动', '股份变动', '增持', '减持', '持股变动',
            '简式权益变动', '详式权益变动', '举牌', '大宗交易', '集中竞价',
            '股份转让', '增持计划', '减持计划', '增持结果', '减持结果'
        ],
        'excludes': [
            '质押', '解押', '解除质押', '回购', '注销', '激励', '员工持股',
            '核查意见', '法律意见', '进展'
        ],
        'name': '权益变动'
    },
    EventType.CONTROL_CHANGE.value: {
        'id': 'category_gqbd_szsh;',
        'keywords': [
            '实际控制人', '控股股东', '控制权', '变更', '易主',
            '一致行动', '表决权', '无实际控制人'
        ],
        'excludes': [
            '未变更', '不发生变更', '稳定', '说明', '回复', '核查意见'
        ],
        'name': '实控人变更'
    },
    
    # === 中等价值 ===
    EventType.MAJOR_CONTRACT.value: {
        'id': 'category_rcjy_szsh;',
        'keywords': [
            '合同', '协议', '中标', '订单', '战略合作', '承接',
            '销售合同', '采购合同', '工程合同', '项目合同', '经营合同',
            '框架协议', '备忘录', '意向书', '达成合作'
        ],
        'excludes': [
            '解除', '终止', '日常关联交易', '补充协议', '进展'
        ],
        'name': '重大合同'
    },
    EventType.LITIGATION.value: {
        'id': 'category_sszdsx_szsh;',
        'keywords': [
            '诉讼', '仲裁', '起诉', '判决', '裁决', '应诉', '执行',
            '法院', '原告', '被告', '立案', '撤诉', '和解', '法律文书'
        ],
        'excludes': [
            '进展', '以前年度', '累计'
        ],
        'name': '诉讼仲裁'
    },
    EventType.BANKRUPTCY.value: {
        'id': 'category_pccz_szsh;', 
        'keywords': [
            '破产', '重整', '预重整', '清算', '债权人会议', '管理人',
            '重整计划', '被申请破产', '招募'
        ],
        'excludes': [
            '进展', '风险提示'
        ],
        'name': '破产重整'
    },
    EventType.SUSPENSION.value: {
        'id': 'category_tfptzb_szsh;',
        'keywords': [
            '停牌', '复牌', '暂停上市', '恢复上市', '临时停牌'
        ],
        'excludes': [
            '进展', '可能'
        ],
        'name': '停复牌'
    },
}


class CninfoEventCollector(BaseEventCollector):
    """
    巨潮资讯事件采集器
    
    核心逻辑：
    1. 使用精准的category ID定向请求
    2. 通过关键词二次过滤
    3. 支持历史回溯（3-5年）
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
        max_pages: int = 200,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集事件数据
        
        Args:
            start_date: 开始日期 (YYYYMMDD 或 YYYY-MM-DD)
            end_date: 结束日期
            event_types: 事件类型列表，默认全部
            stock_codes: 股票代码列表（可选，留空则全市场）
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
                EventType.EQUITY_CHANGE.value,
                EventType.CONTROL_CHANGE.value,
                EventType.MAJOR_CONTRACT.value,
                EventType.LITIGATION.value,
                EventType.BANKRUPTCY.value,
                EventType.SUSPENSION.value,
            ]
        
        # 使用内存set去重
        existing_ids = set()
        
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
                excludes=config['excludes'],
                start_date=start_date,
                end_date=end_date,
                stock_codes=stock_codes,
                max_pages=max_pages,
                existing_ids=existing_ids
            )
            
            if events:
                all_events.extend(events)
                logger.info(f"{config['name']}: 采集到 {len(events)} 条")
        
        logger.info(f"\n事件采集完成，共 {len(all_events)} 条")
        return self.to_dataframe(all_events)
    
    def _collect_event_type(
        self,
        event_type: str,
        category_id: str,
        keywords: List[str],
        excludes: List[str],
        start_date: str,
        end_date: str,
        stock_codes: Optional[List[str]],
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
                    excludes=excludes,
                    existing_ids=existing_ids
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
        excludes: List[str],
        existing_ids: set
    ) -> Optional[EventDocument]:
        """解析单条公告"""
        try:
            title = ann.get('announcementTitle', '')
            ann_date = ann.get('announcementTime', '')
            sec_code = ann.get('secCode', '')
            sec_name = ann.get('secName', '')
            pdf_url = ann.get('adjunctUrl', '')
            ann_id = ann.get('announcementId', '')
            
            # 验证必需字段
            if not title or not sec_code or not ann_date:
                logger.debug(f"跳过无效公告: title={title}, code={sec_code}, date={ann_date}")
                return None
            
            # 过滤无效股票代码
            sec_code_str = str(sec_code).strip()
            if not sec_code_str or sec_code_str == '0' or sec_code_str == '000000':
                logger.debug(f"跳过无效股票代码: {sec_code_str}")
                return None
            
            # 转换日期
            if isinstance(ann_date, (int, float)):
                ann_date = datetime.fromtimestamp(ann_date / 1000).strftime('%Y-%m-%d')
            
            # 关键词过滤
            if keywords:
                if not any(kw in title for kw in keywords):
                    return None

            # 排除词过滤
            if excludes:
                if any(ex in title for ex in excludes):
                    return None
            
            # 生成ts_code
            ts_code = self._code_to_tscode(sec_code_str)
            
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
            
            return EventDocument(
                id=event_id,
                ts_code=ts_code,
                stock_name=sec_name if sec_name else "",
                event_type=event_type,
                event_subtype=event_subtype,
                title=title,
                date=ann_date,
                effective_date='',
                source=self.SOURCE.value,
                url=f"{self.BASE_URL}/new/disclosure/detail?announcementId={ann_id}",
                pdf_url=pdf_url,
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
    max_pages: int = 200
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
        max_pages=max_pages
    )
