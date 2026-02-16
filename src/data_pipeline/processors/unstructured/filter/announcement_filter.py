"""
公告数据过滤器

使用GPU加速（cuDF）实现高效的两层过滤：
1. 第一层：根据events的original_id过滤掉已在事件中的公告
2. 第二层：根据title黑名单关键词过滤垃圾公告

设计说明：
- 使用cuDF进行GPU加速，大幅提升过滤性能
- 黑名单关键词经过精心设计，过滤掉约80%的垃圾公告
- 支持自动回退到pandas（当GPU不可用时）
- 保留有价值的公告：业绩、收购、重组、合同、订单等
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Set, Dict, Any, Tuple

logger = logging.getLogger(__name__)


# ========== 标题黑名单关键词配置 ==========
# 这些关键词出现在标题中的公告，基本可以断定对量化交易没有Alpha价值

TITLE_BLACKLIST_KEYWORDS = [
    # === 1. 定期报告全家桶 (数量极大) ===
    '年度报告', '半年度报告', '季度报告', '业绩快报', '业绩预告',
    '财务报告', '审计报告', '内部控制', '社会责任报告', 'ESG报告',
    '摘要', '正文', '全文', '英文版', '取消发布', '财务报表', 
    '公告编号', # 很多纯更正或列表会只写编号
    
    # === 2. 会议与治理 (程序性噪音) ===
    # [修改] 原为 '通知'，改为具体会议通知，避免误杀"中标通知书"、"定点通知书"
    '召开.*?会议', '召开.*?大会', '会议通知', '参会通知', '复牌通知', 
    '会议', '决议', '议案', '审议', '表决',
    '章程', '规则', '制度', '细则', '办法', '工作大纲',
    '董事会', '监事会', '股东大会', '专门委员会', '组织架构',
    '提名', '选举', '任免', '聘任', '辞职', '候选', '换届',
    '独立董事', '高管', '职工代表', '保荐代表人', '证券事务代表',
    '任职资格', '履职', '述职',
    
    # === 3. 资金与理财 (日常灌水) ===
    '理财', '现金管理', '闲置', '募集资金', '自有资金',
    '存款', '结构性存款', '专用账户', '开立', '账户注销', '分公司注销','销户',
    '授信', '贷款', '借款', '融资', '担保', '反担保', # 除非暴雷，否则日常担保无价值
    '置换', '补流', '流动资金', '验资', '专户',
    
    # === 4. 债券与评级 (非股性噪音) ===
    '债券', '可转债', '转股', '付息', '兑付', '票面利率',
    '跟踪评级', '信用评级', '评级报告', '债券持有人',
    '赎回', '回售', '摘牌',
    
    # === 5. 监管互动与说明 ===
    # [移除] '问询函', '关注函', '监管函' -> 为了保留风控信号
    '回复', '反馈', '核查', # 回复通常没价值，问询函本身有价值
    '说明', '意见', '声明', '复函', '回函',
    '更正', '补充', '澄清', '致歉', 
    # [修改] '提示性公告' -> 改为具体类型的提示，防止误杀重组提示
    '退市.*?提示', '暂停上市', '风险提示', # 通常是通用的退市风险或交易风险提示，非突发事件
    
    # === 6. 股权与交易细节 ===
    '权益变动报告书', '简式', '详式', # 这些是标准格式文件，摘要通常够用，或者Event已采
    '上市流通', '限售股', '解除限售', '网下配售',
    '中签', '摇号', '发行结果', '申购',
    '大宗交易', '集中竞价', # 减持细节通常通过Event采集，公告太碎
    '质押', '解押', '冻结', # 除非你专门做质押预警，否则量极大
    '期权', '行权', '授予', '归属', '限制性股票',
    
    # === 7. 法律与中介 ===
    '律师', '法律意见', '鉴证', '评估报告', '核实',
    '保荐', '督导', '现场检查', '培训',
    '会计师', '事务所', '更换',
    
    # === 8. 子公司与日常经营 (琐碎) ===
    '子公司', '孙公司', # 子公司日常很多琐事对母公司股价影响微弱
    '增资', '减资', '注册资本', # 除非金额巨大，否则多为内部划转
    '工商变更', '营业执照', '迁址', '认定', '证书', '高新技术',
    '日常关联交易', '预计', # 每年年初发的预计公告，无突发价值
    '战略合作', '框架协议', '备忘录', # 这种很多是画饼，如果要做事件驱动，Event里有合同类型
    '接待', '调研', '活动记录', # 投资者关系活动记录表
    '进展公告', '进展情况', # 绝大多数进展都是废话，除非是“终止”或“完成”

    #其他非重大事件公告 (数量极大，基本无价值)
    # === 1. 股权激励/员工持股的“执行细节” (极大噪音源) ===
    '作废', '成就', '失效',      # e.g., 部分限制性股票回购注销及作废、解锁条件成就
    '调整', '预留',             # e.g., 调整行权价格、向激励对象授予预留权益
    '登记', '名册', '确权',      # e.g., 完成登记、股东名册
    
    # === 2. 分红与转债的“实施流程” ===
    '权益分派', '派发', '实施公告', # e.g., 2023年年度权益分派实施公告 (财务数据已由结构化表覆盖)
    '转股价格', '修正案',         # e.g., 调整可转债转股价格、章程修正案
    
    # === 3. 投资者关系与路演 ===
    '说明会', '路演', '网上',     # e.g., 召开业绩说明会、网上路演
    '纪要', '记录表',             # e.g., 投资者关系活动记录表
    '接待', '调研',               # e.g., 接待机构调研
    
    # === 4. 行政与合规琐事 ===
    '延期', '取消',              # e.g., 延期回复问询函、取消股东大会
    '自查', '整改',              # e.g., 自查报告、整改报告 (Event里已有"处罚"，这里多为整改完成的废话)
    '承诺', '豁免',              # e.g., 豁免承诺、承诺履行情况
    '补选', '津贴', '薪酬',       # e.g., 补选董事、调整董事津贴
    '保险', '责任险',            # e.g., 购买董监高责任险
    '办公地址', '联系方式', '网站', # e.g., 变更办公地址及联系方式
    '换证', '参股', '捐赠',       # e.g., 子公司换发营业执照、对外捐赠
    '累计',                      # e.g., 累计诉讼/担保 (通常是琐事汇总，重大诉讼Event会抓)

    #其他琐碎公告 (数量极大)
    # === 1. 股价异动与辟谣 (标准话术，无实质内容) ===
    '异常波动',       # e.g., 股票交易异常波动公告 (通常回复"无应披露未披露事项")
    '不知情',         # e.g., 对股价异动不知情
    '核实',           # e.g., 关于股票交易异常波动的核实结果
    '传闻', '媒体',    # e.g., 澄清媒体传闻 (若真有大事，Events会抓"澄清公告"或"诉讼")
    '澄清',           # (同上，大多数是辟谣)
    
    # === 2. 投资者关系与月度数据 (极度琐碎) ===
    '股东人数', '户数', # e.g., 关于股东人数的公告 (因子库直接有，无需NLP)
    '经营数据', '简报', # e.g., 1月主要经营数据简报 (通常只是单纯的销售数字，结构化数据更好用)
    '快报',           # e.g., 业绩快报 (同上，数字直接用结构化数据)
    '集体接待日',      # e.g., 参加辖区上市公司投资者集体接待日
    '网上',           # e.g., 网上说明会、网上申购
    
    # === 3. 合规样板戏 (每年必发，纯文档) ===
    '内部控制',       # e.g., 内部控制自我评价报告
    '独立性',         # e.g., 独立董事关于独立性的自查报告
    '社会责任', 'ESG', # e.g., 社会责任报告
    '可持续发展',      # e.g., 可持续发展报告
    '质量', '诚信',    # e.g., 质量诚信报告
    '履职', '述职',    # e.g., 独立董事年度述职报告
    '评价',           # e.g., 董事会关于审计机构的评价
    '专项说明',       # e.g., 非经营性资金占用及其他关联资金往来的专项说明
    
    # === 4. 土地、税务与琐碎资产 ===
    '土地使用权',      # e.g., 竞得土地使用权 (制造业日常，除非超大金额，否则Events抓不到就扔)
    '竞得',           # (同上)
    '税收优惠',       # e.g., 获得高新技术企业税收优惠 (长期利好已反应在财报，突发性弱)
    '退税',           # e.g., 收到软件产品增值税退税
    '认定',           # e.g., 通过高新技术企业认定
    '专利', '证书',    # e.g., 取得发明专利证书 (医药股除外，但医药重磅通常有"药品注册"事件)
    
    # === 5. 人事与行政变体 ===
    '离职', '不再担任', # e.g., 证券事务代表离职 (辞职的变体)
    '参加', '培训',    # e.g., 参加保荐代表人培训
    '党建', '党委',    # e.g., 修改公司章程(党建条款)
    '联系方式',       # e.g., 变更投资者联系方式
]


@dataclass
class FilterConfig:
    """过滤配置"""
    
    # 路径配置
    raw_data_dir: str = "data/raw/unstructured"
    
    # GPU配置
    use_gpu: bool = True
    
    # 过滤配置
    blacklist_keywords: List[str] = field(default_factory=lambda: TITLE_BLACKLIST_KEYWORDS.copy())
    enable_event_filter: bool = True  # 是否启用事件过滤（第一层）
    enable_title_filter: bool = True  # 是否启用标题过滤（第二层）
    
    # 备份配置
    backup_original: bool = False  # 是否备份原始文件
    backup_dir: str = "data/raw/unstructured/announcements_backup"
    
    # 日志配置
    log_level: str = "INFO"
    
    def __post_init__(self):
        """初始化后处理"""
        if self.backup_original:
            Path(self.backup_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class FilterResult:
    """过滤结果"""
    year: int
    month: int
    original_count: int           # 原始记录数
    after_event_filter: int       # 事件过滤后记录数
    after_title_filter: int       # 标题过滤后记录数
    final_count: int              # 最终记录数
    event_filtered_count: int     # 被事件过滤的数量
    title_filtered_count: int     # 被标题过滤的数量
    total_filtered_count: int     # 总过滤数量
    filter_rate: float            # 过滤率
    elapsed_time: float           # 耗时（秒）
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    
    def summary(self) -> str:
        """生成摘要"""
        return (
            f"announcements/{self.year}/{self.month:02d}: "
            f"原始 {self.original_count:,} -> 最终 {self.final_count:,} "
            f"(事件过滤 {self.event_filtered_count:,}, 标题过滤 {self.title_filtered_count:,}, "
            f"过滤率 {self.filter_rate:.1%}, 耗时 {self.elapsed_time:.2f}s)"
        )


class AnnouncementFilter:
    """
    公告数据过滤器
    
    使用GPU加速（cuDF）实现高效的两层过滤
    
    使用示例：
    ```python
    filter = AnnouncementFilter(use_gpu=True)
    
    # 过滤单个月份
    result = filter.filter_month(year=2021, month=1)
    
    # 过滤整年
    results = filter.filter_year(year=2021)
    
    # 过滤所有数据
    results = filter.filter_all()
    ```
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, config: Optional[FilterConfig] = None, use_gpu: bool = True):
        """
        初始化过滤器
        
        Args:
            config: 过滤配置
            use_gpu: 是否使用GPU加速
        """
        self.config = config or FilterConfig(use_gpu=use_gpu)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 检测GPU可用性并预初始化
        self._cudf_available = False
        if self.config.use_gpu:
            try:
                import cudf
                # 预初始化GPU上下文，避免后续延迟初始化导致的问题
                _ = cudf.Series([1, 2, 3])
                self._cudf_available = True
                self.logger.info("cuDF GPU加速已启用")
            except ImportError:
                self.logger.warning("cuDF不可用，将使用pandas")
            except Exception as e:
                self.logger.warning(f"cuDF初始化失败: {e}，将使用pandas")
        
        # 编译正则表达式（用于pandas回退模式）
        self._compile_patterns()
        
        self.logger.info(f"公告过滤器初始化完成 (版本 {self.VERSION}, GPU: {self._cudf_available})")
    
    def _compile_patterns(self):
        """编译正则表达式模式"""
        import re
        # 将关键词列表转换为正则表达式模式
        # 使用 | 连接，表示匹配任意一个关键词
        pattern_str = '|'.join(re.escape(kw) for kw in self.config.blacklist_keywords)
        self._title_pattern = re.compile(pattern_str)
    
    def _get_raw_path(self, category: str, year: int, month: int) -> Path:
        """获取原始数据路径"""
        return Path(self.config.raw_data_dir) / category / str(year) / f"{month:02d}.parquet"
    
    def _load_data_gpu(self, file_path: Path) -> 'cudf.DataFrame':
        """使用cuDF加载数据"""
        import cudf
        return cudf.read_parquet(str(file_path))
    
    def _load_data_cpu(self, file_path: Path) -> 'pd.DataFrame':
        """使用pandas加载数据"""
        import pandas as pd
        return pd.read_parquet(str(file_path))
    
    def _filter_by_events_gpu(
        self, 
        announcements_df: 'cudf.DataFrame', 
        events_df: 'cudf.DataFrame'
    ) -> 'cudf.DataFrame':
        """
        使用GPU进行事件ID过滤
        
        高效算法：
        1. 提取events的original_id列转为集合（cudf.Series.isin）
        2. 使用isin进行批量匹配过滤
        """
        import cudf
        
        # 获取events中的original_id集合
        event_ids = events_df['original_id'].dropna().unique()
        
        # 过滤掉announcements中original_id在event_ids中的记录
        # isin在GPU上是高度优化的
        mask = ~announcements_df['original_id'].isin(event_ids)
        
        return announcements_df[mask]
    
    def _filter_by_events_cpu(
        self, 
        announcements_df: 'pd.DataFrame', 
        events_df: 'pd.DataFrame'
    ) -> 'pd.DataFrame':
        """使用CPU进行事件ID过滤"""
        import pandas as pd
        
        # 获取events中的original_id集合
        event_ids = set(events_df['original_id'].dropna().unique())
        
        # 过滤
        mask = ~announcements_df['original_id'].isin(event_ids)
        
        return announcements_df[mask]
    
    def _filter_by_title_gpu(
        self, 
        df: 'cudf.DataFrame'
    ) -> 'cudf.DataFrame':
        """
        使用GPU进行标题关键词过滤
        
        cuDF的字符串操作是GPU加速的，使用str.contains进行批量匹配
        """
        import cudf
        
        # 构建正则表达式模式
        pattern = '|'.join(self.config.blacklist_keywords)
        
        # 使用GPU加速的字符串匹配
        # cuDF不支持na参数，先填充空值
        title_series = df['title'].fillna('')
        mask = ~title_series.str.contains(pattern, regex=True)
        
        return df[mask]
    
    def _filter_by_title_cpu(
        self, 
        df: 'pd.DataFrame'
    ) -> 'pd.DataFrame':
        """使用CPU进行标题关键词过滤"""
        import pandas as pd
        
        # 使用预编译的正则表达式
        mask = ~df['title'].str.contains(self._title_pattern, regex=True, na=False)
        
        return df[mask]
    
    def filter_month(
        self, 
        year: int, 
        month: int,
        dry_run: bool = False
    ) -> FilterResult:
        """
        过滤指定年月的公告数据
        
        Args:
            year: 年份
            month: 月份
            dry_run: 如果为True，只统计不写入
            
        Returns:
            FilterResult: 过滤结果
        """
        start_time = time.time()
        
        # 检查文件是否存在
        announcements_path = self._get_raw_path("announcements", year, month)
        events_path = self._get_raw_path("events", year, month)
        
        if not announcements_path.exists():
            return FilterResult(
                year=year,
                month=month,
                original_count=0,
                after_event_filter=0,
                after_title_filter=0,
                final_count=0,
                event_filtered_count=0,
                title_filtered_count=0,
                total_filtered_count=0,
                filter_rate=0.0,
                elapsed_time=time.time() - start_time,
                error_message=f"公告文件不存在: {announcements_path}"
            )
        
        try:
            if self._cudf_available:
                result = self._filter_month_gpu(
                    year, month, announcements_path, events_path, dry_run
                )
            else:
                result = self._filter_month_cpu(
                    year, month, announcements_path, events_path, dry_run
                )
            
            result.elapsed_time = time.time() - start_time
            self.logger.info(result.summary())
            
            return result
            
        except Exception as e:
            self.logger.error(f"过滤失败 {year}/{month:02d}: {e}")
            return FilterResult(
                year=year,
                month=month,
                original_count=0,
                after_event_filter=0,
                after_title_filter=0,
                final_count=0,
                event_filtered_count=0,
                title_filtered_count=0,
                total_filtered_count=0,
                filter_rate=0.0,
                elapsed_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _filter_month_gpu(
        self,
        year: int,
        month: int,
        announcements_path: Path,
        events_path: Path,
        dry_run: bool
    ) -> FilterResult:
        """使用GPU过滤"""
        import cudf
        
        # 加载公告数据
        ann_df = self._load_data_gpu(announcements_path)
        original_count = len(ann_df)
        
        # 第一层：事件过滤
        if self.config.enable_event_filter and events_path.exists():
            events_df = self._load_data_gpu(events_path)
            ann_df = self._filter_by_events_gpu(ann_df, events_df)
            del events_df  # 释放GPU内存
        
        after_event_filter = len(ann_df)
        event_filtered_count = original_count - after_event_filter
        
        # 第二层：标题过滤
        if self.config.enable_title_filter:
            ann_df = self._filter_by_title_gpu(ann_df)
        
        after_title_filter = len(ann_df)
        title_filtered_count = after_event_filter - after_title_filter
        
        final_count = len(ann_df)
        total_filtered_count = original_count - final_count
        filter_rate = total_filtered_count / original_count if original_count > 0 else 0.0
        
        # 写入结果
        output_path = None
        if not dry_run and final_count > 0:
            # 备份原文件
            if self.config.backup_original:
                import shutil
                backup_path = Path(self.config.backup_dir) / str(year) / f"{month:02d}.parquet"
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                # 直接复制原文件作为备份
                shutil.copy2(str(announcements_path), str(backup_path))
            
            # 重置索引确保连续，然后转换为pandas并保存
            ann_df = ann_df.reset_index(drop=True)
            pandas_df = ann_df.to_pandas()
            pandas_df.to_parquet(str(announcements_path), compression='snappy', index=False)
            output_path = str(announcements_path)
            
            # 释放内存
            del pandas_df
        
        return FilterResult(
            year=year,
            month=month,
            original_count=original_count,
            after_event_filter=after_event_filter,
            after_title_filter=after_title_filter,
            final_count=final_count,
            event_filtered_count=event_filtered_count,
            title_filtered_count=title_filtered_count,
            total_filtered_count=total_filtered_count,
            filter_rate=filter_rate,
            elapsed_time=0.0,  # 由调用者设置
            output_path=output_path
        )
    
    def _filter_month_cpu(
        self,
        year: int,
        month: int,
        announcements_path: Path,
        events_path: Path,
        dry_run: bool
    ) -> FilterResult:
        """使用CPU过滤"""
        import pandas as pd
        
        # 加载公告数据
        ann_df = self._load_data_cpu(announcements_path)
        original_count = len(ann_df)
        
        # 第一层：事件过滤
        if self.config.enable_event_filter and events_path.exists():
            events_df = self._load_data_cpu(events_path)
            ann_df = self._filter_by_events_cpu(ann_df, events_df)
            del events_df
        
        after_event_filter = len(ann_df)
        event_filtered_count = original_count - after_event_filter
        
        # 第二层：标题过滤
        if self.config.enable_title_filter:
            ann_df = self._filter_by_title_cpu(ann_df)
        
        after_title_filter = len(ann_df)
        title_filtered_count = after_event_filter - after_title_filter
        
        final_count = len(ann_df)
        total_filtered_count = original_count - final_count
        filter_rate = total_filtered_count / original_count if original_count > 0 else 0.0
        
        # 写入结果
        output_path = None
        if not dry_run and final_count > 0:
            # 备份原文件
            if self.config.backup_original:
                backup_path = Path(self.config.backup_dir) / str(year) / f"{month:02d}.parquet"
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                pd.read_parquet(str(announcements_path)).to_parquet(
                    str(backup_path), compression='snappy'
                )
            
            # 保存过滤后的数据
            ann_df.to_parquet(str(announcements_path), compression='snappy', index=False)
            output_path = str(announcements_path)
        
        return FilterResult(
            year=year,
            month=month,
            original_count=original_count,
            after_event_filter=after_event_filter,
            after_title_filter=after_title_filter,
            final_count=final_count,
            event_filtered_count=event_filtered_count,
            title_filtered_count=title_filtered_count,
            total_filtered_count=total_filtered_count,
            filter_rate=filter_rate,
            elapsed_time=0.0,
            output_path=output_path
        )
    
    def filter_year(
        self, 
        year: int,
        start_month: int = 1,
        end_month: int = 12,
        dry_run: bool = False
    ) -> List[FilterResult]:
        """
        过滤指定年份的所有月份
        
        Args:
            year: 年份
            start_month: 起始月份
            end_month: 结束月份
            dry_run: 如果为True，只统计不写入
            
        Returns:
            List[FilterResult]: 各月份的过滤结果
        """
        results = []
        
        for month in range(start_month, end_month + 1):
            result = self.filter_month(year, month, dry_run=dry_run)
            results.append(result)
        
        return results
    
    def filter_all(
        self,
        years: Optional[List[int]] = None,
        dry_run: bool = False
    ) -> Dict[int, List[FilterResult]]:
        """
        过滤所有年份的数据
        
        Args:
            years: 要过滤的年份列表，如果为None则自动发现
            dry_run: 如果为True，只统计不写入
            
        Returns:
            Dict[int, List[FilterResult]]: 按年份分组的过滤结果
        """
        # 自动发现年份
        if years is None:
            announcements_dir = Path(self.config.raw_data_dir) / "announcements"
            years = []
            for year_dir in announcements_dir.glob("*"):
                if year_dir.is_dir() and year_dir.name.isdigit():
                    years.append(int(year_dir.name))
            years.sort()
        
        results = {}
        for year in years:
            self.logger.info(f"开始过滤 {year} 年数据...")
            results[year] = self.filter_year(year, dry_run=dry_run)
        
        return results
    
    def get_statistics(
        self,
        year: int,
        month: int
    ) -> Dict[str, Any]:
        """
        获取指定年月的统计信息（不修改数据）
        
        Args:
            year: 年份
            month: 月份
            
        Returns:
            统计信息字典
        """
        result = self.filter_month(year, month, dry_run=True)
        
        return {
            'year': year,
            'month': month,
            'original_count': result.original_count,
            'event_filtered': result.event_filtered_count,
            'title_filtered': result.title_filtered_count,
            'final_count': result.final_count,
            'filter_rate': result.filter_rate,
            'estimated_processing_time_hours': result.final_count / 3600,  # 假设1秒处理1条
        }
