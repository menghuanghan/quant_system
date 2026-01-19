"""
非结构化数据采集综合测试 - 2025年Q1

采集内容：
1. 上市公司公告（年报/中报/季报PDF，临时公告，重大事项说明）
2. 新闻原始数据（东方财富、新浪财经、证券时报、交易所公告解读、央视新闻联播）
3. 研报与分析师观点（投资评级、目标价、EPS预测、评级变化）
4. 舆情与市场情绪（股吧、雪球、热度数据）

时间范围：2025年1-3月
输出目录：data/raw/unstructured/
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 输出基础目录
OUTPUT_BASE = project_root / 'data' / 'raw' / 'unstructured'


def ensure_dir(path: Path):
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ==================== 1. 公告采集 ====================
def collect_announcements():
    """采集上市公司公告"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("1. 上市公司公告采集")
    logger.info("=" * 70)
    
    output_dir = ensure_dir(OUTPUT_BASE / 'announcements')
    results = {}
    
    # 1.1 巨潮资讯公告
    logger.info("")
    logger.info("[1.1] 巨潮资讯公告 (CnInfo)")
    
    try:
        from src.data_pipeline.collectors.unstructured.announcements import (
            CninfoAnnouncementCrawler
        )
        
        crawler = CninfoAnnouncementCrawler()
        
        months = [
            ('2025-01-01', '2025-01-31', '202501'),
            ('2025-02-01', '2025-02-28', '202502'),
            ('2025-03-01', '2025-03-31', '202503'),
        ]
        
        total_cninfo = 0
        for start_date, end_date, month_label in months:
            logger.info(f"  采集 {month_label}...")
            
            for exchange in ['szse', 'sse']:
                try:
                    df = crawler.collect(
                        start_date=start_date,
                        end_date=end_date,
                        exchanges=[exchange]
                    )
                    
                    if not df.empty:
                        logger.info(f"    {exchange.upper()}: {len(df)} 条")
                        total_cninfo += len(df)
                        
                        output_file = output_dir / f'cninfo_{exchange}_{month_label}.csv'
                        df.to_csv(output_file, index=False, encoding='utf-8-sig')
                except Exception as e:
                    logger.warning(f"    {exchange.upper()} 采集失败: {e}")
        
        results['cninfo'] = total_cninfo
        logger.info(f"  ✓ 巨潮资讯总计: {total_cninfo} 条")
        
    except Exception as e:
        logger.error(f"  ✗ 巨潮资讯采集失败: {e}")
        results['cninfo'] = 0
    
    # 1.2 Tushare公告
    logger.info("")
    logger.info("[1.2] Tushare公告接口")
    
    try:
        from src.data_pipeline.collectors.unstructured.announcements import (
            TushareAnnouncementCollector
        )
        
        ts_collector = TushareAnnouncementCollector()
        
        # 采集几只代表性股票的公告
        test_stocks = ['000001.SZ', '600519.SH', '000002.SZ', '600036.SH']
        
        all_data = []
        for ts_code in test_stocks:
            try:
                df = ts_collector.collect(
                    start_date='2025-01-01',
                    end_date='2025-03-31',
                    ts_code=ts_code
                )
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"    {ts_code}: {len(df)} 条")
            except Exception as e:
                logger.warning(f"    {ts_code} 采集失败: {e}")
        
        if all_data:
            import pandas as pd
            df_combined = pd.concat(all_data, ignore_index=True)
            df_combined.to_csv(output_dir / 'tushare_announcements_2025Q1.csv', 
                              index=False, encoding='utf-8-sig')
            results['tushare_ann'] = len(df_combined)
            logger.info(f"  ✓ Tushare公告总计: {len(df_combined)} 条")
        else:
            results['tushare_ann'] = 0
            
    except Exception as e:
        logger.error(f"  ✗ Tushare公告采集失败: {e}")
        results['tushare_ann'] = 0
    
    # 1.3 AKShare公告
    logger.info("")
    logger.info("[1.3] AKShare公告接口")
    
    try:
        from src.data_pipeline.collectors.unstructured.announcements import (
            AKShareAnnouncementCollector
        )
        
        ak_collector = AKShareAnnouncementCollector()
        
        df = ak_collector.collect(
            start_date='2025-01-01',
            end_date='2025-03-31'
        )
        
        if not df.empty:
            df.to_csv(output_dir / 'akshare_announcements_2025Q1.csv',
                     index=False, encoding='utf-8-sig')
            results['akshare_ann'] = len(df)
            logger.info(f"  ✓ AKShare公告总计: {len(df)} 条")
        else:
            results['akshare_ann'] = 0
            logger.warning("  无数据")
            
    except Exception as e:
        logger.error(f"  ✗ AKShare公告采集失败: {e}")
        results['akshare_ann'] = 0
    
    return results


# ==================== 2. 新闻采集 ====================
def collect_news():
    """采集新闻原始数据"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("2. 新闻原始数据采集")
    logger.info("=" * 70)
    
    output_dir = ensure_dir(OUTPUT_BASE / 'news')
    results = {}
    
    # 2.1 央视新闻联播
    logger.info("")
    logger.info("[2.1] 央视新闻联播")
    
    try:
        from src.data_pipeline.collectors.unstructured.news import CCTVNewsCollector
        
        cctv_collector = CCTVNewsCollector()
        cctv_count = 0
        
        months = [
            ('2025-01-01', '2025-01-31', '01'),
            ('2025-02-01', '2025-02-28', '02'),
            ('2025-03-01', '2025-03-31', '03'),
        ]
        
        for start, end, month in months:
            try:
                df = cctv_collector.collect(start, end)
                if not df.empty:
                    cctv_count += len(df)
                    df.to_csv(output_dir / f'cctv_news_2025{month}.csv',
                             index=False, encoding='utf-8-sig')
                    logger.info(f"    {month}月: {len(df)} 条")
            except Exception as e:
                logger.warning(f"    {month}月采集失败: {e}")
        
        results['cctv'] = cctv_count
        logger.info(f"  ✓ 央视新闻联播总计: {cctv_count} 条")
        
    except Exception as e:
        logger.error(f"  ✗ 央视新闻联播采集失败: {e}")
        results['cctv'] = 0
    
    # 2.2 东方财富新闻
    logger.info("")
    logger.info("[2.2] 东方财富新闻")
    
    try:
        from src.data_pipeline.collectors.unstructured.news import EastMoneyNewsCollector
        
        em_collector = EastMoneyNewsCollector()
        
        # 热门股票个股新闻
        hot_stocks = ['000001', '600519', '000002', '600036', '300750']
        
        df = em_collector.collect(
            start_date='2025-01-01',
            end_date='2025-03-31',
            symbols=hot_stocks,
            include_market_news=True
        )
        
        if not df.empty:
            df.to_csv(output_dir / 'eastmoney_news_2025Q1.csv',
                     index=False, encoding='utf-8-sig')
            results['eastmoney'] = len(df)
            logger.info(f"  ✓ 东方财富新闻: {len(df)} 条")
        else:
            results['eastmoney'] = 0
            logger.warning("  无数据")
            
    except Exception as e:
        logger.error(f"  ✗ 东方财富新闻采集失败: {e}")
        results['eastmoney'] = 0
    
    # 2.3 新浪财经
    logger.info("")
    logger.info("[2.3] 新浪财经")
    
    try:
        from src.data_pipeline.collectors.unstructured.news import SinaFinanceCrawler
        
        sina_crawler = SinaFinanceCrawler()
        
        df = sina_crawler.collect(
            start_date='2025-01-01',
            end_date='2025-03-31',
            categories=['stock', 'finance', 'market']
        )
        
        if not df.empty:
            df.to_csv(output_dir / 'sina_news_2025Q1.csv',
                     index=False, encoding='utf-8-sig')
            results['sina'] = len(df)
            logger.info(f"  ✓ 新浪财经: {len(df)} 条")
        else:
            results['sina'] = 0
            logger.warning("  无数据")
            
    except Exception as e:
        logger.error(f"  ✗ 新浪财经采集失败: {e}")
        results['sina'] = 0
    
    # 2.4 证券时报
    logger.info("")
    logger.info("[2.4] 证券时报")
    
    try:
        from src.data_pipeline.collectors.unstructured.news import STCNCrawler
        
        stcn_crawler = STCNCrawler()
        
        df = stcn_crawler.collect(
            start_date='2025-01-01',
            end_date='2025-03-31',
            channels=['stock', 'company', 'market']
        )
        
        if not df.empty:
            df.to_csv(output_dir / 'stcn_news_2025Q1.csv',
                     index=False, encoding='utf-8-sig')
            results['stcn'] = len(df)
            logger.info(f"  ✓ 证券时报: {len(df)} 条")
        else:
            results['stcn'] = 0
            logger.warning("  无数据")
            
    except Exception as e:
        logger.error(f"  ✗ 证券时报采集失败: {e}")
        results['stcn'] = 0
    
    # 2.5 交易所公告解读
    logger.info("")
    logger.info("[2.5] 交易所公告解读")
    
    try:
        from src.data_pipeline.collectors.unstructured.news import ExchangeNewsCrawler
        
        exchange_crawler = ExchangeNewsCrawler()
        
        df = exchange_crawler.collect(
            start_date='2025-01-01',
            end_date='2025-03-31'
        )
        
        if not df.empty:
            df.to_csv(output_dir / 'exchange_news_2025Q1.csv',
                     index=False, encoding='utf-8-sig')
            results['exchange'] = len(df)
            logger.info(f"  ✓ 交易所公告解读: {len(df)} 条")
        else:
            results['exchange'] = 0
            logger.warning("  无数据")
            
    except Exception as e:
        logger.error(f"  ✗ 交易所公告解读采集失败: {e}")
        results['exchange'] = 0
    
    return results


# ==================== 3. 研报采集 ====================
def collect_reports():
    """采集研报与分析师观点"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("3. 研报与分析师观点采集")
    logger.info("=" * 70)
    
    output_dir = ensure_dir(OUTPUT_BASE / 'reports')
    results = {}
    
    import pandas as pd
    
    # 覆盖多行业的股票列表
    stock_list = [
        # 银行
        '000001', '600036', '601166',
        # 白酒
        '600519', '000858',
        # 科技
        '002415', '300059',
        # 新能源
        '300750', '002594',
        # 医药
        '600276', '000538',
        # 地产
        '000002', '600048',
        # 消费
        '000651', '600887',
        # 券商
        '600030', '601688',
    ]
    
    # 3.1 券商研报
    logger.info("")
    logger.info("[3.1] 券商研报 (投资评级、目标价、PDF链接)")
    
    try:
        from src.data_pipeline.collectors.unstructured.reports import (
            EastMoneyReportCollector
        )
        
        report_collector = EastMoneyReportCollector()
        
        all_reports = []
        batch_size = 6
        
        for i in range(0, len(stock_list), batch_size):
            batch = stock_list[i:i+batch_size]
            logger.info(f"    批次 {i//batch_size + 1}: {batch}")
            
            try:
                df = report_collector.collect(
                    start_date='2025-01-01',
                    end_date='2025-03-31',
                    stock_codes=batch
                )
                
                if not df.empty:
                    all_reports.append(df)
            except Exception as e:
                logger.warning(f"    批次采集失败: {e}")
        
        if all_reports:
            df_reports = pd.concat(all_reports, ignore_index=True)
            df_reports = df_reports.drop_duplicates(subset=['report_id'], keep='first')
            
            df_reports.to_csv(output_dir / 'reports_2025Q1.csv',
                             index=False, encoding='utf-8-sig')
            results['reports'] = len(df_reports)
            logger.info(f"  ✓ 券商研报: {len(df_reports)} 条")
            
            # 统计
            if 'rating' in df_reports.columns:
                logger.info("    [评级分布]")
                for rating, count in df_reports['rating'].value_counts().head(5).items():
                    logger.info(f"      {rating}: {count}")
        else:
            results['reports'] = 0
            
    except Exception as e:
        logger.error(f"  ✗ 券商研报采集失败: {e}")
        results['reports'] = 0
    
    # 3.2 盈利预测 (EPS)
    logger.info("")
    logger.info("[3.2] 盈利预测 (EPS)")
    
    try:
        from src.data_pipeline.collectors.unstructured.reports import get_eps_forecast
        
        df_eps = get_eps_forecast(stock_list[:15])
        
        if not df_eps.empty:
            df_eps.to_csv(output_dir / 'eps_forecast_2025.csv',
                         index=False, encoding='utf-8-sig')
            results['eps'] = len(df_eps)
            logger.info(f"  ✓ 盈利预测: {len(df_eps)} 条")
        else:
            results['eps'] = 0
            logger.warning("  无数据")
            
    except Exception as e:
        logger.error(f"  ✗ 盈利预测采集失败: {e}")
        results['eps'] = 0
    
    # 3.3 分析师排名
    logger.info("")
    logger.info("[3.3] 分析师排名")
    
    try:
        from src.data_pipeline.collectors.unstructured.reports import AnalystCollector
        
        analyst_collector = AnalystCollector()
        
        df_analysts = analyst_collector.collect_analyst_rank()
        
        if not df_analysts.empty:
            df_analysts.to_csv(output_dir / 'analyst_rank_2025Q1.csv',
                              index=False, encoding='utf-8-sig')
            results['analysts'] = len(df_analysts)
            logger.info(f"  ✓ 分析师排名: {len(df_analysts)} 条")
        else:
            results['analysts'] = 0
            logger.warning("  无数据")
            
    except Exception as e:
        logger.error(f"  ✗ 分析师排名采集失败: {e}")
        results['analysts'] = 0
    
    return results


# ==================== 4. 舆情采集 ====================
def collect_sentiment():
    """采集舆情与市场情绪数据"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("4. 舆情与市场情绪采集")
    logger.info("=" * 70)
    
    output_dir = ensure_dir(OUTPUT_BASE / 'sentiment')
    results = {}
    
    import pandas as pd
    
    # 测试股票
    test_stocks = ['000001.SZ', '600519.SH', '000002.SZ', '300750.SZ', '600036.SH']
    
    # 4.1 实时热榜
    logger.info("")
    logger.info("[4.1] 市场热度 - 实时热榜")
    
    try:
        from src.data_pipeline.collectors.unstructured.sentiment import (
            get_realtime_hotlist
        )
        
        # 东方财富热榜
        try:
            df_em_hot = get_realtime_hotlist(source='eastmoney')
            if not df_em_hot.empty:
                df_em_hot.to_csv(output_dir / 'hotlist_eastmoney.csv',
                               index=False, encoding='utf-8-sig')
                results['hotlist_em'] = len(df_em_hot)
                logger.info(f"    东方财富热榜: {len(df_em_hot)} 条")
            else:
                results['hotlist_em'] = 0
        except Exception as e:
            logger.warning(f"    东方财富热榜失败: {e}")
            results['hotlist_em'] = 0
        
        # 雪球热榜
        try:
            df_xq_hot = get_realtime_hotlist(source='xueqiu')
            if not df_xq_hot.empty:
                df_xq_hot.to_csv(output_dir / 'hotlist_xueqiu.csv',
                               index=False, encoding='utf-8-sig')
                results['hotlist_xq'] = len(df_xq_hot)
                logger.info(f"    雪球热榜: {len(df_xq_hot)} 条")
            else:
                results['hotlist_xq'] = 0
        except Exception as e:
            logger.warning(f"    雪球热榜失败: {e}")
            results['hotlist_xq'] = 0
        
        logger.info(f"  ✓ 实时热榜总计: {results.get('hotlist_em', 0) + results.get('hotlist_xq', 0)} 条")
        
    except Exception as e:
        logger.error(f"  ✗ 实时热榜采集失败: {e}")
        results['hotlist_em'] = 0
        results['hotlist_xq'] = 0
    
    # 4.2 互动易问答
    logger.info("")
    logger.info("[4.2] 互动易问答 (cninfo_interaction)")
    
    try:
        from src.data_pipeline.collectors.unstructured.sentiment import (
            get_cninfo_interaction
        )
        
        df_interaction = get_cninfo_interaction(
            ts_codes=test_stocks,
            start_date='2025-01-01',
            end_date='2025-03-31'
        )
        
        if not df_interaction.empty:
            df_interaction.to_csv(output_dir / 'cninfo_interaction_2025Q1.csv',
                                 index=False, encoding='utf-8-sig')
            results['interaction'] = len(df_interaction)
            logger.info(f"  ✓ 互动易问答: {len(df_interaction)} 条")
        else:
            results['interaction'] = 0
            logger.warning("  无数据 (可能需要Tushare积分)")
            
    except Exception as e:
        logger.error(f"  ✗ 互动易问答采集失败: {e}")
        results['interaction'] = 0
    
    # 4.3 股吧评论
    logger.info("")
    logger.info("[4.3] 股吧评论")
    
    try:
        from src.data_pipeline.collectors.unstructured.sentiment import (
            get_guba_comments
        )
        
        df_guba = get_guba_comments(
            ts_codes=test_stocks,
            max_pages=5
        )
        
        if not df_guba.empty:
            df_guba.to_csv(output_dir / 'guba_comments_2025Q1.csv',
                          index=False, encoding='utf-8-sig')
            results['guba'] = len(df_guba)
            logger.info(f"  ✓ 股吧评论: {len(df_guba)} 条")
        else:
            results['guba'] = 0
            logger.warning("  无数据")
            
    except Exception as e:
        logger.error(f"  ✗ 股吧评论采集失败: {e}")
        results['guba'] = 0
    
    # 4.4 雪球讨论
    logger.info("")
    logger.info("[4.4] 雪球讨论")
    
    try:
        from src.data_pipeline.collectors.unstructured.sentiment import (
            get_xueqiu_comments
        )
        
        df_xq = get_xueqiu_comments(
            ts_codes=test_stocks[:3],  # 限制数量
            max_pages=3
        )
        
        if not df_xq.empty:
            df_xq.to_csv(output_dir / 'xueqiu_comments_2025Q1.csv',
                        index=False, encoding='utf-8-sig')
            results['xueqiu'] = len(df_xq)
            logger.info(f"  ✓ 雪球讨论: {len(df_xq)} 条")
        else:
            results['xueqiu'] = 0
            logger.warning("  无数据 (需要配置XUEQIU_COOKIE)")
            
    except Exception as e:
        logger.error(f"  ✗ 雪球讨论采集失败: {e}")
        results['xueqiu'] = 0
    
    # 4.5 历史热度代理指标
    logger.info("")
    logger.info("[4.5] 历史热度代理指标 (换手率+新闻)")
    
    try:
        from src.data_pipeline.collectors.unstructured.sentiment import (
            get_historical_heat_proxy
        )
        
        df_heat = get_historical_heat_proxy(
            ts_codes=test_stocks,
            start_date='2025-01-01',
            end_date='2025-03-31'
        )
        
        if not df_heat.empty:
            df_heat.to_csv(output_dir / 'heat_proxy_2025Q1.csv',
                          index=False, encoding='utf-8-sig')
            results['heat_proxy'] = len(df_heat)
            logger.info(f"  ✓ 历史热度代理: {len(df_heat)} 条")
        else:
            results['heat_proxy'] = 0
            logger.warning("  无数据")
            
    except Exception as e:
        logger.error(f"  ✗ 历史热度代理采集失败: {e}")
        results['heat_proxy'] = 0
    
    return results


# ==================== 主函数 ====================
def main():
    """运行所有采集测试"""
    logger.info("=" * 70)
    logger.info("非结构化数据采集综合测试")
    logger.info(f"时间范围: 2025年1-3月")
    logger.info(f"输出目录: {OUTPUT_BASE}")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    all_results = {}
    
    # 1. 公告采集
    try:
        ann_results = collect_announcements()
        all_results['announcements'] = ann_results
    except Exception as e:
        logger.error(f"公告采集模块异常: {e}")
        all_results['announcements'] = {}
    
    # 2. 新闻采集
    try:
        news_results = collect_news()
        all_results['news'] = news_results
    except Exception as e:
        logger.error(f"新闻采集模块异常: {e}")
        all_results['news'] = {}
    
    # 3. 研报采集
    try:
        report_results = collect_reports()
        all_results['reports'] = report_results
    except Exception as e:
        logger.error(f"研报采集模块异常: {e}")
        all_results['reports'] = {}
    
    # 4. 舆情采集
    try:
        sentiment_results = collect_sentiment()
        all_results['sentiment'] = sentiment_results
    except Exception as e:
        logger.error(f"舆情采集模块异常: {e}")
        all_results['sentiment'] = {}
    
    # ==================== 汇总 ====================
    logger.info("")
    logger.info("=" * 70)
    logger.info("采集结果汇总")
    logger.info("=" * 70)
    
    for category, results in all_results.items():
        logger.info(f"\n[{category}]")
        if results:
            for key, count in results.items():
                status = "✓" if count > 0 else "✗"
                logger.info(f"  {status} {key}: {count} 条")
        else:
            logger.info("  无数据")
    
    # 列出所有生成的文件
    logger.info("")
    logger.info("=" * 70)
    logger.info("生成的文件列表")
    logger.info("=" * 70)
    
    for root, dirs, files in os.walk(OUTPUT_BASE):
        for f in sorted(files):
            if f.endswith(('.csv', '.parquet', '.jsonl')):
                filepath = os.path.join(root, f)
                size = os.path.getsize(filepath)
                rel_path = os.path.relpath(filepath, OUTPUT_BASE)
                logger.info(f"  {rel_path}: {size/1024:.1f} KB")
    
    logger.info("")
    logger.info(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
