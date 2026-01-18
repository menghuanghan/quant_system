"""
新闻原始数据采集 - 2025年Q1完整测试（修复版）

采集2025年1月、2月、3月新闻数据（5个数据源）
"""

import os
import sys
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect_news_q1_2025():
    """采集2025年Q1新闻数据"""
    
    from src.data_pipeline.collectors.unstructured.news import (
        CCTVNewsCollector,
        EastMoneyNewsCollector,
        SinaFinanceCrawler,
        STCNCrawler,
        ExchangeNewsCrawler,
    )
    
    # 输出目录
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'unstructured', 'news'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"输出目录: {output_dir}")
    
    total_records = 0
    results = {}
    
    # ========== 1. 央视新闻联播 ==========
    logger.info("=" * 60)
    logger.info("1. 采集央视新闻联播 (2025年1-3月)")
    logger.info("=" * 60)
    
    try:
        cctv_collector = CCTVNewsCollector()
        cctv_count = 0
        
        for month, (start, end) in enumerate([
            ('2025-01-01', '2025-01-31'),
            ('2025-02-01', '2025-02-28'),
            ('2025-03-01', '2025-03-31'),
        ], 1):
            logger.info(f"  采集 {month} 月...")
            df = cctv_collector.collect(start, end)
            
            if not df.empty:
                logger.info(f"    {len(df)} 条")
                cctv_count += len(df)
                
                output_file = os.path.join(output_dir, f'cctv_news_20250{month}.csv')
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        total_records += cctv_count
        results['cctv'] = cctv_count
        logger.info(f"  央视新闻联播总计: {cctv_count} 条")
                
    except Exception as e:
        logger.error(f"央视新闻联播采集失败: {e}")
        import traceback
        traceback.print_exc()
        results['cctv'] = 0
    
    # ========== 2. 东方财富个股新闻 ==========
    logger.info("=" * 60)
    logger.info("2. 采集东方财富个股新闻")
    logger.info("=" * 60)
    
    try:
        em_collector = EastMoneyNewsCollector()
        
        # 采集热门股票个股新闻（使用AKShare接口，更稳定）
        hot_stocks = ['000001', '600000', '000002', '600036', '600519', '000858']
        logger.info(f"  采集热门股票新闻: {hot_stocks}")
        
        df_stocks = em_collector.collect(
            start_date='2025-01-01',
            end_date='2025-03-31',
            symbols=hot_stocks,
            include_market_news=False  # 市场新闻API不稳定，跳过
        )
        
        if not df_stocks.empty:
            logger.info(f"  个股新闻: {len(df_stocks)} 条")
            total_records += len(df_stocks)
            results['eastmoney'] = len(df_stocks)
            
            output_file = os.path.join(output_dir, 'eastmoney_stock_news_2025Q1.csv')
            df_stocks.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"  已保存: {output_file}")
        else:
            results['eastmoney'] = 0
            
    except Exception as e:
        logger.error(f"东方财富新闻采集失败: {e}")
        import traceback
        traceback.print_exc()
        results['eastmoney'] = 0
    
    # ========== 3. 新浪财经 ==========
    logger.info("=" * 60)
    logger.info("3. 采集新浪财经新闻")
    logger.info("=" * 60)
    
    try:
        sina_crawler = SinaFinanceCrawler()
        
        # 由于API返回的是最新新闻，不能精确按日期过滤
        # 设置宽泛的日期范围以获取尽可能多的数据
        df_sina = sina_crawler.collect(
            start_date='2025-01-01',
            end_date='2026-12-31',  # 宽泛范围
            categories=['stock', 'finance', 'market']
        )
        
        if not df_sina.empty:
            logger.info(f"  新浪财经: {len(df_sina)} 条")
            total_records += len(df_sina)
            results['sina'] = len(df_sina)
            
            # 按类别统计
            if 'category' in df_sina.columns:
                for cat, count in df_sina['category'].value_counts().items():
                    logger.info(f"    {cat}: {count} 条")
            
            output_file = os.path.join(output_dir, 'sina_news_2025Q1.csv')
            df_sina.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"  已保存: {output_file}")
        else:
            results['sina'] = 0
            
    except Exception as e:
        logger.error(f"新浪财经采集失败: {e}")
        import traceback
        traceback.print_exc()
        results['sina'] = 0
    
    # ========== 4. 证券时报 ==========
    logger.info("=" * 60)
    logger.info("4. 采集证券时报新闻")
    logger.info("=" * 60)
    
    try:
        stcn_crawler = STCNCrawler()
        
        df_stcn = stcn_crawler.collect(
            start_date='2025-01-01',
            end_date='2026-12-31',  # 宽泛范围
            channels=['stock', 'company', 'market']
        )
        
        if not df_stcn.empty:
            logger.info(f"  证券时报: {len(df_stcn)} 条")
            total_records += len(df_stcn)
            results['stcn'] = len(df_stcn)
            
            output_file = os.path.join(output_dir, 'stcn_news_2025Q1.csv')
            df_stcn.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"  已保存: {output_file}")
        else:
            results['stcn'] = 0
            logger.warning("  无数据")
            
    except Exception as e:
        logger.error(f"证券时报采集失败: {e}")
        import traceback
        traceback.print_exc()
        results['stcn'] = 0
    
    # ========== 5. 交易所公告解读 ==========
    logger.info("=" * 60)
    logger.info("5. 采集交易所公告解读")
    logger.info("=" * 60)
    
    try:
        exchange_crawler = ExchangeNewsCrawler()
        
        df_exchange = exchange_crawler.collect(
            start_date='2025-01-01',
            end_date='2025-03-31'
        )
        
        if not df_exchange.empty:
            logger.info(f"  公告解读: {len(df_exchange)} 条")
            total_records += len(df_exchange)
            results['exchange'] = len(df_exchange)
            
            output_file = os.path.join(output_dir, 'exchange_news_2025Q1.csv')
            df_exchange.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"  已保存: {output_file}")
        else:
            results['exchange'] = 0
            logger.warning("  无数据")
            
    except Exception as e:
        logger.error(f"交易所公告解读采集失败: {e}")
        import traceback
        traceback.print_exc()
        results['exchange'] = 0
    
    # ========== 汇总 ==========
    logger.info("=" * 60)
    logger.info("采集结果汇总")
    logger.info("=" * 60)
    
    for source, count in results.items():
        status = "✓" if count > 0 else "✗"
        logger.info(f"  {status} {source}: {count} 条")
    
    logger.info(f"总计: {total_records} 条新闻")
    logger.info("=" * 60)
    
    # 列出生成的文件
    logger.info("生成的文件:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.csv'):
            filepath = os.path.join(output_dir, f)
            size = os.path.getsize(filepath)
            logger.info(f"  {f}: {size/1024:.1f} KB")


if __name__ == '__main__':
    collect_news_q1_2025()
