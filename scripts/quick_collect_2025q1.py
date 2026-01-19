"""
快速非结构化数据采集脚本 - 2025年Q1采样
针对速度优化的采集版本

采集内容：
1. 东方财富新闻（近期数据）
2. 新浪财经新闻（近期数据）  
3. 证券时报新闻（少量页面）
4. 股吧评论（限制页数）
5. 东方财富评论
6. 雪球评论（小样本）
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据保存目录
DATA_DIR = project_root / "data" / "raw" / "unstructured"
NEWS_DIR = DATA_DIR / "news"
SENTIMENT_DIR = DATA_DIR / "sentiment"

# 确保目录存在
NEWS_DIR.mkdir(parents=True, exist_ok=True)
SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

def collect_eastmoney_news_quick():
    """快速采集东方财富新闻（近期数据）"""
    print("\n" + "="*50)
    print("1. 快速采集东方财富新闻")
    print("="*50)
    
    try:
        from src.data_pipeline.collectors.unstructured.news.eastmoney_collector import EastMoneyNewsCollector
        
        collector = EastMoneyNewsCollector()
        
        # 采集近期数据（不做严格日期过滤）
        df = collector.collect(
            start_date="20250101", 
            end_date="20260131",  # 扩大范围获取近期数据
            channels=['7x24', 'stock'],  # 只采集主要频道
            max_pages=2  # 限制页数
        )
        
        if not df.empty:
            file_path = NEWS_DIR / "eastmoney_news_2025Q1_quick.csv"
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"✓ 东方财富新闻采集完成: {len(df)} 条")
            print(f"  保存至: {file_path}")
            return True
        else:
            print("✗ 东方财富新闻采集失败")
            return False
            
    except Exception as e:
        print(f"✗ 东方财富新闻采集失败: {e}")
        return False

def collect_sina_news_quick():
    """快速采集新浪财经新闻（近期数据）"""
    print("\n" + "="*50)
    print("2. 快速采集新浪财经新闻")
    print("="*50)
    
    try:
        from src.data_pipeline.collectors.unstructured.news.sina_crawler import SinaNewsCrawler
        
        crawler = SinaNewsCrawler()
        
        # 采集近期数据
        df = crawler.collect(
            start_date="20250101",
            end_date="20260131",  # 扩大范围获取近期数据
            channels=['stock', 'finance'],
            max_pages=3  # 限制页数
        )
        
        if not df.empty:
            file_path = NEWS_DIR / "sina_news_2025Q1_quick.csv"
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"✓ 新浪财经新闻采集完成: {len(df)} 条")
            print(f"  保存至: {file_path}")
            return True
        else:
            print("✗ 新浪财经新闻采集失败")
            return False
            
    except Exception as e:
        print(f"✗ 新浪财经新闻采集失败: {e}")
        return False

def collect_stcn_news_quick():
    """快速采集证券时报新闻"""
    print("\n" + "="*50)
    print("3. 快速采集证券时报新闻")
    print("="*50)
    
    try:
        from src.data_pipeline.collectors.unstructured.news.stcn_crawler import STCNCrawler
        
        crawler = STCNCrawler()
        
        # 采集近期数据，不做严格日期过滤
        df_raw = crawler._collect_by_channel_http('kx', max_pages=2)  # 只采集2页
        
        if not df_raw.empty:
            file_path = NEWS_DIR / "stcn_news_2025Q1_quick.csv"
            df_raw.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"✓ 证券时报新闻采集完成: {len(df_raw)} 条")
            print(f"  保存至: {file_path}")
            return True
        else:
            print("✗ 证券时报新闻采集失败")
            return False
            
    except Exception as e:
        print(f"✗ 证券时报新闻采集失败: {e}")
        return False

def collect_guba_comments_quick():
    """快速采集股吧评论（限制数量）"""
    print("\n" + "="*50)
    print("4. 快速采集股吧评论")
    print("="*50)
    
    try:
        from src.data_pipeline.collectors.unstructured.sentiment.investor_sentiment import (
            InvestorSentimentCollector, SentimentSource
        )
        
        collector = InvestorSentimentCollector()
        
        # 只采集少数股票的评论
        sample_stocks = ['600519.SH', '000001.SZ', '000002.SZ']  # 茅台、平安、万科
        
        all_data = []
        for ts_code in sample_stocks:
            print(f"采集 {ts_code} 股吧评论...")
            
            # 限制页数为2页，加快速度
            df = collector._scrape_guba_comments(ts_code, fetch_content=False)  # 不获取详细内容以加快速度
            if not df.empty:
                all_data.append(df)
                print(f"  {ts_code}: {len(df)} 条")
        
        if all_data:
            import pandas as pd
            df_all = pd.concat(all_data, ignore_index=True)
            
            file_path = SENTIMENT_DIR / "guba_comments_2025Q1_quick.csv"
            df_all.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"✓ 股吧评论采集完成: {len(df_all)} 条")
            print(f"  保存至: {file_path}")
            return True
        else:
            print("✗ 股吧评论采集失败")
            return False
            
    except Exception as e:
        print(f"✗ 股吧评论采集失败: {e}")
        return False

def collect_eastmoney_comments_quick():
    """快速采集东方财富评论"""
    print("\n" + "="*50)
    print("5. 快速采集东方财富评论")
    print("="*50)
    
    try:
        from src.data_pipeline.collectors.unstructured.scraper_base import EastMoneyCollector
        import pandas as pd
        
        collector = EastMoneyCollector()
        
        # 只采集少数股票
        sample_stocks = ['600519', '000001']
        all_data = []
        
        for stock_code in sample_stocks:
            print(f"采集 {stock_code} 东财评论...")
            
            comments = collector.get_guba_comments(stock_code, max_pages=2)  # 限制2页
            if comments:
                all_data.extend(comments)
                print(f"  {stock_code}: {len(comments)} 条")
        
        if all_data:
            df = pd.DataFrame(all_data)
            
            file_path = SENTIMENT_DIR / "eastmoney_comments_2025Q1_quick.csv"
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"✓ 东方财富评论采集完成: {len(df)} 条")
            print(f"  保存至: {file_path}")
            return True
        else:
            print("✗ 东方财富评论采集失败")
            return False
            
    except Exception as e:
        print(f"✗ 东方财富评论采集失败: {e}")
        return False

def collect_xueqiu_comments_quick():
    """快速采集雪球评论（小样本）"""
    print("\n" + "="*50)
    print("6. 快速采集雪球评论")
    print("="*50)
    
    try:
        from src.data_pipeline.collectors.unstructured.scraper_base import XueqiuCollector
        import pandas as pd
        
        collector = XueqiuCollector()
        
        # 只采集1只股票作为样本
        stock_symbol = 'SH600519'  # 茅台
        print(f"采集 {stock_symbol} 雪球评论...")
        
        comments = collector.get_stock_comments(stock_symbol, max_pages=1)  # 只采集1页
        
        if comments:
            df = pd.DataFrame(comments)
            
            file_path = SENTIMENT_DIR / "xueqiu_comments_2025Q1_quick.csv"
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"✓ 雪球评论采集完成: {len(df)} 条")
            print(f"  保存至: {file_path}")
            return True
        else:
            print("✗ 雪球评论采集失败")
            return False
            
    except Exception as e:
        print(f"✗ 雪球评论采集失败: {e}")
        return False

def main():
    """主函数 - 快速采集"""
    print("="*60)
    print("非结构化数据快速采集 - 2025年Q1采样")
    print(f"开始时间: {datetime.now()}")
    print("="*60)
    
    results = {}
    
    # 1. 东方财富新闻
    results['eastmoney_news'] = collect_eastmoney_news_quick()
    
    # 2. 新浪财经新闻
    results['sina_news'] = collect_sina_news_quick()
    
    # 3. 证券时报新闻
    results['stcn_news'] = collect_stcn_news_quick()
    
    # 4. 股吧评论
    results['guba_comments'] = collect_guba_comments_quick()
    
    # 5. 东方财富评论
    results['eastmoney_comments'] = collect_eastmoney_comments_quick()
    
    # 6. 雪球评论
    results['xueqiu_comments'] = collect_xueqiu_comments_quick()
    
    # 结果汇总
    print("\n" + "="*60)
    print("快速采集结果汇总")
    print("="*60)
    
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    for task, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {task}: {status}")
    
    print(f"\n总计: {success_count}/{total_count} 项成功")
    print(f"完成时间: {datetime.now()}")
    
    if success_count > 0:
        print(f"\n数据保存位置:")
        print(f"  新闻数据: {NEWS_DIR}")
        print(f"  情感数据: {SENTIMENT_DIR}")

if __name__ == "__main__":
    main()