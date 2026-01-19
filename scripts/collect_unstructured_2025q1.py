"""
非结构化数据采集脚本 - 2025年Q1
采集范围：2025年1月1日 - 2025年3月31日

采集内容：
1. 东方财富新闻
2. 新浪财经新闻
3. 证券时报新闻
4. 股吧评论（含内容）
5. 东方财富评论
6. 互动易问答
7. 雪球评论
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

# 采集时间范围
START_DATE = "20250101"
END_DATE = "20250331"

def collect_eastmoney_news():
    """采集东方财富新闻"""
    print("\n" + "="*60)
    print("1. 采集东方财富新闻")
    print("="*60)
    
    try:
        from src.data_pipeline.collectors.unstructured.news.eastmoney_collector import (
            EastMoneyNewsCollector
        )
        
        collector = EastMoneyNewsCollector()
        
        # 注意：大部分新闻API只返回近期数据
        # 扩大日期范围以获取更多数据
        df = collector.collect(
            start_date='20240101',  # 扩大日期范围
            end_date='20261231',
            channels=['7x24', 'cjyw', 'stock']
        )
        
        if not df.empty:
            output_file = NEWS_DIR / "eastmoney_news_2025Q1.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"✓ 采集完成: {len(df)} 条新闻")
            if 'pub_date' in df.columns:
                valid_dates = df['pub_date'].dropna()
                if len(valid_dates) > 0:
                    print(f"  数据日期范围: {valid_dates.min()} ~ {valid_dates.max()}")
            print(f"  保存至: {output_file}")
            return len(df)
        else:
            print("✗ 采集失败 - 无数据")
            return 0
            
    except Exception as e:
        print(f"✗ 采集失败: {e}")
        import traceback
        traceback.print_exc()
        return 0


def collect_sina_news():
    """采集新浪财经新闻"""
    print("\n" + "="*60)
    print("2. 采集新浪财经新闻")
    print("="*60)
    
    try:
        from src.data_pipeline.collectors.unstructured.news.sina_crawler import (
            SinaFinanceCrawler
        )
        
        crawler = SinaFinanceCrawler()
        
        # 注意：新浪财经滚动新闻API只能获取近期数据
        # 对于历史数据，需要使用搜索API或其他方式
        # 这里先采集可用数据，不严格过滤日期
        df = crawler.collect(
            start_date='20240101',  # 扩大日期范围
            end_date='20261231',
            categories=['stock', 'finance', 'fund']
        )
        
        if not df.empty:
            output_file = NEWS_DIR / "sina_news_2025Q1.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"✓ 采集完成: {len(df)} 条新闻")
            print(f"  数据日期范围: {df['pub_date'].min()} ~ {df['pub_date'].max()}")
            print(f"  保存至: {output_file}")
            print("  注意: 新浪财经API只返回近期数据，历史数据需要其他方式获取")
            return len(df)
        else:
            print("✗ 采集失败 - 无数据")
            return 0
            
    except Exception as e:
        print(f"✗ 采集失败: {e}")
        import traceback
        traceback.print_exc()
        return 0


def collect_stcn_news():
    """采集证券时报新闻"""
    print("\n" + "="*60)
    print("3. 采集证券时报新闻")
    print("="*60)
    
    try:
        from src.data_pipeline.collectors.unstructured.news.stcn_crawler import (
            STCNCrawler
        )
        
        crawler = STCNCrawler()
        
        # 先尝试HTTP采集
        df = crawler.collect(
            start_date=START_DATE,
            end_date=END_DATE,
            channels=['sd', 'kx', 'cj'],
            use_playwright=False
        )
        
        # 如果HTTP采集数据少，尝试Playwright
        if len(df) < 50:
            print("  HTTP采集数据较少，尝试Playwright...")
            df_pw = crawler.collect(
                start_date=START_DATE,
                end_date=END_DATE,
                channels=['sd', 'kx'],
                use_playwright=True
            )
            if len(df_pw) > len(df):
                df = df_pw
        
        if not df.empty:
            output_file = NEWS_DIR / "stcn_news_2025Q1.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"✓ 采集完成: {len(df)} 条新闻")
            print(f"  保存至: {output_file}")
            return len(df)
        else:
            print("⚠ 采集数据为空（网站可能返回历史缓存数据）")
            return 0
            
    except Exception as e:
        print(f"✗ 采集失败: {e}")
        import traceback
        traceback.print_exc()
        return 0


def collect_guba_comments():
    """采集股吧评论（含内容）"""
    print("\n" + "="*60)
    print("4. 采集股吧评论（含内容）")
    print("="*60)
    
    try:
        from src.data_pipeline.collectors.unstructured.sentiment.investor_sentiment import (
            InvestorSentimentCollector,
            SentimentSource
        )
        
        # 热门股票列表
        hot_stocks = [
            '600519.SH',  # 贵州茅台
            '000001.SZ',  # 平安银行
            '000858.SZ',  # 五粮液
            '601318.SH',  # 中国平安
            '000651.SZ',  # 格力电器
        ]
        
        collector = InvestorSentimentCollector()
        df = collector.collect(
            start_date=START_DATE,
            end_date=END_DATE,
            ts_codes=hot_stocks,
            source=SentimentSource.GUBA
        )
        
        if not df.empty:
            output_file = SENTIMENT_DIR / "guba_comments_2025Q1.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 统计内容
            has_content = df['content'].astype(str).str.len() > 10
            content_count = has_content.sum()
            
            print(f"✓ 采集完成: {len(df)} 条评论")
            print(f"  有内容: {content_count} 条 ({content_count/len(df)*100:.1f}%)")
            print(f"  保存至: {output_file}")
            return len(df)
        else:
            print("✗ 采集失败 - 无数据")
            return 0
            
    except Exception as e:
        print(f"✗ 采集失败: {e}")
        import traceback
        traceback.print_exc()
        return 0


def collect_eastmoney_comments():
    """采集东方财富评论 (EastMoneyCollector)"""
    print("\n" + "="*60)
    print("5. 采集东方财富评论")
    print("="*60)
    
    try:
        from src.data_pipeline.collectors.unstructured.scraper_base import EastMoneyCollector
        import pandas as pd
        
        hot_stocks = [
            '600519',  # 贵州茅台
            '000001',  # 平安银行
            '000858',  # 五粮液
        ]
        
        collector = EastMoneyCollector()
        all_data = []
        
        for symbol in hot_stocks:
            print(f"  采集 {symbol}...")
            result = collector.get_guba_comments(symbol, max_pages=5)
            # result 是 List[Dict]
            if result:
                df = pd.DataFrame(result)
                df['symbol'] = symbol
                all_data.append(df)
                print(f"    {symbol}: {len(df)} 条")
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            output_file = SENTIMENT_DIR / "eastmoney_comments_2025Q1.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            has_content = df['content'].astype(str).str.len() > 10
            content_count = has_content.sum()
            
            print(f"✓ 采集完成: {len(df)} 条评论")
            print(f"  有内容: {content_count} 条")
            print(f"  保存至: {output_file}")
            return len(df)
        else:
            print("✗ 采集失败 - 无数据")
            return 0
            
    except Exception as e:
        print(f"✗ 采集失败: {e}")
        import traceback
        traceback.print_exc()
        return 0


def collect_cninfo_interaction():
    """采集互动易问答"""
    print("\n" + "="*60)
    print("6. 采集互动易问答")
    print("="*60)
    
    try:
        from src.data_pipeline.collectors.unstructured.sentiment.investor_sentiment import (
            InvestorSentimentCollector,
            SentimentSource
        )
        
        hot_stocks = [
            '600519.SH',
            '000001.SZ',
            '000858.SZ',
        ]
        
        collector = InvestorSentimentCollector()
        df = collector.collect(
            start_date=START_DATE,
            end_date=END_DATE,
            ts_codes=hot_stocks,
            source=SentimentSource.CNINFO_INTERACTION  # 修正：使用正确的枚举值
        )
        
        if not df.empty:
            output_file = SENTIMENT_DIR / "cninfo_interaction_2025Q1.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"✓ 采集完成: {len(df)} 条问答")
            print(f"  保存至: {output_file}")
            return len(df)
        else:
            print("⚠ 无数据（需要Tushare高级权限或接口变化）")
            return 0
            
    except Exception as e:
        print(f"✗ 采集失败: {e}")
        import traceback
        traceback.print_exc()
        return 0


def collect_xueqiu_comments():
    """采集雪球评论"""
    print("\n" + "="*60)
    print("7. 采集雪球评论")
    print("="*60)
    
    try:
        from src.data_pipeline.collectors.unstructured.scraper_base import XueqiuCollector
        import pandas as pd
        
        hot_stocks = [
            'SH600519',  # 贵州茅台
            'SZ000001',  # 平安银行
            'SZ000858',  # 五粮液
        ]
        
        collector = XueqiuCollector()
        all_data = []
        
        for symbol in hot_stocks:
            print(f"  采集 {symbol}...")
            result = collector.get_stock_comments(symbol, count=50)
            # result 是 List[Dict]
            if result:
                df = pd.DataFrame(result)
                df['symbol'] = symbol
                all_data.append(df)
                print(f"    {symbol}: {len(df)} 条")
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            output_file = SENTIMENT_DIR / "xueqiu_comments_2025Q1.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"✓ 采集完成: {len(df)} 条评论")
            print(f"  保存至: {output_file}")
            return len(df)
        else:
            print("⚠ 无数据（可能需要重新获取Cookie）")
            return 0
            
    except Exception as e:
        print(f"✗ 采集失败: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    """主函数"""
    print("="*60)
    print("非结构化数据采集 - 2025年Q1")
    print(f"采集时间范围: {START_DATE} ~ {END_DATE}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {}
    
    # 1. 东方财富新闻
    results['eastmoney_news'] = collect_eastmoney_news()
    
    # 2. 新浪财经新闻
    results['sina_news'] = collect_sina_news()
    
    # 3. 证券时报新闻
    results['stcn_news'] = collect_stcn_news()
    
    # 4. 股吧评论
    results['guba_comments'] = collect_guba_comments()
    
    # 5. 东方财富评论
    results['eastmoney_comments'] = collect_eastmoney_comments()
    
    # 6. 互动易问答
    results['cninfo_interaction'] = collect_cninfo_interaction()
    
    # 7. 雪球评论
    results['xueqiu_comments'] = collect_xueqiu_comments()
    
    # 汇总
    print("\n" + "="*60)
    print("采集结果汇总")
    print("="*60)
    
    total = 0
    for name, count in results.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {name}: {count} 条")
        total += count
    
    print(f"\n总计采集: {total} 条数据")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == "__main__":
    main()
