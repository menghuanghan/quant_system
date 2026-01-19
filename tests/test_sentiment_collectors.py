"""
舆情与市场情绪模块测试脚本

测试内容：
1. ScraperBase 基础功能
2. MarketHeatCollector 市场热度采集
3. InvestorSentimentCollector 投资者舆情采集
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd


def test_scraper_base():
    """测试增强型爬虫基类"""
    print("\n" + "="*60)
    print("测试 1: ScraperBase 基础功能")
    print("="*60)
    
    from src.data_pipeline.collectors.unstructured.scraper_base import (
        ScraperBase,
        UserAgentManager,
        CookieManager,
        get_cookie_manager
    )
    
    # 1. 测试 UserAgentManager
    print("\n[1.1] UserAgentManager 测试")
    ua_manager = UserAgentManager(use_fake_ua=False)
    for i in range(3):
        ua = ua_manager.get_random()
        print(f"  Random UA {i+1}: {ua[:50]}...")
    
    # 2. 测试 CookieManager
    print("\n[1.2] CookieManager 测试")
    cookie_manager = get_cookie_manager()
    
    # 检查是否从环境变量加载了雪球Cookie
    xq_cookie = cookie_manager.get_cookie_string('xueqiu.com')
    if xq_cookie:
        print(f"  ✓ 雪球Cookie已加载: {xq_cookie[:30]}...")
    else:
        print("  ✗ 雪球Cookie未配置 (可在.env中设置 XUEQIU_COOKIE)")
    
    # 打印统计
    stats = cookie_manager.get_stats()
    print(f"  Cookie统计: {stats}")
    
    # 3. 测试 ScraperBase 请求
    print("\n[1.3] ScraperBase 请求测试")
    with ScraperBase(rate_limit=True) as scraper:
        # 简单请求测试
        response = scraper.get("https://httpbin.org/get", timeout=10)
        if response and response.status_code == 200:
            print("  ✓ HTTP GET 请求成功")
        else:
            print("  ✗ HTTP GET 请求失败")
    
    print("\n✓ ScraperBase 测试完成")
    return True


def test_market_heat():
    """测试市场热度采集器"""
    print("\n" + "="*60)
    print("测试 2: MarketHeatCollector 市场热度采集")
    print("="*60)
    
    from src.data_pipeline.collectors.unstructured.sentiment import (
        MarketHeatCollector,
        get_realtime_hotlist,
        get_market_heat
    )
    
    # 1. 测试东方财富实时热榜
    print("\n[2.1] 东方财富实时热榜")
    try:
        hotlist_em = get_realtime_hotlist(source='eastmoney')
        if not hotlist_em.empty:
            print(f"  ✓ 获取 {len(hotlist_em)} 条热榜数据")
            print(f"  Top 5:")
            print(hotlist_em[['rank', 'ts_code', 'name', 'search_index_proxy']].head())
        else:
            print("  ✗ 热榜数据为空")
    except Exception as e:
        print(f"  ✗ 采集失败: {e}")
    
    # 2. 测试雪球实时热榜
    print("\n[2.2] 雪球实时热榜")
    try:
        hotlist_xq = get_realtime_hotlist(source='xueqiu')
        if not hotlist_xq.empty:
            print(f"  ✓ 获取 {len(hotlist_xq)} 条热榜数据")
            print(f"  Top 5:")
            print(hotlist_xq[['rank', 'ts_code', 'name', 'search_index_proxy']].head())
        else:
            print("  ✗ 热榜数据为空")
    except Exception as e:
        print(f"  ✗ 采集失败: {e}")
    
    print("\n✓ MarketHeatCollector 测试完成")
    return True


def test_investor_sentiment():
    """测试投资者舆情采集器"""
    print("\n" + "="*60)
    print("测试 3: InvestorSentimentCollector 投资者舆情采集")
    print("="*60)
    
    from src.data_pipeline.collectors.unstructured.sentiment import (
        InvestorSentimentCollector,
        get_cninfo_interaction,
        get_guba_comments,
        SentimentConfig
    )
    
    test_codes = ['000001.SZ']  # 平安银行
    
    # 1. 测试互动易采集
    print("\n[3.1] 互动易问答采集")
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        interaction_df = get_cninfo_interaction(
            ts_codes=test_codes,
            start_date=start_date,
            end_date=end_date
        )
        
        if not interaction_df.empty:
            print(f"  ✓ 获取 {len(interaction_df)} 条互动易数据")
            print(f"  字段: {list(interaction_df.columns)}")
        else:
            print("  ✗ 互动易数据为空 (可能需要Tushare积分)")
    except Exception as e:
        print(f"  ✗ 采集失败: {e}")
    
    # 2. 测试股吧采集
    print("\n[3.2] 股吧评论采集")
    try:
        config = SentimentConfig(max_pages=2, event_driven=False)
        collector = InvestorSentimentCollector(config=config)
        
        guba_df = collector._collect_guba_comments(
            start_date=datetime.now().strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d'),
            ts_codes=test_codes
        )
        
        if not guba_df.empty:
            print(f"  ✓ 获取 {len(guba_df)} 条股吧评论")
            if 'title' in guba_df.columns:
                print(f"  示例标题: {guba_df['title'].iloc[0][:30] if len(guba_df['title'].iloc[0]) > 30 else guba_df['title'].iloc[0]}...")
        else:
            print("  ✗ 股吧数据为空")
    except Exception as e:
        print(f"  ✗ 采集失败: {e}")
    
    # 3. 测试事件驱动筛选
    print("\n[3.3] 事件驱动日期筛选")
    try:
        # 模拟股价波动表
        volatility_df = pd.DataFrame({
            'ts_code': ['000001.SZ'] * 5,
            'trade_date': ['2024-06-01', '2024-06-10', '2024-06-15', '2024-06-20', '2024-06-25'],
            'pct_chg': [1.5, 5.2, -0.8, -4.1, 2.0],
            'volume_ratio': [1.2, 1.8, 0.9, 2.5, 1.1]
        })
        
        collector = InvestorSentimentCollector()
        target_dates = collector._filter_event_dates(volatility_df, test_codes)
        
        print(f"  输入: {len(volatility_df)} 个交易日")
        print(f"  筛选结果: {target_dates}")
        print(f"  ✓ 事件驱动筛选成功，保留 {sum(len(v) for v in target_dates.values())} 个异常日期")
    except Exception as e:
        print(f"  ✗ 筛选失败: {e}")
    
    print("\n✓ InvestorSentimentCollector 测试完成")
    return True


def test_storage():
    """测试存储功能"""
    print("\n" + "="*60)
    print("测试 4: 存储功能 (JSONL / Parquet)")
    print("="*60)
    
    from src.data_pipeline.collectors.unstructured.base import UnstructuredCollector
    
    # 创建测试数据
    test_df = pd.DataFrame({
        'ts_code': ['000001.SZ', '600519.SH', '000002.SZ'],
        'trade_date': ['2024-01-01', '2024-01-01', '2024-01-02'],
        'title': ['测试标题1', '测试标题2', '测试标题3'],
        'content': ['测试内容1', '测试内容2', '测试内容3'],
        'ann_date': ['2024-01-01', '2024-01-01', '2024-01-02'],
        'name': ['平安银行', '贵州茅台', '万科A'],
        'ann_time': ['09:00:00', '09:30:00', '10:00:00'],
        'category': ['公告', '公告', '公告'],
        'url': ['http://test1', 'http://test2', 'http://test3'],
        'source': ['test', 'test', 'test'],
        'is_correction': [False, False, False],
        'correction_of': [None, None, None],
        'list_status': ['L', 'L', 'L'],
        'original_id': ['1', '2', '3']
    })
    
    # 创建临时收集器
    class TestCollector(UnstructuredCollector):
        def collect(self, start_date, end_date, **kwargs):
            return test_df
    
    collector = TestCollector()
    
    # 测试目录
    test_dir = Path(project_root) / 'data' / 'raw' / 'test_storage'
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 测试 JSONL 保存
    print("\n[4.1] JSONL 保存测试")
    try:
        jsonl_path = test_dir / 'test.jsonl'
        success = collector._save_to_jsonl(test_df, jsonl_path, append=False)
        if success:
            print(f"  ✓ JSONL 保存成功: {jsonl_path}")
            # 读取验证
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"  已保存 {len(lines)} 条记录")
        else:
            print("  ✗ JSONL 保存失败")
    except Exception as e:
        print(f"  ✗ JSONL 保存失败: {e}")
    
    # 2. 测试 Parquet 保存
    print("\n[4.2] Parquet 保存测试")
    try:
        parquet_path = test_dir / 'test.parquet'
        success = collector._save_to_parquet(test_df, parquet_path)
        if success:
            print(f"  ✓ Parquet 保存成功: {parquet_path}")
            # 读取验证
            read_df = collector._read_parquet(parquet_path)
            print(f"  已保存 {len(read_df)} 条记录")
        else:
            print("  ✗ Parquet 保存失败 (可能需要安装 pyarrow)")
    except Exception as e:
        print(f"  ✗ Parquet 保存失败: {e}")
    
    print("\n✓ 存储功能测试完成")
    return True


def main():
    """运行所有测试"""
    print("="*60)
    print("舆情与市场情绪模块测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = []
    
    # 运行测试
    tests = [
        ("ScraperBase", test_scraper_base),
        ("MarketHeat", test_market_heat),
        ("InvestorSentiment", test_investor_sentiment),
        ("Storage", test_storage),
    ]
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✓ 通过" if success else "✗ 失败"))
        except Exception as e:
            print(f"\n✗ {name} 测试异常: {e}")
            results.append((name, f"✗ 异常: {str(e)[:30]}"))
    
    # 汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    for name, result in results:
        print(f"  {name}: {result}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
