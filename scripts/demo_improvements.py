"""
非结构化数据采集模块改进示例

展示：
1. DataSink统一存储接口
2. PDF完整性校验
3. 健康检查机制
4. 文本相似度匹配
5. Mobile UserAgent
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

# ============== 1. DataSink 统一存储示例 ==============

def demo_data_sink():
    """演示DataSink的使用"""
    from src.data_pipeline.collectors.unstructured.storage import (
        DataSink,
        StorageFormat,
        CompressionType
    )
    
    print("="*70)
    print("1. DataSink 统一存储示例")
    print("="*70)
    
    # 创建DataSink实例
    sink = DataSink(
        base_path="data/raw/unstructured",
        default_format=StorageFormat.PARQUET,
        compression=CompressionType.SNAPPY,
        enable_backup=True  # 启用JSONL备份
    )
    
    # 示例数据
    data = pd.DataFrame({
        'ts_code': ['000001.SZ', '600000.SH'],
        'event_type': ['merger', 'penalty'],
        'title': ['并购公告', '处罚公告'],
        'ann_date': ['2024-12-01', '2024-12-02'],
        'content': ['内容1', '内容2']
    })
    
    # 保存为Parquet（默认格式）
    path = sink.save(
        data=data,
        domain="events",
        sub_domain="penalty",
        partition_by="ann_date",  # 按日期分区
        filename="events_demo"
    )
    print(f"✓ 数据已保存到: {path}")
    
    # 读取数据
    loaded_data = sink.load(
        domain="events",
        sub_domain="penalty",
        filters={'year': 2024, 'month': 12}  # 分区过滤
    )
    print(f"✓ 读取了 {len(loaded_data)} 条记录")
    
    # 存储统计
    stats = sink.get_storage_stats(domain="events")
    print("\n存储统计:")
    print(f"  总文件数: {stats['total_files']}")
    print(f"  总大小: {stats['total_size_mb']:.2f} MB")
    for fmt, info in stats['by_format'].items():
        print(f"  {fmt}: {info['count']} 文件, {info['size_mb']:.2f} MB")
    
    print()


# ============== 2. PDF完整性校验示例 ==============

def demo_pdf_validation():
    """演示PDF完整性校验"""
    from src.data_pipeline.collectors.unstructured.pdf_utils import (
        download_pdf_with_validation,
        PDFValidator
    )
    
    print("="*70)
    print("2. PDF完整性校验示例")
    print("="*70)
    
    # 示例PDF链接（巨潮资讯）
    test_url = "http://static.cninfo.com.cn/finalpage/2024-12-31/1222963269.PDF"
    save_path = Path("data/temp/test_validation.pdf")
    
    # 下载并校验
    result = download_pdf_with_validation(
        url=test_url,
        save_path=save_path,
        check_scanned=True
    )
    
    if result['success']:
        print("✓ PDF下载成功")
        metadata = result['metadata']
        print(f"  文件大小: {metadata['file_size_mb']:.2f} MB")
        print(f"  页数: {metadata['page_count']}")
        print(f"  是否有效: {metadata['is_valid']}")
        print(f"  是否扫描件: {metadata['is_scanned']}")
        if metadata.get('is_scanned'):
            print("  ⚠️ 检测到扫描件PDF，需要OCR处理")
    else:
        print(f"✗ PDF下载失败: {result['error']}")
    
    print()


# ============== 3. 健康检查机制示例 ==============

def demo_health_check():
    """演示健康检查机制"""
    from src.data_pipeline.collectors.unstructured.scraper_base import ScraperBase
    
    print("="*70)
    print("3. 健康检查机制示例")
    print("="*70)
    
    scraper = ScraperBase(enable_health_check=True)
    
    # 正常请求
    try:
        response = scraper.get("https://www.baidu.com")
        print("✓ 正常请求成功")
    except Exception as e:
        print(f"请求失败: {e}")
    
    # 查看健康状态
    health = scraper.get_health_status()
    print(f"\n健康状态:")
    print(f"  总请求数: {health['total_requests']}")
    print(f"  失败请求数: {health['failed_requests']}")
    print(f"  成功率: {health['success_rate']:.1f}%")
    print(f"  连续失败: {health['consecutive_failures']}")
    print(f"  是否暂停: {health['is_paused']}")
    
    # 健康检查
    is_healthy = scraper.health_check()
    print(f"\n健康检查: {'✓ 健康' if is_healthy else '✗ 不健康'}")
    
    # 模拟连续失败（触发暂停）
    print("\n模拟连续失败场景...")
    for i in range(12):
        scraper._record_failure(status_code=403, error="Forbidden")
    
    health = scraper.get_health_status()
    print(f"  连续失败: {health['consecutive_failures']}")
    print(f"  是否暂停: {health['is_paused']}")
    if health['is_paused']:
        print(f"  暂停原因: {health['pause_reason']}")
        print("\n  调用 scraper.resume() 可恢复运行")
    
    print()


# ============== 4. 文本相似度匹配示例 ==============

def demo_text_matching():
    """演示文本相似度匹配（用于事件对齐）"""
    from src.data_pipeline.collectors.unstructured.text_matcher import TextMatcher
    
    print("="*70)
    print("4. 文本相似度匹配示例")
    print("="*70)
    
    # 示例：巨潮公告标题 vs 东财事件标题
    cninfo_title = "关于收购资产暨关联交易的补充公告"
    eastmoney_titles = [
        "公司发布重大资产重组预案",
        "关于资产收购的补充说明公告",  # 最匹配
        "关于股权转让的提示性公告",
        "独立董事关于关联交易的独立意见"
    ]
    
    print(f"查询: {cninfo_title}")
    print(f"\n候选标题:")
    for i, title in enumerate(eastmoney_titles):
        print(f"  [{i}] {title}")
    
    # 使用混合相似度找最佳匹配
    match_idx, score = TextMatcher.find_best_match(
        query=cninfo_title,
        candidates=eastmoney_titles,
        threshold=0.5,
        method='hybrid'
    )
    
    if match_idx is not None:
        print(f"\n✓ 最佳匹配: [{match_idx}] {eastmoney_titles[match_idx]}")
        print(f"  相似度分数: {score:.3f}")
    else:
        print("\n✗ 未找到匹配（相似度低于阈值）")
    
    # 对比不同算法
    print("\n不同算法对比:")
    methods = ['levenshtein', 'jaccard', 'sequence', 'hybrid']
    for method in methods:
        if method == 'jaccard':
            score = TextMatcher.jaccard_similarity(cninfo_title, eastmoney_titles[1], ngram=2)
        elif method == 'hybrid':
            score = TextMatcher.hybrid_similarity(cninfo_title, eastmoney_titles[1])
        else:
            func = getattr(TextMatcher, f'{method}_similarity')
            score = func(cninfo_title, eastmoney_titles[1])
        print(f"  {method:12s}: {score:.3f}")
    
    print()


# ============== 5. Mobile UserAgent 示例 ==============

def demo_mobile_ua():
    """演示移动端UserAgent（反爬更宽松）"""
    from src.data_pipeline.collectors.unstructured.scraper_base import UserAgentManager
    
    print("="*70)
    print("5. Mobile UserAgent 示例")
    print("="*70)
    
    ua_manager = UserAgentManager()
    
    # 桌面端UA
    print("桌面端 User-Agent:")
    for i in range(3):
        ua = ua_manager.get_random()
        print(f"  {ua[:80]}...")
    
    # 移动端UA
    print("\n移动端 User-Agent (反爬通常更宽松):")
    for i in range(3):
        ua = ua_manager.get_mobile()
        print(f"  {ua[:80]}...")
    
    # 使用场景说明
    print("\n使用建议:")
    print("  - 雪球、东方财富的移动端H5页面反爬更宽松")
    print("  - 可在请求失败时切换到移动端UA重试")
    print("  - 部分API只允许移动端访问")
    
    print()


# ============== 6. 综合示例：改进后的事件采集 ==============

def demo_improved_event_collection():
    """演示改进后的事件采集流程"""
    print("="*70)
    print("6. 改进后的事件采集示例")
    print("="*70)
    
    print("""
改进点汇总:

1. 统一存储 (DataSink)
   - Parquet格式 + Snappy压缩 (GB级数据高效)
   - 按日期自动分区
   - JSONL流式备份
   
2. PDF完整性校验
   - 下载时验证Content-Length
   - 检测PDF头尾完整性
   - 自动识别扫描件（需OCR）
   
3. 健康检查 (HealthCheck)
   - 连续失败自动暂停
   - 403/404高错误率预警
   - 429限流自动避让
   - 可扩展报警通知 (Sentry/Email/企业微信)
   
4. 文本相似度对齐
   - 混合算法 (Levenshtein + Jaccard + Sequence + Keyword)
   - 解决"同一天多条公告"的对齐问题
   - 替代单纯的日期匹配
   
5. Mobile UserAgent
   - 支持移动端UA (Android/iOS)
   - 反爬更宽松
   - 适配微信WebView
   
6. 代码优化
   - Edge浏览器不再排除 (Chromium内核兼容性好)
   - 增强错误处理
   - 模块化设计
    """)
    
    print("示例代码片段:")
    print("""
# 使用改进后的采集器
from src.data_pipeline.collectors.unstructured import ScraperBase
from src.data_pipeline.collectors.unstructured.storage import get_data_sink
from src.data_pipeline.collectors.unstructured.pdf_utils import download_pdf_with_validation

scraper = ScraperBase(enable_health_check=True)
sink = get_data_sink()

# 采集事件
events = []
for page in range(1, 51):
    # 健康检查
    if not scraper.health_check():
        print("健康检查失败，暂停采集")
        break
    
    # 发送请求（自动记录健康状态）
    response = scraper.post(url, data=payload)
    events.extend(parse_events(response))
    
    # 下载PDF（带校验）
    for event in events:
        result = download_pdf_with_validation(
            url=event['pdf_url'],
            save_path=event['save_path']
        )
        event['pdf_valid'] = result['success']
        event['is_scanned'] = result.get('metadata', {}).get('is_scanned')

# 保存数据（Parquet + 分区）
df = pd.DataFrame(events)
sink.save(
    data=df,
    domain="events",
    sub_domain="penalty",
    partition_by="ann_date",
    format=StorageFormat.PARQUET
)
    """)
    
    print()


def main():
    """运行所有示例"""
    print("\n")
    print("*"*70)
    print("非结构化数据采集模块改进示例")
    print("*"*70)
    print()
    
    try:
        # demo_data_sink()
        # demo_pdf_validation()
        demo_health_check()
        demo_text_matching()
        demo_mobile_ua()
        demo_improved_event_collection()
        
        print("="*70)
        print("✓ 所有示例运行完成")
        print("="*70)
        
    except Exception as e:
        import traceback
        print(f"\n错误: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
