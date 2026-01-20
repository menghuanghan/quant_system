"""
路径管理器集成示例

演示如何在各类采集器中使用统一路径管理工具
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

# 导入路径管理工具
from src.utils.paths import (
    get_announcement_metadata_path,
    get_announcement_file_path,
    get_news_path,
    get_report_metadata_path,
    get_report_pdf_path,
    get_sentiment_path,
    get_policy_rules_path,
    get_policy_file_path,
    get_event_meta_path,
    get_event_pdf_path,
    UnstructuredDataPaths
)

# 导入DataSink（可选）
from src.data_pipeline.collectors.unstructured.storage import (
    get_data_sink,
    StorageFormat
)


def example_announcement_collector():
    """
    示例：公告采集器使用路径管理
    
    旧方式：硬编码路径
    new_path = Path(f"data/raw/announcements/{year}/{month}/metadata.csv")
    
    新方式：使用路径管理器
    """
    print("\n" + "=" * 60)
    print("1. 公告采集器示例")
    print("=" * 60)
    
    # 模拟采集的公告数据
    announcements = pd.DataFrame({
        'ts_code': ['000001.SZ', '600519.SH'],
        'stock_name': ['平安银行', '贵州茅台'],
        'ann_date': ['2025-01-15', '2025-01-15'],
        'title': ['2024年度报告', '2024年度报告'],
        'pdf_url': ['http://example.com/1.pdf', 'http://example.com/2.pdf']
    })
    
    # 方式1: 保存元数据（Parquet格式）
    metadata_path = get_announcement_metadata_path(
        source='cninfo',
        year='2025',
        month='01',
        format='parquet'
    )
    print(f"元数据路径: {metadata_path}")
    announcements.to_parquet(metadata_path, compression='snappy', index=False)
    print(f"✓ 已保存 {len(announcements)} 条元数据")
    
    # 方式2: 使用DataSink（推荐）
    sink = get_data_sink()
    sink.save(
        data=announcements,
        domain="announcements",
        sub_domain="cninfo",
        partition_by="ann_date",
        format=StorageFormat.PARQUET
    )
    print("✓ 已使用DataSink保存（自动分区）")
    
    # 获取PDF存储路径
    for _, row in announcements.iterrows():
        pdf_path = get_announcement_file_path(
            ts_code=row['ts_code'],
            ann_date=row['ann_date'],
            filename=f"{row['ann_date']}_{row['title']}.pdf",
            use_stock_partition=True
        )
        print(f"  PDF路径: {pdf_path}")


def example_news_collector():
    """示例：新闻采集器使用路径管理"""
    print("\n" + "=" * 60)
    print("2. 新闻采集器示例")
    print("=" * 60)
    
    # 模拟采集的新闻数据
    news_items = [
        {
            'news_id': 'sina_20250115_001',
            'title': 'A股开门红',
            'content': '今日A股三大指数集体高开...',
            'publish_time': '2025-01-15 09:30:00',
            'source': 'sina'
        },
        {
            'news_id': 'sina_20250115_002',
            'title': '央行降准释放流动性',
            'content': '央行宣布降准0.5个百分点...',
            'publish_time': '2025-01-15 10:00:00',
            'source': 'sina'
        }
    ]
    
    # 保存到JSONL（按天归档）
    import json
    news_path = get_news_path(source='sina', date='20250115', format='jsonl')
    print(f"新闻路径: {news_path}")
    
    news_path.parent.mkdir(parents=True, exist_ok=True)
    with open(news_path, 'a', encoding='utf-8') as f:
        for item in news_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ 已保存 {len(news_items)} 条新闻到 JSONL")


def example_report_collector():
    """示例：研报采集器使用路径管理"""
    print("\n" + "=" * 60)
    print("3. 研报采集器示例")
    print("=" * 60)
    
    # 模拟采集的研报数据
    reports = pd.DataFrame({
        'ts_code': ['600519.SH', '600036.SH'],
        'stock_name': ['贵州茅台', '招商银行'],
        'report_date': ['20250115', '20250115'],
        'org_name': ['中信证券', '华泰证券'],
        'analyst': ['张三', '李四'],
        'rating': ['买入', '增持'],
        'target_price': [2000.0, 45.0]
    })
    
    # 保存元数据
    metadata_path = get_report_metadata_path(year='2025', month='01', format='parquet')
    print(f"元数据路径: {metadata_path}")
    reports.to_parquet(metadata_path, compression='snappy', index=False)
    print(f"✓ 已保存 {len(reports)} 条研报元数据")
    
    # 获取PDF路径
    for _, row in reports.iterrows():
        pdf_path = get_report_pdf_path(
            ts_code=row['ts_code'],
            report_date=row['report_date'],
            org_name=row['org_name'],
            rating=row['rating']
        )
        print(f"  PDF路径: {pdf_path}")


def example_sentiment_collector():
    """示例：舆情采集器使用路径管理"""
    print("\n" + "=" * 60)
    print("4. 舆情采集器示例")
    print("=" * 60)
    
    # 模拟采集的舆情数据（海量）
    comments = pd.DataFrame({
        'comment_id': [f'xq_20250115_{i:04d}' for i in range(1000)],
        'ts_code': ['000001.SZ'] * 500 + ['600519.SH'] * 500,
        'content': ['看多'] * 1000,
        'like_count': range(1000),
        'publish_time': ['2025-01-15 14:30:00'] * 1000,
        'source': ['xueqiu'] * 1000
    })
    
    # 保存到Parquet（按月归档，压缩存储）
    sentiment_path = get_sentiment_path(source='xueqiu', date='20250115', format='parquet')
    print(f"舆情路径: {sentiment_path}")
    
    comments.to_parquet(sentiment_path, compression='snappy', index=False)
    
    file_size_mb = sentiment_path.stat().st_size / 1024 / 1024
    print(f"✓ 已保存 {len(comments)} 条评论到 Parquet")
    print(f"  文件大小: {file_size_mb:.2f} MB")
    
    # 演示Parquet读取效率
    import time
    start = time.time()
    df = pd.read_parquet(sentiment_path)
    elapsed = time.time() - start
    print(f"  读取耗时: {elapsed:.3f} 秒")


def example_policy_collector():
    """示例：政策采集器使用路径管理"""
    print("\n" + "=" * 60)
    print("5. 政策采集器示例")
    print("=" * 60)
    
    # 模拟采集的政策数据
    policies = [
        {
            'policy_id': 'csrc_2025_001',
            'title': '关于加强上市公司监管的通知',
            'publish_date': '2025-01-15',
            'agency': 'csrc',
            'category': '部门规章',
            'content': '为进一步规范上市公司信息披露...',
            'attachments': ['附件1.pdf', '附件2.doc']
        }
    ]
    
    # 保存政策文本
    import json
    policy_path = get_policy_rules_path(agency='csrc', year='2025', format='jsonl')
    print(f"政策文本路径: {policy_path}")
    
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    with open(policy_path, 'a', encoding='utf-8') as f:
        for policy in policies:
            f.write(json.dumps(policy, ensure_ascii=False) + '\n')
    
    print(f"✓ 已保存 {len(policies)} 条政策文本")
    
    # 获取附件路径
    for policy in policies:
        for attachment in policy['attachments']:
            file_path = get_policy_file_path(
                agency='csrc',
                policy_id=policy['policy_id'],
                filename=attachment
            )
            print(f"  附件路径: {file_path}")


def example_event_collector():
    """示例：事件驱动采集器使用路径管理"""
    print("\n" + "=" * 60)
    print("6. 事件驱动采集器示例")
    print("=" * 60)
    
    # 模拟采集的事件数据
    events = pd.DataFrame({
        'id': ['md5_001', 'md5_002'],
        'ts_code': ['000001.SZ', '600519.SH'],
        'stock_name': ['平安银行', '贵州茅台'],
        'event_type': ['penalty', 'penalty'],
        'title': ['行政处罚决定书', '监管警示函'],
        'ann_date': ['2025-01-15', '2025-01-15'],
        'summary': ['因信披违规，罚款100万', '因XXX行为，警告']
    })
    
    # 保存元数据（按事件类型分类）
    meta_path = get_event_meta_path(event_type='penalty', year='2025', month='01', format='parquet')
    print(f"事件元数据路径: {meta_path}")
    events.to_parquet(meta_path, compression='snappy', index=False)
    print(f"✓ 已保存 {len(events)} 条事件元数据")
    
    # 获取PDF路径
    for _, row in events.iterrows():
        pdf_path = get_event_pdf_path(
            event_type='penalty',
            ts_code=row['ts_code'],
            ann_date=row['ann_date'],
            filename=f"{row['ts_code'].replace('.', '_')}_{row['title']}.pdf"
        )
        print(f"  PDF路径: {pdf_path}")


def example_storage_summary():
    """示例：查看存储统计"""
    print("\n" + "=" * 60)
    print("7. 存储统计")
    print("=" * 60)
    
    summary = UnstructuredDataPaths.get_storage_summary()
    
    import json
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    total_files = sum(s['file_count'] for s in summary.values())
    total_size_mb = sum(s['total_size_mb'] for s in summary.values())
    
    print(f"\n总计:")
    print(f"  文件数量: {total_files}")
    print(f"  总大小: {total_size_mb:.2f} MB")


def main():
    """运行所有示例"""
    print("=" * 60)
    print("路径管理器集成示例")
    print("=" * 60)
    
    try:
        example_announcement_collector()
        example_news_collector()
        example_report_collector()
        example_sentiment_collector()
        example_policy_collector()
        example_event_collector()
        example_storage_summary()
        
        print("\n" + "=" * 60)
        print("✓ 所有示例运行成功！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
