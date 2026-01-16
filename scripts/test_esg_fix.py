"""
测试ESG采集器修复
验证字段映射是否正确
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_pipeline.collectors.structured.deep_risk_quality.esg_ratings import (
    MSCIESGCollector,
    HZESGCollector,
    RefinitivESGCollector,
    ZhidingESGCollector
)

def test_collector_fields():
    """测试采集器的字段配置"""
    
    collectors = {
        'MSCI': MSCIESGCollector(),
        '华证': HZESGCollector(),
        '路孚特': RefinitivESGCollector(),
        '秩鼎': ZhidingESGCollector()
    }
    
    print("=" * 80)
    print("ESG采集器字段配置验证")
    print("=" * 80)
    
    for name, collector in collectors.items():
        print(f"\n{name} ESG采集器:")
        print(f"  OUTPUT_FIELDS: {collector.OUTPUT_FIELDS}")
        print(f"  字段数量: {len(collector.OUTPUT_FIELDS)}")
    
    print("\n" + "=" * 80)
    print("预期 AkShare 返回列名 (从源码):")
    print("=" * 80)
    
    expected_columns = {
        'MSCI': ['股票代码', 'ESG评分', '环境总评', '社会责任总评', '治理总评', '评级日期', '交易市场'],
        '华证': ['日期', '股票代码', '交易市场', '股票名称', 'ESG评分', 'ESG等级', '环境', '环境等级', '社会', '社会等级', '公司治理', '公司治理等级'],
        '路孚特': ['股票代码', 'ESG评分', 'ESG评分日期', '环境总评', '环境总评日期', '社会责任总评', '社会责任总评日期', '治理总评', '治理总评日期', '争议总评', '争议总评日期', '行业', '交易所'],
        '秩鼎': ['股票代码', 'ESG评分', '环境总评', '社会责任总评', '治理总评', '评分日期']
    }
    
    for name, columns in expected_columns.items():
        print(f"\n{name}:")
        print(f"  {columns}")
        print(f"  字段数量: {len(columns)}")


def test_column_mapping():
    """测试列名映射逻辑"""
    import pandas as pd
    from src.data_pipeline.collectors.structured.base import BaseCollector
    
    print("\n" + "=" * 80)
    print("测试列名映射功能")
    print("=" * 80)
    
    # 模拟 AkShare MSCI 数据
    msci_data = pd.DataFrame({
        '股票代码': ['000001', '000002'],
        'ESG评分': ['AAA', 'AA'],
        '环境总评': [85.0, 80.0],
        '社会责任总评': [90.0, 85.0],
        '治理总评': [88.0, 82.0],
        '评级日期': ['2025-12-31', '2025-12-31'],
        '交易市场': ['深圳', '深圳']
    })
    
    # 使用 MSCI 采集器测试
    collector = MSCIESGCollector()
    
    # 使用 _standardize_columns 方法
    column_mapping = {
        '股票代码': 'ts_code',
        'ESG评分': 'esg_rating',
        '环境总评': 'env_score',
        '社会责任总评': 'social_score',
        '治理总评': 'governance_score',
        '评级日期': 'rating_date',
        '交易市场': 'market',
    }
    
    result = collector._standardize_columns(msci_data.copy(), column_mapping)
    
    print("\n原始数据列名:")
    print(f"  {msci_data.columns.tolist()}")
    
    print("\n映射后列名:")
    print(f"  {result.columns.tolist()}")
    
    print("\n映射后数据样例:")
    print(result.head())
    
    # 检查是否包含所有必需字段
    missing_fields = set(collector.OUTPUT_FIELDS) - set(result.columns)
    if missing_fields:
        print(f"\n⚠️  缺失字段: {missing_fields}")
    else:
        print("\n✅ 所有字段都已正确映射!")


if __name__ == '__main__':
    test_collector_fields()
    test_column_mapping()
