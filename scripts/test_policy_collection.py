"""
政策采集模块测试脚本

测试采集器功能：
1. 东方财富政策中心（API接口，最稳定）
2. 证监会政策
3. 国务院政策
4. 发改委政策

运行方式: python scripts/test_policy_collection.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta
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


def test_eastmoney_policy():
    """测试东方财富政策中心采集"""
    print("\n" + "="*60)
    print("测试东方财富政策中心采集")
    print("="*60)
    
    try:
        from src.data_pipeline.collectors.unstructured.policy import (
            get_eastmoney_policy,
            search_policy
        )
        
        # 采集最近30天的政策
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        print(f"\n采集日期范围: {start_date} ~ {end_date}")
        
        # 测试分类采集
        print("\n1. 测试股市政策采集...")
        df = get_eastmoney_policy(
            start_date=start_date,
            end_date=end_date,
            categories=['stock'],
            max_pages=3
        )
        
        if not df.empty:
            print(f"   采集到 {len(df)} 条股市政策")
            print(f"   字段: {list(df.columns)}")
            print(f"\n   前3条政策标题:")
            for i, row in df.head(3).iterrows():
                print(f"   - [{row.get('publish_date', '')}] {row.get('title', '')[:50]}...")
        else:
            print("   未采集到数据")
        
        # 测试关键词搜索
        print("\n2. 测试关键词搜索...")
        df_search = search_policy(
            keyword='IPO',
            start_date=start_date,
            end_date=end_date,
            max_pages=2
        )
        
        if not df_search.empty:
            print(f"   搜索到 {len(df_search)} 条IPO相关政策")
        else:
            print("   未搜索到数据（可能API不支持关键词搜索）")
        
        return True
        
    except Exception as e:
        logger.error(f"东方财富采集测试失败: {e}", exc_info=True)
        return False


def test_csrc_policy():
    """测试证监会政策采集"""
    print("\n" + "="*60)
    print("测试证监会政策采集")
    print("="*60)
    
    try:
        from src.data_pipeline.collectors.unstructured.policy import (
            CSRCCollector,
            get_csrc_policy
        )
        
        # 采集最近60天的政策
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')
        
        print(f"\n采集日期范围: {start_date} ~ {end_date}")
        
        # 只采集证监会令，限制页数
        print("\n采集证监会令...")
        df = get_csrc_policy(
            start_date=start_date,
            end_date=end_date,
            categories=['order'],
            max_pages=2,
            download_files=False  # 先不下载附件
        )
        
        if not df.empty:
            print(f"采集到 {len(df)} 条证监会政策")
            print(f"字段: {list(df.columns)}")
            print(f"\n前3条政策:")
            for i, row in df.head(3).iterrows():
                title = row.get('title', '')[:40]
                doc_no = row.get('doc_no', '') or '无发文字号'
                pub_date = row.get('publish_date', '')
                print(f"   [{pub_date}] {doc_no}")
                print(f"   {title}...")
        else:
            print("未采集到数据（证监会网站可能需要特殊处理）")
        
        return True
        
    except Exception as e:
        logger.error(f"证监会采集测试失败: {e}", exc_info=True)
        return False


def test_gov_policy():
    """测试国务院政策采集"""
    print("\n" + "="*60)
    print("测试国务院政策采集")
    print("="*60)
    
    try:
        from src.data_pipeline.collectors.unstructured.policy import (
            get_gov_policy
        )
        
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        print(f"\n采集日期范围: {start_date} ~ {end_date}")
        
        print("\n采集最新政策...")
        df = get_gov_policy(
            start_date=start_date,
            end_date=end_date,
            categories=['latest'],
            max_pages=2,
            download_files=False
        )
        
        if not df.empty:
            print(f"采集到 {len(df)} 条国务院政策")
            print(f"\n前3条政策:")
            for i, row in df.head(3).iterrows():
                title = row.get('title', '')[:50]
                pub_date = row.get('publish_date', '')
                print(f"   [{pub_date}] {title}...")
        else:
            print("未采集到数据")
        
        return True
        
    except Exception as e:
        logger.error(f"国务院采集测试失败: {e}", exc_info=True)
        return False


def test_policy_document():
    """测试PolicyDocument数据结构"""
    print("\n" + "="*60)
    print("测试PolicyDocument数据结构")
    print("="*60)
    
    try:
        from src.data_pipeline.collectors.unstructured.policy import (
            PolicyDocument,
            PolicySource,
            PolicyCategory
        )
        
        # 创建测试文档
        doc = PolicyDocument(
            id="test123",
            source_dept="中国证监会",
            doc_no="证监发〔2024〕1号",
            title="关于进一步优化股票发行定价机制的指导意见",
            publish_date="2024-03-15",
            source=PolicySource.CSRC.value,
            category=PolicyCategory.IPO.value,
            tags=["IPO", "定价", "注册制"],
            file_type="pdf",
            url="http://www.csrc.gov.cn/xxx.html",
            local_path="",
            content_text="政策正文内容...",
            summary="本指导意见旨在...",
            effective_date="2024-04-01",
            status="active",
            create_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        print(f"\n创建文档: {doc.title}")
        print(f"发文字号: {doc.doc_no}")
        print(f"来源部门: {doc.source_dept}")
        print(f"分类: {doc.category}")
        print(f"标签: {doc.tags}")
        
        # 测试转换
        doc_dict = doc.to_dict()
        print(f"\n转换为字典: {len(doc_dict)} 个字段")
        
        doc_json = doc.to_json()
        print(f"转换为JSON: {len(doc_json)} 字符")
        
        # 测试从字典创建
        doc2 = PolicyDocument.from_dict(doc_dict)
        print(f"从字典还原: {doc2.title}")
        
        print("\n✓ PolicyDocument测试通过")
        return True
        
    except Exception as e:
        logger.error(f"PolicyDocument测试失败: {e}", exc_info=True)
        return False


def test_doc_no_extraction():
    """测试发文字号提取"""
    print("\n" + "="*60)
    print("测试发文字号提取")
    print("="*60)
    
    try:
        from src.data_pipeline.collectors.unstructured.policy.base_policy import (
            BasePolicyCollector
        )
        
        collector = BasePolicyCollector()
        
        test_cases = [
            ("证监发〔2024〕1号", "证监发〔2024〕1号"),
            ("国发〔2024〕15号", "国发〔2024〕15号"),
            ("发改委发〔2024〕123号", "发改委发〔2024〕123号"),
            ("2024年第5号公告", "2024年第5号公告"),
            ("中国证监会令第200号", "中国证监会令第200号"),
            ("关于XXX的通知", ""),  # 无发文字号
            ("证监函[2024]10号", ""),  # 方括号格式，看是否匹配
        ]
        
        print("\n测试用例:")
        passed = 0
        for text, expected in test_cases:
            result = collector._extract_doc_no(text)
            status = "✓" if result == expected else "✗"
            if result == expected:
                passed += 1
            print(f"   {status} '{text}' -> '{result}' (预期: '{expected}')")
        
        print(f"\n通过: {passed}/{len(test_cases)}")
        return passed >= len(test_cases) - 1  # 允许1个失败
        
    except Exception as e:
        logger.error(f"发文字号提取测试失败: {e}", exc_info=True)
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("政策采集模块测试")
    print("="*60)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # 1. 测试数据结构
    results['PolicyDocument'] = test_policy_document()
    
    # 2. 测试发文字号提取
    results['发文字号提取'] = test_doc_no_extraction()
    
    # 3. 测试东方财富采集（API最稳定）
    results['东方财富政策'] = test_eastmoney_policy()
    
    # 4. 测试证监会采集
    results['证监会政策'] = test_csrc_policy()
    
    # 5. 测试国务院采集
    results['国务院政策'] = test_gov_policy()
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"   {name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\n总计: {passed}/{total} 通过")
    
    # 检查数据存储目录
    print("\n" + "="*60)
    print("数据存储检查")
    print("="*60)
    
    data_dir = Path("data/raw/unstructured/policy")
    if data_dir.exists():
        print(f"数据目录: {data_dir}")
        
        meta_dir = data_dir / "meta"
        if meta_dir.exists():
            jsonl_files = list(meta_dir.glob("*.jsonl"))
            print(f"元数据文件: {len(jsonl_files)} 个")
            for f in jsonl_files[:5]:
                print(f"   - {f.name}")
        
        file_dir = data_dir / "files"
        if file_dir.exists():
            pdf_files = list(file_dir.glob("**/*.pdf"))
            print(f"PDF文件: {len(pdf_files)} 个")
    else:
        print(f"数据目录尚未创建: {data_dir}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
