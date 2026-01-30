#!/usr/bin/env python3
"""
Scorer 模块测试脚本

测试规则打分器和 LLM 打分器的功能
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_import():
    """测试模块导入"""
    print("=" * 60)
    print("测试 1: 模块导入")
    print("=" * 60)
    
    try:
        from src.data_pipeline.processors.unstructured.scorer import (
            Scorer,
            BatchScoreResult,
            ScoreResult,
            ScorerConfig,
            ScoreLevel,
            ScoringMethod,
            RuleScorer,
            LLMScorer,
            score,
            score_batch,
        )
        print("✓ 所有模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False


def test_rule_scorer():
    """测试规则打分器"""
    print("\n" + "=" * 60)
    print("测试 2: 规则打分器 (RuleScorer)")
    print("=" * 60)
    
    from src.data_pipeline.processors.unstructured.scorer import RuleScorer, ScoreLevel
    
    scorer = RuleScorer()
    
    # 测试用例：覆盖不同分数级别
    test_cases = [
        # 极度利空
        ("关于公司被中国证监会立案调查的公告", ScoreLevel.EXTREMELY_BEARISH, "立案调查"),
        ("关于公司股票被实施退市风险警示的公告", ScoreLevel.EXTREMELY_BEARISH, "退市风险"),
        ("关于公司股票被实施*ST的公告", ScoreLevel.EXTREMELY_BEARISH, "*ST"),
        
        # 利空
        ("关于收到中国证监会行政监管措施决定书的公告", ScoreLevel.BEARISH, "监管措施"),
        ("关于公司控股股东减持股份的公告", ScoreLevel.BEARISH, "减持"),
        ("2024年度业绩预亏公告", ScoreLevel.BEARISH, "业绩预亏"),
        ("关于重大诉讼案件的公告", ScoreLevel.BEARISH, "诉讼"),
        
        # 轻度利空
        ("关于会计差错更正的公告", ScoreLevel.SLIGHTLY_BEARISH, "会计差错"),
        ("关于公司董事辞职的公告", ScoreLevel.SLIGHTLY_BEARISH, "辞职"),
        
        # 中性
        ("2024年第三季度报告", ScoreLevel.NEUTRAL, "定期报告"),
        ("关于召开2024年第一次临时股东大会的通知", ScoreLevel.NEUTRAL, "股东大会"),
        
        # 轻度利好
        ("关于获得政府补助的公告", ScoreLevel.SLIGHTLY_BULLISH, "政府补助"),
        ("关于公司获得新专利的公告", ScoreLevel.SLIGHTLY_BULLISH, "专利"),
        
        # 利好
        ("关于公司中标重大项目的公告", ScoreLevel.BULLISH, "中标"),
        ("2024年度业绩预增公告", ScoreLevel.BULLISH, "业绩预增"),
        ("关于控股股东增持公司股份的公告", ScoreLevel.BULLISH, "增持"),
        
        # 极度利好
        ("关于公司重大资产重组方案获得证监会核准的公告", ScoreLevel.EXTREMELY_BULLISH, "核准"),
    ]
    
    passed = 0
    failed = 0
    
    print("\n测试规则匹配:")
    for content, expected_level, rule_name in test_cases:
        result = scorer.score(content)
        
        # 检查是否成功匹配
        if result.success and result.matched_rule:
            level_match = result.level == expected_level
            # 允许相邻级别的误差
            adjacent_levels = [
                (ScoreLevel.EXTREMELY_BULLISH, ScoreLevel.BULLISH),
                (ScoreLevel.BULLISH, ScoreLevel.SLIGHTLY_BULLISH),
                (ScoreLevel.SLIGHTLY_BULLISH, ScoreLevel.NEUTRAL),
                (ScoreLevel.NEUTRAL, ScoreLevel.SLIGHTLY_BEARISH),
                (ScoreLevel.SLIGHTLY_BEARISH, ScoreLevel.BEARISH),
                (ScoreLevel.BEARISH, ScoreLevel.EXTREMELY_BEARISH),
            ]
            adjacent_match = any(
                (result.level == a and expected_level == b) or 
                (result.level == b and expected_level == a)
                for a, b in adjacent_levels
            )
            
            if level_match:
                status = "✓ 完全匹配"
                passed += 1
            elif adjacent_match:
                status = "○ 相邻匹配"
                passed += 1
            else:
                status = "✗ 级别不匹配"
                failed += 1
            
            print(f"  {status}")
            print(f"    内容: {content[:50]}...")
            print(f"    预期: {expected_level.name}, 实际: {result.level.name}")
            print(f"    分数: {result.score}, 规则: {result.matched_rule}")
        else:
            print(f"  ✗ 未匹配到规则: {content[:50]}...")
            failed += 1
    
    print(f"\n规则打分器测试结果: {passed}/{len(test_cases)} 通过")
    return failed == 0


def test_rule_scorer_batch():
    """测试规则打分器批量处理"""
    print("\n" + "=" * 60)
    print("测试 3: 规则打分器批量处理")
    print("=" * 60)
    
    from src.data_pipeline.processors.unstructured.scorer import RuleScorer
    
    scorer = RuleScorer()
    
    contents = [
        "关于公司被中国证监会立案调查的公告",
        "2024年度业绩预增公告",
        "关于公司中标重大项目的公告",
        "2024年第三季度报告",
        "关于控股股东减持股份的公告",
    ]
    
    print(f"\n批量处理 {len(contents)} 条内容...")
    start = time.time()
    results = scorer.score_batch(contents)
    elapsed = time.time() - start
    
    print(f"耗时: {elapsed:.3f}s ({elapsed/len(contents)*1000:.1f}ms/条)")
    
    for i, result in enumerate(results):
        print(f"\n  [{i+1}] {contents[i][:40]}...")
        print(f"      分数: {result.score:+4d} ({result.level.name})")
        if result.matched_rule:
            print(f"      规则: {result.matched_rule}")
    
    return True


def test_llm_scorer():
    """测试 LLM 打分器"""
    print("\n" + "=" * 60)
    print("测试 4: LLM 打分器 (LLMScorer)")
    print("=" * 60)
    
    from src.data_pipeline.processors.unstructured.scorer import LLMScorer, ScorerConfig
    
    # 检查 Ollama 是否可用
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code != 200:
            print("⚠ Ollama 服务不可用，跳过 LLM 打分器测试")
            return True
    except Exception as e:
        print(f"⚠ 无法连接 Ollama ({e})，跳过 LLM 打分器测试")
        return True
    
    print("✓ Ollama 服务可用")
    
    # ScorerConfig 使用正确的参数名
    config = ScorerConfig(
        model_name="qwen2.5:7b-instruct",
        ollama_host="http://localhost:11434",
        timeout=60.0,
    )
    
    scorer = LLMScorer(config)
    
    # 测试用例
    test_cases = [
        "公司发布2024年业绩快报，净利润同比增长150%，超出市场预期",
        "公司收到证监会行政处罚决定书，因财务造假被罚款5000万元",
        "公司发布2024年第三季度报告，业绩符合预期",
        "公司宣布拟以自有资金10亿元回购股份用于员工持股计划",
    ]
    
    print("\nLLM 打分测试:")
    for content in test_cases:
        print(f"\n  内容: {content[:50]}...")
        
        start = time.time()
        result = scorer.score(content)
        elapsed = time.time() - start
        
        if result.success:
            print(f"  ✓ 分数: {result.score:+4d} ({result.level.name})")
            print(f"    理由: {result.reason}")
            print(f"    耗时: {elapsed:.2f}s")
        else:
            print(f"  ✗ 打分失败: {result.reason}")
    
    return True


def test_main_scorer():
    """测试主打分器接口"""
    print("\n" + "=" * 60)
    print("测试 5: 主打分器接口 (Scorer)")
    print("=" * 60)
    
    from src.data_pipeline.processors.unstructured.scorer import (
        Scorer, ScoringMethod, ScorerConfig, score
    )
    
    # 测试规则模式 - 通过 config 指定默认方法
    print("\n--- 规则模式 ---")
    config = ScorerConfig(default_method=ScoringMethod.RULE)
    scorer = Scorer(config=config)
    
    result = scorer.score("关于公司被中国证监会立案调查的公告")
    print(f"内容: 关于公司被中国证监会立案调查的公告")
    print(f"分数: {result.score:+4d} ({result.level.name})")
    print(f"方法: {result.method}")
    
    # 测试便捷函数 - 通过 method 参数指定
    print("\n--- 便捷函数 ---")
    result = score("2024年度业绩预增公告", method=ScoringMethod.RULE)
    print(f"内容: 2024年度业绩预增公告")
    print(f"分数: {result.score:+4d} ({result.level.name})")
    
    # 测试自动模式（对于交易所公告）
    print("\n--- 自动模式 ---")
    config_auto = ScorerConfig(default_method=ScoringMethod.AUTO)
    scorer_auto = Scorer(config=config_auto)
    
    # 这应该使用规则打分（因为是交易所公告格式）
    result = scorer_auto.score(
        "关于公司中标重大项目的公告",
        data_type="news/exchange"
    )
    print(f"内容: 关于公司中标重大项目的公告")
    print(f"数据类型: news/exchange")
    print(f"分数: {result.score:+4d} ({result.level.name})")
    print(f"方法: {result.method}")
    
    return True


def test_dataframe_support():
    """测试 DataFrame 支持"""
    print("\n" + "=" * 60)
    print("测试 6: DataFrame 支持")
    print("=" * 60)
    
    try:
        import pandas as pd
    except ImportError:
        print("⚠ pandas 未安装，跳过 DataFrame 测试")
        return True
    
    from src.data_pipeline.processors.unstructured.scorer import (
        Scorer, ScoringMethod, ScorerConfig
    )
    
    # 创建测试 DataFrame
    df = pd.DataFrame({
        'title': [
            '关于公司被中国证监会立案调查的公告',
            '2024年度业绩预增公告',
            '关于公司中标重大项目的公告',
            '2024年第三季度报告',
            '关于控股股东减持股份的公告',
        ],
        'data_type': [
            'news/exchange',
            'news/exchange',
            'news/exchange',
            'news/exchange',
            'news/exchange',
        ],
    })
    
    print(f"\n输入 DataFrame: {len(df)} 行")
    print(df[['title']].head())
    
    config = ScorerConfig(default_method=ScoringMethod.RULE)
    scorer = Scorer(config=config)
    
    start = time.time()
    result_df = scorer.score_dataframe(
        df,
        content_column='title',
        score_column='score',
        level_column='score_level',
        reason_column='score_reason',
    )
    elapsed = time.time() - start
    
    print(f"\n输出 DataFrame: (耗时 {elapsed:.3f}s)")
    print(result_df[['title', 'score', 'score_level']].to_string())
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Scorer 模块测试")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_import),
        ("规则打分器", test_rule_scorer),
        ("规则打分器批量", test_rule_scorer_batch),
        ("LLM 打分器", test_llm_scorer),
        ("主打分器接口", test_main_scorer),
        ("DataFrame 支持", test_dataframe_support),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ 测试 '{name}' 发生异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
