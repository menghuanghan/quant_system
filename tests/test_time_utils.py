"""
时间标准化模块测试脚本

测试 time_utils 的所有核心功能：
1. 时间标准化（保守/激进/超保守模式）
2. 精度检测
3. 文本提取
4. 未来函数检测
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_standardize_publish_time():
    """测试时间标准化核心功能"""
    print("\n" + "=" * 70)
    print("1. 时间标准化测试 - 防止未来函数")
    print("=" * 70)
    
    from src.data_pipeline.clean.unstructured import standardize_publish_time, TimeAccuracy
    
    test_cases = [
        # (输入, 模式, 期望的小时, 描述)
        ("2024-01-05", "conservative", 17, "仅日期 → 盘后17:00"),
        ("2024-01-05", "aggressive", 8, "仅日期 → 盘前08:00"),
        ("2024年01月05日", "conservative", 17, "中文日期 → 盘后17:00"),
        ("2024-01-05 14:30:45", "conservative", 14, "完整时间戳 → 保持原值"),
        ("2024/01/05 09:30", "conservative", 9, "斜杠格式 → 保持原值"),
    ]
    
    print("\n时间填充模式对比:")
    passed = 0
    for raw_time, mode, expected_hour, desc in test_cases:
        result, accuracy = standardize_publish_time(raw_time, mode, with_accuracy=True)
        
        # 提取结果中的小时
        actual_hour = int(result.split(' ')[1].split(':')[0]) if ' ' in result else None
        
        status = "✓" if actual_hour == expected_hour else "✗"
        if actual_hour == expected_hour:
            passed += 1
        
        print(f"  {status} [{desc}]")
        print(f"     输入: {raw_time}")
        print(f"     模式: {mode}")
        print(f"     输出: {result}")
        print(f"     精度: {accuracy}")
    
    print(f"\n通过: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_detect_time_accuracy():
    """测试时间精度检测"""
    print("\n" + "=" * 70)
    print("2. 时间精度检测")
    print("=" * 70)
    
    from src.data_pipeline.clean.unstructured import TimeNormalizer, TimeAccuracy
    
    test_cases = [
        ("2024", TimeAccuracy.YEAR, "年份"),
        ("2024-01", TimeAccuracy.MONTH, "月份"),
        ("2024-01-05", TimeAccuracy.DAY, "日期"),
        ("2024-01-05 14", TimeAccuracy.HOUR, "小时"),
        ("2024-01-05 14:30", TimeAccuracy.MINUTE, "分钟"),
        ("2024-01-05 14:30:45", TimeAccuracy.SECOND, "秒"),
        ("2024年01月05日", TimeAccuracy.DAY, "中文日期"),
        ("2024年01月05日 14:30", TimeAccuracy.MINUTE, "中文日期含时间"),
    ]
    
    print("\n精度等级检测:")
    passed = 0
    for raw_time, expected_accuracy, desc in test_cases:
        actual_accuracy = TimeNormalizer.detect_time_accuracy(raw_time)
        status = "✓" if actual_accuracy == expected_accuracy else "✗"
        if actual_accuracy == expected_accuracy:
            passed += 1
        
        print(f"  {status} [{desc}]")
        print(f"     输入: {raw_time}")
        print(f"     期望: {expected_accuracy}")
        print(f"     实际: {actual_accuracy}")
    
    print(f"\n通过: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_extract_time_from_text():
    """测试从文本中提取时间戳"""
    print("\n" + "=" * 70)
    print("3. 文本时间提取")
    print("=" * 70)
    
    from src.data_pipeline.clean.unstructured import extract_time_from_text
    
    test_cases = [
        # (文本, 期望包含的日期, 描述)
        ("新华社 北京 2024年1月15日 电：上证指数今日涨幅3.2%", "2024-01-15", "新华社新闻头"),
        ("2024-01-15 美国 Fed 主席 鲍威尔 发表讲话", "2024-01-15", "英文日期格式"),
        ("【公告】公司于2024年5月12日发布年度报告", "2024-05-12", "公告格式"),
        ("3月15日上午，公司召开董事会会议", None, "仅月日，无年份"),
    ]
    
    print("\n从文本提取发布时间:")
    passed = 0
    for text, expected_date, desc in test_cases:
        extracted = extract_time_from_text(text, max_chars=200)
        
        if expected_date is None:
            status = "✓" if extracted is None else "✗"
            if extracted is None:
                passed += 1
        else:
            status = "✓" if extracted and expected_date in extracted else "✗"
            if extracted and expected_date in extracted:
                passed += 1
        
        print(f"  {status} [{desc}]")
        print(f"     原文: {text[:60]}...")
        print(f"     期望: {expected_date}")
        print(f"     提取: {extracted}")
    
    print(f"\n通过: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_is_future_data():
    """测试未来函数检测"""
    print("\n" + "=" * 70)
    print("4. 未来函数检测（回测数据泄露检查）")
    print("=" * 70)
    
    from src.data_pipeline.clean.unstructured import is_future_data
    
    test_cases = [
        # (发布时间, 回测时刻, 是否泄露, 描述)
        ("2024-01-05 17:00:00", "2024-01-05 14:30:00", True, "发布时间晚于回测时刻 → 泄露"),
        ("2024-01-05 09:00:00", "2024-01-05 14:30:00", False, "发布时间早于回测时刻 → 安全"),
        ("2024-01-04 17:00:00", "2024-01-05 09:00:00", False, "发布日期较早 → 安全"),
        ("2024-01-05 17:00:00", "2024-01-05 17:00:00", False, "发布时间等于回测时刻 → 边界安全"),
        ("2024-01-06 08:00:00", "2024-01-05 14:30:00", True, "发布日期更晚 → 泄露"),
    ]
    
    print("\n未来函数风险评估:")
    passed = 0
    for pub_time, bt_time, expected_leak, desc in test_cases:
        is_leak = is_future_data(pub_time, bt_time)
        status = "✓" if is_leak == expected_leak else "✗"
        if is_leak == expected_leak:
            passed += 1
        
        leak_status = "⚠️ 泄露" if is_leak else "✓ 安全"
        print(f"  {status} [{desc}] → {leak_status}")
        print(f"     发布时间: {pub_time}")
        print(f"     回测时刻: {bt_time}")
    
    print(f"\n通过: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_conservative_vs_aggressive():
    """对比保守策略和激进策略的区别"""
    print("\n" + "=" * 70)
    print("5. 保守策略 vs 激进策略对比（防泄露关键）")
    print("=" * 70)
    
    from src.data_pipeline.clean.unstructured import standardize_publish_time, is_future_data
    
    # 场景：2024-01-05 仅发布了日期，没有时间
    pub_date = "2024-01-05"
    backtest_time_am = "2024-01-05 10:00:00"  # 盘中时刻
    backtest_time_pm = "2024-01-05 17:30:00"  # 盘后时刻
    
    print("\n场景：数据源仅提供发布日期 2024-01-05（无时分秒）")
    print("回测时间：T 日上午 10:00（盘中）和下午 17:30（盘后）")
    
    # 保守模式
    conservative_time = standardize_publish_time(pub_date, mode='conservative')
    print(f"\n【保守模式】填充为 {conservative_time}")
    
    leak_am = is_future_data(conservative_time, backtest_time_am)
    leak_pm = is_future_data(conservative_time, backtest_time_pm)
    
    print(f"  T 日盘中 {backtest_time_am}: {'⚠️ 泄露' if leak_am else '✓ 安全'}")
    print(f"  T 日盘后 {backtest_time_pm}: {'⚠️ 泄露' if leak_pm else '✓ 安全'}")
    print(f"  → 推论：T 日内无法使用该数据，最早 T+1 使用")
    
    # 激进模式
    aggressive_time = standardize_publish_time(pub_date, mode='aggressive')
    print(f"\n【激进模式】填充为 {aggressive_time}")
    
    leak_am_agg = is_future_data(aggressive_time, backtest_time_am)
    leak_pm_agg = is_future_data(aggressive_time, backtest_time_pm)
    
    print(f"  T 日盘中 {backtest_time_am}: {'⚠️ 泄露' if leak_am_agg else '✓ 安全'}")
    print(f"  T 日盘后 {backtest_time_pm}: {'⚠️ 泄露' if leak_pm_agg else '✓ 安全'}")
    print(f"  → 推论：T 日盘前可使用该数据【存在作弊风险】")
    
    print("\n结论：")
    print(f"  ✓ 保守模式：防止数据泄露（推荐生产使用）")
    print(f"  ✗ 激进模式：可能泄露（仅在确认源为盘前专递时使用）")
    
    return not leak_am and not leak_pm  # 保守模式必须安全


def test_datetime_formats():
    """测试各种时间格式的支持"""
    print("\n" + "=" * 70)
    print("6. 多格式日期支持")
    print("=" * 70)
    
    from src.data_pipeline.clean.unstructured import standardize_publish_time
    
    formats = [
        "2024-01-05",
        "2024/01/05",
        "2024年01月05日",
        "2024.01.05",
        "01/05/2024",
        "05-Jan-2024",
        "2024-01-05 14:30:45",
        "2024年01月05日 14:30",
    ]
    
    print("\n支持的日期格式:")
    passed = 0
    for fmt in formats:
        try:
            result = standardize_publish_time(fmt)
            if result:
                status = "✓"
                passed += 1
            else:
                status = "✗"
            
            print(f"  {status} {fmt:30} → {result}")
        except Exception as e:
            print(f"  ✗ {fmt:30} → 错误: {e}")
    
    print(f"\n通过: {passed}/{len(formats)}")
    return passed >= len(formats) * 0.8


def main():
    """运行所有测试"""
    print("=" * 70)
    print("时间标准化模块完整测试")
    print("=" * 70)
    
    results = []
    
    try:
        results.append(("时间标准化", test_standardize_publish_time()))
    except Exception as e:
        print(f"✗ 时间标准化测试失败: {e}")
        results.append(("时间标准化", False))
    
    try:
        results.append(("精度检测", test_detect_time_accuracy()))
    except Exception as e:
        print(f"✗ 精度检测测试失败: {e}")
        results.append(("精度检测", False))
    
    try:
        results.append(("文本提取", test_extract_time_from_text()))
    except Exception as e:
        print(f"✗ 文本提取测试失败: {e}")
        results.append(("文本提取", False))
    
    try:
        results.append(("未来函数检测", test_is_future_data()))
    except Exception as e:
        print(f"✗ 未来函数检测测试失败: {e}")
        results.append(("未来函数检测", False))
    
    try:
        results.append(("策略对比", test_conservative_vs_aggressive()))
    except Exception as e:
        print(f"✗ 策略对比测试失败: {e}")
        results.append(("策略对比", False))
    
    try:
        results.append(("多格式支持", test_datetime_formats()))
    except Exception as e:
        print(f"✗ 多格式支持测试失败: {e}")
        results.append(("多格式支持", False))
    
    # 总结
    print("\n" + "=" * 70)
    print("测试结果总结")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status}: {name}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过!")
        print("\n核心收获：")
        print("  1. 时间标准化确保了时间戳的一致性（YYYY-MM-DD HH:MM:SS）")
        print("  2. 保守填充策略（盘后17:00）防止数据泄露")
        print("  3. 精度检测支持差异化回测策略")
        print("  4. 支持多种时间格式自动转换")
        return 0
    else:
        print("\n⚠️ 部分测试失败，请检查实现")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
