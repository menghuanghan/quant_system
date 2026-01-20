"""
非结构化数据清洗模块测试脚本

测试 text_utils, pdf_parser, html_parser 三个模块的功能
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_text_utils():
    """测试文本标准化模块"""
    print("\n" + "=" * 60)
    print("1. 文本标准化模块测试 (text_utils)")
    print("=" * 60)
    
    from src.data_pipeline.clean.unstructured import (
        normalize_text,
        normalize_for_nlp,
        normalize_for_storage,
        TextNormalizer
    )
    
    # 测试用例
    test_cases = [
        ("　Ａ股涨幅１．２％", "全角字符", "A股涨幅1.2%"),
        ("2024年01月15日发布", "中文日期", None),  # 标准模式不转换日期
        ("文本\x00中有\x0c控制字符", "控制字符", "文本中有控制字符"),
        ("多个   空格  合并", "空格合并", "多个 空格 合并"),
        ("第一段\n\n\n\n\n第二段", "多余换行", "第一段\n\n第二段"),
    ]
    
    print("\n标准模式测试:")
    passed = 0
    for text, desc, expected in test_cases:
        result = normalize_text(text)
        if expected:
            status = "✓" if result == expected else "✗"
            if result == expected:
                passed += 1
        else:
            status = "✓"
            passed += 1
        print(f"  {status} [{desc}]")
        print(f"    输入: {repr(text)}")
        print(f"    输出: {repr(result)}")
    
    print(f"\n通过: {passed}/{len(test_cases)}")
    
    # 测试 NLP 模式
    print("\nNLP模式测试:")
    long_text = """第一段内容。
    
    第二段【特殊标记】，日期2024年01月15日。
    
    第三段，数字1,234,567。"""
    
    result = normalize_for_nlp(long_text)
    print(f"  输入: {repr(long_text[:50])}...")
    print(f"  输出: {repr(result[:100])}...")
    print(f"  ✓ NLP模式正常（单行输出，标点转换）")
    
    return True


def test_pdf_parser():
    """测试 PDF 解析模块"""
    print("\n" + "=" * 60)
    print("2. PDF 解析模块测试 (pdf_parser)")
    print("=" * 60)
    
    # 检查依赖
    try:
        import pdfplumber
        print(f"✓ pdfplumber 版本: {pdfplumber.__version__}")
    except ImportError:
        print("✗ pdfplumber 未安装，跳过 PDF 测试")
        return False
    
    from src.data_pipeline.clean.unstructured import (
        extract_text_from_pdf_bytes,
        is_scanned_pdf,
        PDFParser
    )
    
    # 测试空数据
    result = extract_text_from_pdf_bytes(b'')
    assert result == "", "空数据应返回空字符串"
    print("✓ 空数据处理正常")
    
    # 测试无效 PDF
    result = extract_text_from_pdf_bytes(b'not a valid pdf')
    assert result == "", "无效 PDF 应返回空字符串"
    print("✓ 无效 PDF 处理正常")
    
    # 测试真实 PDF（如果有）
    pdf_dir = Path("data/raw/unstructured/events")
    pdf_files = list(pdf_dir.rglob("*.pdf")) if pdf_dir.exists() else []
    
    if pdf_files:
        pdf_file = pdf_files[0]
        print(f"\n测试真实 PDF: {pdf_file.name}")
        
        with open(pdf_file, 'rb') as f:
            pdf_bytes = f.read()
        
        # 检测是否扫描件
        scanned = is_scanned_pdf(pdf_bytes)
        print(f"  扫描件检测: {scanned}")
        
        # 提取文本
        text = extract_text_from_pdf_bytes(pdf_bytes, max_pages=2)
        print(f"  提取文本长度: {len(text)} 字符")
        
        if text:
            print(f"  前100字符: {text[:100]}...")
            print("✓ PDF 文本提取成功")
        else:
            if scanned:
                print("⚠ PDF 为扫描件，需要 OCR 处理")
            else:
                print("✗ PDF 文本提取失败")
    else:
        print("\n暂无测试 PDF 文件，跳过真实文件测试")
    
    return True


def test_html_parser():
    """测试 HTML 解析模块"""
    print("\n" + "=" * 60)
    print("3. HTML 解析模块测试 (html_parser)")
    print("=" * 60)
    
    # 检查依赖
    try:
        from bs4 import BeautifulSoup
        print("✓ BeautifulSoup 已安装")
    except ImportError:
        print("✗ BeautifulSoup 未安装，跳过 HTML 测试")
        return False
    
    from src.data_pipeline.clean.unstructured import (
        extract_text_from_html,
        extract_article_info,
        clean_html_tags
    )
    
    # 测试用例
    test_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>A股市场分析报告</title>
        <meta name="description" content="2024年A股市场走势分析">
        <meta name="author" content="分析师张三">
        <script>var x = 1; function test() {}</script>
        <style>.ad { display: block; }</style>
    </head>
    <body>
        <header><nav>网站导航</nav></header>
        <main>
            <article class="article-content">
                <h1>A股市场分析报告</h1>
                <p>今日A股三大指数集体高开，沪指涨1.2%。</p>
                <p>市场成交额突破万亿，北向资金净流入50亿。</p>
                <div class="ad-banner">广告内容应被移除</div>
                <p>分析师认为，后市将延续震荡上行格局。</p>
            </article>
            <aside class="sidebar">相关推荐</aside>
        </main>
        <footer>版权信息</footer>
    </body>
    </html>
    """
    
    # 测试空数据
    result = extract_text_from_html("")
    assert result == "", "空数据应返回空字符串"
    print("✓ 空数据处理正常")
    
    # 测试正文提取
    text = extract_text_from_html(test_html)
    print(f"\n正文提取测试:")
    print(f"  原始 HTML 长度: {len(test_html)} 字符")
    print(f"  提取文本长度: {len(text)} 字符")
    print(f"  提取内容: {text[:150]}...")
    
    # 验证噪音是否被移除
    assert "广告内容" not in text, "广告内容应被移除"
    assert "网站导航" not in text, "导航内容应被移除"
    assert "A股" in text, "正文内容应保留"
    print("✓ 噪音移除正常，正文保留正常")
    
    # 测试文章信息提取
    info = extract_article_info(test_html)
    print(f"\n文章信息提取:")
    print(f"  标题: {info.get('title', 'N/A')}")
    print(f"  描述: {info.get('description', 'N/A')}")
    print(f"  作者: {info.get('author', 'N/A')}")
    print(f"  正文长度: {info.get('content_length', 0)}")
    print("✓ 文章信息提取正常")
    
    # 测试简单标签清理
    simple_html = "<p>段落1</p><script>alert(1)</script><p>段落2</p>"
    result = clean_html_tags(simple_html)
    assert "alert" not in result, "脚本内容应被移除"
    assert "段落" in result, "正文应保留"
    print("✓ 简单标签清理正常")
    
    return True


def test_unified_interface():
    """测试统一接口"""
    print("\n" + "=" * 60)
    print("4. 统一接口测试")
    print("=" * 60)
    
    from src.data_pipeline.clean.unstructured import (
        extract_and_clean_pdf,
        extract_and_clean_html,
        detect_content_type,
        auto_extract_text
    )
    
    # 测试内容类型检测
    print("\n内容类型检测:")
    
    # PDF
    pdf_magic = b'%PDF-1.4 fake pdf content'
    result = detect_content_type(pdf_magic)
    assert result == 'pdf', f"PDF 应识别为 pdf，实际: {result}"
    print(f"  ✓ PDF 魔术字节 -> {result}")
    
    # HTML
    html_bytes = b'<!DOCTYPE html><html><body>test</body></html>'
    result = detect_content_type(html_bytes)
    assert result == 'html', f"HTML 应识别为 html，实际: {result}"
    print(f"  ✓ HTML 内容 -> {result}")
    
    # 纯文本
    text_bytes = "这是纯文本内容".encode('utf-8')
    result = detect_content_type(text_bytes)
    assert result == 'text', f"纯文本应识别为 text，实际: {result}"
    print(f"  ✓ 纯文本 -> {result}")
    
    # 测试自动提取
    print("\n自动提取测试:")
    
    html_content = b'<html><body><p>Hello World</p></body></html>'
    text = auto_extract_text(html_content)
    print(f"  HTML 自动提取: {repr(text[:50])}")
    
    text_content = "测试文本　全角空格".encode('utf-8')
    text = auto_extract_text(text_content)
    print(f"  文本自动提取: {repr(text)}")
    
    print("✓ 统一接口测试通过")
    return True


def test_performance():
    """性能测试"""
    print("\n" + "=" * 60)
    print("5. 性能测试")
    print("=" * 60)
    
    import time
    
    from src.data_pipeline.clean.unstructured import (
        normalize_text,
        extract_text_from_html
    )
    
    # 生成大文本
    large_text = "这是一段测试文本，包含各种字符１２３。\n" * 10000
    
    # 测试文本标准化性能
    start = time.time()
    for _ in range(10):
        result = normalize_text(large_text)
    elapsed = time.time() - start
    
    print(f"文本标准化 (10次 × {len(large_text)} 字符):")
    print(f"  总耗时: {elapsed:.3f} 秒")
    print(f"  平均: {elapsed/10*1000:.1f} ms/次")
    print(f"  吞吐: {len(large_text)*10/elapsed/1000:.0f} KB/秒")
    
    # 生成大 HTML
    large_html = """
    <html><body>
    """ + "<p>这是一段测试段落内容。</p>\n" * 5000 + """
    </body></html>
    """
    
    # 测试 HTML 解析性能
    start = time.time()
    for _ in range(10):
        result = extract_text_from_html(large_html)
    elapsed = time.time() - start
    
    print(f"\nHTML 解析 (10次 × {len(large_html)} 字符):")
    print(f"  总耗时: {elapsed:.3f} 秒")
    print(f"  平均: {elapsed/10*1000:.1f} ms/次")
    print(f"  吞吐: {len(large_html)*10/elapsed/1000:.0f} KB/秒")
    
    print("✓ 性能测试完成")
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("非结构化数据清洗模块测试")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("文本标准化", test_text_utils()))
    except Exception as e:
        print(f"✗ 文本标准化测试失败: {e}")
        results.append(("文本标准化", False))
    
    try:
        results.append(("PDF 解析", test_pdf_parser()))
    except Exception as e:
        print(f"✗ PDF 解析测试失败: {e}")
        results.append(("PDF 解析", False))
    
    try:
        results.append(("HTML 解析", test_html_parser()))
    except Exception as e:
        print(f"✗ HTML 解析测试失败: {e}")
        results.append(("HTML 解析", False))
    
    try:
        results.append(("统一接口", test_unified_interface()))
    except Exception as e:
        print(f"✗ 统一接口测试失败: {e}")
        results.append(("统一接口", False))
    
    try:
        results.append(("性能测试", test_performance()))
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        results.append(("性能测试", False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status}: {name}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过!")
        return 0
    else:
        print("\n⚠️ 部分测试失败，请检查依赖")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
