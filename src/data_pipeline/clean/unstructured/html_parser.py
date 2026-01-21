"""
HTML 网页清洗模块 (HTML Parser)

核心职责：从杂乱的网页源代码中剥离出"正文"，丢弃广告和导航。

主要功能：
1. DOM 树解析 - 使用 BeautifulSoup + lxml 解析
2. 噪音标签移除 - 删除 script, style, nav, footer 等
3. 结构化提取 - 保持段落结构，使用换行符分隔
4. 特定网站适配 - 针对雪球、东财等网站的优化

设计原则：即时提取（Extract-on-the-fly），不依赖磁盘存储
"""

import re
import logging
from typing import Optional, List, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HTMLCleanConfig:
    """
    HTML 清洗配置
    
    可针对不同网站调整清洗策略
    """
    # 需要移除的标签（黑名单）
    remove_tags: Set[str] = field(default_factory=lambda: {
        'script', 'style', 'head', 'meta', 'link', 'noscript',
        'iframe', 'frame', 'frameset', 'object', 'embed',
        'nav', 'footer', 'header', 'aside', 'menu',
        'form', 'input', 'button', 'select', 'textarea',
        'svg', 'canvas', 'audio', 'video', 'source',
        'template', 'slot'
    })
    
    # 需要移除的 class 关键词（广告、推荐等）
    remove_class_keywords: Set[str] = field(default_factory=lambda: {
        'ad', 'ads', 'advert', 'advertisement', 'banner',
        'sidebar', 'side-bar', 'recommend', 'related',
        'comment', 'comments', 'share', 'social',
        'footer', 'header', 'nav', 'navigation', 'menu',
        'popup', 'modal', 'overlay', 'cookie',
        'breadcrumb', 'pagination'
    })
    
    # 需要移除的 id 关键词
    remove_id_keywords: Set[str] = field(default_factory=lambda: {
        'ad', 'ads', 'sidebar', 'footer', 'header',
        'nav', 'comment', 'share', 'recommend'
    })
    
    # 正文容器的常见 class（优先提取）
    content_class_keywords: Set[str] = field(default_factory=lambda: {
        'article', 'content', 'post', 'entry', 'main',
        'body', 'text', 'story', 'news'
    })
    
    # 分隔符标签（这些标签后需要换行）
    block_tags: Set[str] = field(default_factory=lambda: {
        'p', 'div', 'br', 'hr', 'li', 'tr',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'article', 'section', 'blockquote', 'pre'
    })
    
    # 最小正文长度（字符数，低于此值认为提取失败）
    min_content_length: int = 50
    
    # 是否保留链接文本
    keep_link_text: bool = True
    
    # 是否保留表格内容
    keep_table_text: bool = True


class HTMLParser:
    """
    HTML 网页清洗器
    
    从网页源码中提取正文文本，去除广告、导航等噪音。
    """
    
    # 常见广告/噪音正则模式
    NOISE_PATTERNS = [
        r'var\s+\w+\s*=',           # JS 变量声明
        r'function\s*\(',            # JS 函数
        r'\{[^}]{10,}\}',           # 残留的 JSON/CSS
        r'<[^>]+>',                  # 残留的 HTML 标签
        r'<!--.*?-->',               # HTML 注释
        r'http[s]?://\S+',          # URL（可选移除）
    ]
    
    # 雪球特定选择器
    XUEQIU_SELECTORS = {
        'content': ['.article__bd', '.status-content', '.detail__content'],
        'remove': ['.article__share', '.stock-links', '.recommend']
    }
    
    # 东方财富特定选择器
    EASTMONEY_SELECTORS = {
        'content': ['.article-body', '.newsContent', '.news-content'],
        'remove': ['.related-news', '.stock-list', '.em-share']
    }
    
    # 股吧特定选择器
    GUBA_SELECTORS = {
        'content': ['.article-body', '.stockbar_content', '.post_content'],
        'remove': ['.related-post', '.ad-wrapper']
    }
    
    def __init__(self, config: Optional[HTMLCleanConfig] = None):
        """
        初始化 HTML 清洗器
        
        Args:
            config: 清洗配置，None 使用默认配置
        """
        self.config = config or HTMLCleanConfig()
        
        # 编译噪音正则
        self._noise_patterns = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.NOISE_PATTERNS
        ]
    
    def extract_text(
        self,
        html: str,
        site_type: Optional[str] = None,
        normalize: bool = True
    ) -> str:
        """
        从 HTML 提取正文文本（核心方法）
        
        Args:
            html: HTML 源码字符串
            site_type: 网站类型 ('xueqiu', 'eastmoney', 'guba', None)
            normalize: 是否进行文本标准化
            
        Returns:
            提取的纯文本
            
        Examples:
            >>> html = requests.get(news_url).text
            >>> text = parser.extract_text(html)
        """
        if not html or not html.strip():
            return ""
        
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("BeautifulSoup 未安装，请执行: pip install beautifulsoup4")
            return ""
        
        try:
            # 优先使用 lxml 解析器（最快）
            try:
                soup = BeautifulSoup(html, 'lxml')
            except Exception:
                # 降级到 html.parser
                soup = BeautifulSoup(html, 'html.parser')
            
            # 1. 针对特定网站优化提取
            if site_type:
                text = self._extract_by_site(soup, site_type)
                if text and len(text) >= self.config.min_content_length:
                    if normalize:
                        from .text_utils import normalize_for_storage
                        text = normalize_for_storage(text)
                    return text
            
            # 2. 通用提取流程
            text = self._generic_extract(soup)
            
            if normalize and text:
                from .text_utils import normalize_for_storage
                text = normalize_for_storage(text)
            
            return text
            
        except Exception as e:
            logger.warning(f"HTML 解析失败: {type(e).__name__}: {e}")
            return ""
    
    def _extract_by_site(self, soup, site_type: str) -> str:
        """
        根据网站类型使用特定选择器提取
        
        Args:
            soup: BeautifulSoup 对象
            site_type: 网站类型
            
        Returns:
            提取的文本
        """
        selectors_map = {
            'xueqiu': self.XUEQIU_SELECTORS,
            'eastmoney': self.EASTMONEY_SELECTORS,
            'guba': self.GUBA_SELECTORS
        }
        
        selectors = selectors_map.get(site_type)
        if not selectors:
            return ""
        
        # 先移除噪音元素
        for selector in selectors.get('remove', []):
            for elem in soup.select(selector):
                elem.decompose()
        
        # 尝试匹配正文容器
        for selector in selectors.get('content', []):
            content_elem = soup.select_one(selector)
            if content_elem:
                return self._extract_text_from_element(content_elem)
        
        return ""
    
    def _generic_extract(self, soup) -> str:
        """
        通用 HTML 文本提取
        
        Args:
            soup: BeautifulSoup 对象
            
        Returns:
            提取的文本
        """
        # 1. 移除黑名单标签
        for tag_name in self.config.remove_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # 2. 移除带有噪音 class/id 的元素
        self._remove_noise_elements(soup)
        
        # 3. 尝试找到正文容器
        content_elem = self._find_content_container(soup)
        
        if content_elem:
            return self._extract_text_from_element(content_elem)
        
        # 4. 降级：直接从 body 提取
        body = soup.find('body')
        if body:
            return self._extract_text_from_element(body)
        
        # 5. 最后降级：整个文档
        return self._extract_text_from_element(soup)
    
    def _remove_noise_elements(self, soup):
        """
        移除带有噪音关键词的元素
        
        Args:
            soup: BeautifulSoup 对象
        """
        # 收集需要删除的元素（避免在迭代中修改）
        to_remove = []
        
        for elem in soup.find_all(True):
            should_remove = False
            
            # 检查 class
            classes = elem.get('class', [])
            if isinstance(classes, str):
                classes = [classes]
            
            class_str = ' '.join(classes).lower() if classes else ''
            for keyword in self.config.remove_class_keywords:
                if keyword in class_str:
                    should_remove = True
                    break
            
            # 检查 id（如果还没有被标记删除）
            if not should_remove:
                elem_id = elem.get('id', '')
                if elem_id:
                    elem_id_lower = elem_id.lower()
                    for keyword in self.config.remove_id_keywords:
                        if keyword in elem_id_lower:
                            should_remove = True
                            break
            
            if should_remove:
                to_remove.append(elem)
        
        # 批量删除
        for elem in to_remove:
            try:
                elem.decompose()
            except Exception:
                pass  # 元素可能已经被父元素删除
    
    def _find_content_container(self, soup):
        """
        查找正文容器元素
        
        Args:
            soup: BeautifulSoup 对象
            
        Returns:
            正文容器元素或 None
        """
        def get_classes(tag):
            """安全获取 class 列表"""
            classes = tag.get('class', [])
            if classes is None:
                return []
            if isinstance(classes, str):
                return [classes]
            return classes
        
        # 方法1：通过 class 关键词查找
        for keyword in self.config.content_class_keywords:
            # class 包含关键词
            elems = soup.find_all(
                lambda tag: tag.name in ['div', 'article', 'section', 'main'] and
                any(keyword in c.lower() for c in get_classes(tag))
            )
            if elems:
                # 选择文本最长的那个
                return max(elems, key=lambda e: len(e.get_text()))
        
        # 方法2：查找 article 标签
        article = soup.find('article')
        if article:
            return article
        
        # 方法3：查找 main 标签
        main = soup.find('main')
        if main:
            return main
        
        # 方法4：查找文本最密集的 div
        divs = soup.find_all('div')
        if divs:
            # 计算文本密度（文本长度 / 标签数量）
            best_div = None
            best_density = 0
            
            for div in divs:
                text_len = len(div.get_text(strip=True))
                tag_count = len(div.find_all(True)) + 1
                density = text_len / tag_count if tag_count > 0 else 0
                
                # 排除太短的文本
                if text_len >= self.config.min_content_length and density > best_density:
                    best_density = density
                    best_div = div
            
            return best_div
        
        return None
    
    def _extract_text_from_element(self, elem) -> str:
        """
        从元素中提取文本，保持段落结构
        
        Args:
            elem: BeautifulSoup 元素
            
        Returns:
            提取的文本
        """
        # 使用 separator 在块级元素间插入换行
        # 注意：不能简单用 get_text()，否则标题和正文会粘在一起
        
        text_parts = []
        
        for element in elem.descendants:
            if element.name is None:
                # 文本节点
                text = element.strip() if isinstance(element, str) else str(element).strip()
                if text:
                    text_parts.append(text)
            elif element.name in self.config.block_tags:
                # 块级标签，添加换行
                text_parts.append('\n')
        
        text = ' '.join(text_parts)
        
        # 清理连续空白
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n +', '\n', text)
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def extract_article_info(self, html: str) -> dict:
        """
        提取文章结构化信息
        
        Args:
            html: HTML 源码
            
        Returns:
            包含标题、作者、时间等信息的字典
        """
        if not html:
            return {}
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'lxml')
        except Exception:
            return {}
        
        info = {}
        
        # 提取标题
        title_elem = soup.find('title') or soup.find('h1')
        if title_elem:
            info['title'] = title_elem.get_text(strip=True)
        
        # 提取 meta 信息
        meta_mapping = {
            'description': ['description', 'og:description'],
            'keywords': ['keywords'],
            'author': ['author', 'og:author'],
            'publish_time': ['publish_time', 'article:published_time', 'pubdate']
        }
        
        for key, meta_names in meta_mapping.items():
            for meta_name in meta_names:
                meta = soup.find('meta', attrs={'name': meta_name}) or \
                       soup.find('meta', attrs={'property': meta_name})
                if meta and meta.get('content'):
                    info[key] = meta['content']
                    break
        
        # 提取正文
        info['content'] = self.extract_text(html)
        info['content_length'] = len(info.get('content', ''))
        
        return info


# 全局单例
_default_parser: Optional[HTMLParser] = None


def get_html_parser() -> HTMLParser:
    """获取全局 HTML 解析器单例"""
    global _default_parser
    if _default_parser is None:
        _default_parser = HTMLParser()
    return _default_parser


# 便捷函数
def extract_text_from_html(
    html: str,
    site_type: Optional[str] = None,
    normalize: bool = True
) -> str:
    """
    从 HTML 提取正文文本（便捷函数）
    
    这是采集器调用的主要接口，不需要知道底层实现。
    
    Args:
        html: HTML 源码字符串
        site_type: 网站类型 ('xueqiu', 'eastmoney', 'guba', None)
        normalize: 是否进行文本标准化
        
    Returns:
        提取的纯文本
        
    Examples:
        >>> html = requests.get(news_url).text
        >>> text = extract_text_from_html(html)
        
        >>> # 针对雪球优化
        >>> text = extract_text_from_html(html, site_type='xueqiu')
    """
    return get_html_parser().extract_text(html, site_type, normalize)


def extract_article_info(html: str) -> dict:
    """
    提取文章结构化信息（便捷函数）
    
    Args:
        html: HTML 源码
        
    Returns:
        包含标题、作者、时间、正文等信息的字典
    """
    return get_html_parser().extract_article_info(html)


def clean_html_tags(html: str) -> str:
    """
    简单去除所有 HTML 标签
    
    适用于简单场景，不进行复杂的正文提取。
    
    Args:
        html: HTML 源码
        
    Returns:
        去除标签后的文本
    """
    if not html:
        return ""
    
    # 移除 script 和 style 内容
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.IGNORECASE | re.DOTALL)
    
    # 移除所有标签
    html = re.sub(r'<[^>]+>', ' ', html)
    
    # 清理空白
    html = re.sub(r'\s+', ' ', html)
    
    return html.strip()


if __name__ == '__main__':
    """测试 HTML 解析功能"""
    
    print("=" * 60)
    print("HTML 网页清洗模块测试")
    print("=" * 60)
    
    # 检查 BeautifulSoup 是否安装
    try:
        from bs4 import BeautifulSoup
        print(f"✓ BeautifulSoup 已安装")
    except ImportError:
        print("✗ BeautifulSoup 未安装，请执行: pip install beautifulsoup4")
        import sys
        sys.exit(1)
    
    # 测试解析器初始化
    parser = HTMLParser()
    print(f"✓ HTMLParser 初始化成功")
    
    # 测试用例
    test_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>测试新闻标题</title>
        <meta name="description" content="这是新闻描述">
        <meta name="author" content="张三">
        <script>var x = 1;</script>
        <style>.ad { display: block; }</style>
    </head>
    <body>
        <header>网站头部导航</header>
        <nav>导航菜单</nav>
        <main>
            <article class="article-content">
                <h1>测试新闻标题</h1>
                <p>这是新闻的第一段内容。</p>
                <p>这是新闻的第二段内容，包含一些数字123和中文。</p>
                <div class="ad-banner">广告内容</div>
                <p>这是新闻的第三段内容。</p>
            </article>
        </main>
        <aside class="sidebar">侧边栏推荐</aside>
        <footer>网站底部</footer>
    </body>
    </html>
    """
    
    print("\n测试 HTML 内容:")
    print("-" * 40)
    print(test_html[:200] + "...")
    
    print("\n1. 通用提取测试:")
    text = extract_text_from_html(test_html)
    print(f"  提取结果: {repr(text[:200])}")
    print(f"  字符数: {len(text)}")
    
    print("\n2. 文章信息提取:")
    info = extract_article_info(test_html)
    for key, value in info.items():
        if key != 'content':
            print(f"  {key}: {value}")
    
    print("\n3. 简单标签清理:")
    simple_html = "<p>段落1</p><p>段落2</p><script>alert(1)</script>"
    result = clean_html_tags(simple_html)
    print(f"  输入: {simple_html}")
    print(f"  输出: {result}")
    
    print("\n4. 空数据处理:")
    result = extract_text_from_html("")
    assert result == "", "空数据应返回空字符串"
    print("  ✓ 空数据处理正常")
    
    print("\n" + "=" * 60)
    print("测试完成!")
