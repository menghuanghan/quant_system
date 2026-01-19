"""调试证监会网页结构"""

import requests
from bs4 import BeautifulSoup

# 测试证监会令页面
url = 'http://www.csrc.gov.cn/csrc/c101953/zfxxgk_zdgk.shtml'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

resp = requests.get(url, headers=headers, timeout=10)
resp.encoding = 'utf-8'  # 强制UTF-8编码

soup = BeautifulSoup(resp.text, 'html.parser')

print(f"页面标题: {soup.title.text if soup.title else '无'}")
print("\n" + "="*60)

# 查找所有链接
links = soup.find_all('a', href=True)
print(f"找到 {len(links)} 个链接")

# 筛选政策相关链接
policy_links = []
for link in links:
    href = link.get('href', '')
    text = link.get_text(strip=True)
    
    # 跳过空链接和导航链接
    if not text or len(text) < 5:
        continue
    if text in ['首页', '返回', '更多', '详细']:
        continue
    
    # 包含关键词的链接
    if any(k in text for k in ['办法', '规定', '条例', '通知', '意见', '决定', '令']):
        policy_links.append({
            'title': text,
            'href': href
        })

print(f"\n找到 {len(policy_links)} 个政策链接:")
for i, item in enumerate(policy_links[:10], 1):
    print(f"{i}. {item['title']}")
    print(f"   {item['href']}")
