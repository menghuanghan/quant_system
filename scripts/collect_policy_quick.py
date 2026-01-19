"""
政策采集快速脚本

简化版本，聚焦于可靠的数据源

运行: python scripts/collect_policy_quick.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_csrc_news():
    """
    采集证监会要闻
    
    使用证监会要闻页面，这个页面内容直接渲染，不需要JS
    """
    import requests
    import re
    from bs4 import BeautifulSoup
    import pandas as pd
    import hashlib
    
    logger.info("采集证监会要闻...")
    
    url = "http://www.csrc.gov.cn/csrc/c100028/common_xq_list.shtml"
    
    try:
        response = requests.get(url, timeout=15)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        records = []
        
        # 查找所有新闻链接
        for a_tag in soup.select('a'):
            href = a_tag.get('href', '')
            if '/content.shtml' not in href:
                continue
            
            title = a_tag.get_text(strip=True)
            if len(title) < 10:
                continue
            
            # 排除无关内容
            if any(kw in title for kw in ['年度报表', '网站地图', '联系我们']):
                continue
            
            # 提取日期
            pub_date = ''
            parent = a_tag.parent
            if parent:
                text = parent.get_text()
                match = re.search(r'(\d{2}-\d{2})', text)
                if match:
                    pub_date = f"2026-{match.group(1)}"
            
            # 构建完整URL
            if href.startswith('/'):
                full_url = "http://www.csrc.gov.cn" + href
            else:
                full_url = href
            
            # 生成ID
            doc_id = hashlib.md5(title.encode()).hexdigest()
            
            records.append({
                'id': doc_id,
                'source_dept': '中国证监会',
                'title': title,
                'url': full_url,
                'publish_date': pub_date,
                'source': 'csrc',
                'category': 'news',
                'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.drop_duplicates(subset=['title'])
        
        logger.info(f"采集到 {len(df)} 条证监会要闻")
        return df
        
    except Exception as e:
        logger.error(f"采集证监会要闻失败: {e}")
        return pd.DataFrame()


def collect_gov_latest():
    """
    采集国务院最新政策
    """
    import requests
    import re
    from bs4 import BeautifulSoup
    import pandas as pd
    import hashlib
    
    logger.info("采集国务院最新政策...")
    
    # 尝试多个可能的URL
    urls = [
        "https://www.gov.cn/zhengce/xxgk/index.htm",
        "https://www.gov.cn/zhengce/",
        "https://www.gov.cn/xinwen/lianbo/index.htm",
    ]
    
    records = []
    
    for url in urls:
        try:
            response = requests.get(url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code != 200:
                continue
            
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找新闻/政策链接
            for a_tag in soup.select('a'):
                href = a_tag.get('href', '')
                title = a_tag.get_text(strip=True)
                
                if not title or len(title) < 10:
                    continue
                
                # 只要政策相关的链接
                if not any(kw in href for kw in ['/zhengce/', '/xinwen/', '/gongbao/']):
                    continue
                
                # 排除无关内容
                if any(kw in title for kw in ['更多', '加载', '下一页']):
                    continue
                
                # 构建完整URL
                if href.startswith('/'):
                    full_url = "https://www.gov.cn" + href
                elif href.startswith('http'):
                    full_url = href
                else:
                    continue
                
                doc_id = hashlib.md5(title.encode()).hexdigest()
                
                records.append({
                    'id': doc_id,
                    'source_dept': '国务院',
                    'title': title,
                    'url': full_url,
                    'publish_date': '',
                    'source': 'gov',
                    'category': 'latest',
                    'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            if records:
                break  # 找到数据就停止
            
        except Exception as e:
            logger.debug(f"URL {url} 请求失败: {e}")
            continue
    
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=['title'])
    
    logger.info(f"采集到 {len(df)} 条国务院政策")
    return df


def collect_eastmoney_finance_news():
    """
    采集东方财富财经要闻（包含政策相关）
    """
    import requests
    import re
    from bs4 import BeautifulSoup
    import pandas as pd
    import hashlib
    
    logger.info("采集东方财富财经要闻...")
    
    # 东方财富财经频道
    url = "https://finance.eastmoney.com/"
    
    try:
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        records = []
        
        # 查找新闻链接
        for a_tag in soup.select('a'):
            href = a_tag.get('href', '')
            title = a_tag.get_text(strip=True)
            
            if not title or len(title) < 10:
                continue
            
            # 只要新闻链接
            if 'eastmoney.com' not in href and not href.startswith('/'):
                continue
            
            # 过滤政策相关新闻
            policy_keywords = ['政策', '监管', '证监会', '央行', '财政', '发改委', '国务院', 
                              '法规', '规定', '办法', '通知', '意见', 'IPO', '上市']
            
            if not any(kw in title for kw in policy_keywords):
                continue
            
            doc_id = hashlib.md5(title.encode()).hexdigest()
            
            records.append({
                'id': doc_id,
                'source_dept': '东方财富',
                'title': title,
                'url': href if href.startswith('http') else f"https://finance.eastmoney.com{href}",
                'publish_date': '',
                'source': 'eastmoney',
                'category': 'news',
                'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.drop_duplicates(subset=['title'])
        
        logger.info(f"采集到 {len(df)} 条政策相关新闻")
        return df
        
    except Exception as e:
        logger.error(f"采集东方财富要闻失败: {e}")
        return pd.DataFrame()


def main():
    """主函数"""
    import pandas as pd
    
    print("\n" + "="*60)
    print("政策数据快速采集")
    print("="*60)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建输出目录
    output_dir = Path("data/raw/unstructured/policy/csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    # 1. 证监会要闻
    df_csrc = collect_csrc_news()
    if not df_csrc.empty:
        all_data.append(df_csrc)
    
    # 2. 国务院最新政策
    df_gov = collect_gov_latest()
    if not df_gov.empty:
        all_data.append(df_gov)
    
    # 3. 东方财富政策新闻
    df_em = collect_eastmoney_finance_news()
    if not df_em.empty:
        all_data.append(df_em)
    
    # 合并并保存
    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        df_all = df_all.drop_duplicates(subset=['title'])
        
        # 保存CSV
        date_str = datetime.now().strftime('%Y%m%d')
        output_file = output_dir / f"policy_news_{date_str}.csv"
        df_all.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*60)
        print("采集结果汇总")
        print("="*60)
        print(f"总计: {len(df_all)} 条政策相关新闻/文件")
        print(f"\n按来源统计:")
        for source, count in df_all['source'].value_counts().items():
            print(f"  - {source}: {count} 条")
        
        print(f"\n数据已保存至: {output_file}")
        
        # 显示样例
        print("\n" + "="*60)
        print("最新政策样例（前10条）")
        print("="*60)
        for _, row in df_all.head(10).iterrows():
            print(f"[{row['source_dept']}] {row['title'][:50]}...")
    else:
        print("\n未采集到任何数据")
    
    return len(all_data) > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
