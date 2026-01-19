"""
为现有政策数据下载PDF附件

策略：
1. 读取已采集的政策数据
2. 对于有URL但没有附件的记录，尝试访问详情页查找PDF链接
3. 下载PDF附件到files目录
"""

import sys
import re
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

def download_file(url: str, save_path: Path) -> bool:
    """下载文件"""
    try:
        response = requests.get(url, timeout=30, stream=True, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if response.status_code == 200:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"  下载失败: {e}")
    return False

def find_pdf_links(url: str) -> list:
    """从页面中查找PDF链接"""
    try:
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if response.status_code != 200:
            return []
        
        response.encoding = response.apparent_encoding or 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        pdf_links = []
        for a in soup.select('a[href$=".pdf"], a[href$=".PDF"]'):
            href = a.get('href', '')
            if href:
                if href.startswith('/'):
                    # 根据URL判断基础域名
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    base_url = f"{parsed.scheme}://{parsed.netloc}"
                    href = base_url + href
                elif not href.startswith('http'):
                    href = urljoin(url, href)
                pdf_links.append(href)
        
        return pdf_links
    except Exception as e:
        print(f"  查找PDF失败: {e}")
        return []

def main():
    # 读取现有数据
    data_file = Path("data/raw/unstructured/policy/csv/policy_yearly_20260119.csv")
    if not data_file.exists():
        print(f"数据文件不存在: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    print("="*60)
    print("为现有政策数据下载PDF附件")
    print("="*60)
    print(f"总记录数: {len(df)}")
    
    # 确保列存在
    if 'local_path' not in df.columns:
        df['local_path'] = ''
    if 'file_type' not in df.columns:
        df['file_type'] = 'html'
    
    # 筛选需要下载附件的记录
    # 1. 有URL
    # 2. 没有附件路径（或附件路径为空）
    needs_download = df['url'].notna() & (
        df['local_path'].isna() | (df['local_path'] == '')
    )
    
    download_candidates = df[needs_download]
    print(f"需要处理的记录: {len(download_candidates)} 条")
    
    if len(download_candidates) == 0:
        print("所有记录已有附件或无URL")
        return
    
    # 限制数量（避免处理时间过长）
    max_process = 50
    download_candidates = download_candidates.head(max_process)
    print(f"本次处理: {len(download_candidates)} 条")
    print()
    
    success_count = 0
    files_dir = Path("data/raw/unstructured/policy/files")
    
    for idx, row in download_candidates.iterrows():
        print(f"[{idx+1}/{len(download_candidates)}] {row['title'][:40]}...")
        
        url = row['url']
        source = row['source']
        doc_no = row.get('doc_no', '')
        title = row['title']
        
        # 查找PDF链接
        pdf_links = find_pdf_links(url)
        
        if not pdf_links:
            print(f"  未找到PDF链接")
            continue
        
        print(f"  找到 {len(pdf_links)} 个PDF链接")
        
        # 下载第一个PDF
        pdf_url = pdf_links[0]
        
        # 生成文件名
        safe_title = re.sub(r'[\\/:*?"<>|]', '', title)[:50]
        filename = f"{doc_no}_{safe_title}.pdf" if doc_no else f"{safe_title}.pdf"
        
        # 按来源和年份组织目录
        pub_date = row.get('publish_date')
        year = '2025'
        if pd.notna(pub_date) and str(pub_date):
            year = str(pub_date)[:4]
        
        save_path = files_dir / year / source / filename
        
        # 下载
        if download_file(pdf_url, save_path):
            print(f"  ✓ 下载成功: {save_path}")
            # 更新DataFrame
            df.at[idx, 'local_path'] = str(save_path)
            df.at[idx, 'file_type'] = 'pdf'
            success_count += 1
        
        time.sleep(0.5)  # 控制速率
    
    print()
    print("="*60)
    print(f"下载完成: {success_count}/{len(download_candidates)} 个附件")
    print("="*60)
    
    # 保存更新后的数据
    if success_count > 0:
        df.to_csv(data_file, index=False, encoding='utf-8-sig')
        print(f"\n数据已更新: {data_file}")
        
        # 更新JSONL
        jsonl_file = data_file.parent.parent / 'meta' / 'policy_yearly_20260119.jsonl'
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            import json
            for record in df.to_dict('records'):
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"元数据已更新: {jsonl_file}")

if __name__ == "__main__":
    main()
