"""
近一年政策与监管文本数据采集

数据源：
1. 国务院（政府信息公开）
2. 证监会（令、公告、要闻）
3. 发改委（政策文件）
4. 行业主管部门（工信部、财政部等）

时间范围：2025-01-19 至 2026-01-19

运行方式: python scripts/collect_policy_yearly.py
"""

import os
import sys
import time
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.collectors.unstructured.request_utils import safe_request

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/policy_collection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class YearlyPolicyCollector:
    """年度政策采集器"""
    
    def __init__(self, start_date: str, end_date: str):
        """
        Args:
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
        """
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path("data/raw/unstructured/policy")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_records = []
    
    def collect_csrc(self, max_pages: int = 20):
        """采集证监会政策"""
        logger.info(f"开始采集证监会政策 (最多{max_pages}页)...")
        
        # 证监会要闻页面
        base_url = "http://www.csrc.gov.cn/csrc/c100028/common_xq_list.shtml"
        
        for page in range(max_pages):
            try:
                if page == 0:
                    url = base_url
                else:
                    url = base_url.replace('.shtml', f'_{page}.shtml')
                
                logger.info(f"  采集第 {page+1} 页: {url}")
                
                response = safe_request(url, timeout=15)
                if not response or response.status_code != 200:
                    logger.warning(f"  页面请求失败")
                    break
                
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                
                page_records = 0
                
                # 查找所有链接
                for a_tag in soup.select('a'):
                    href = a_tag.get('href', '')
                    if '/content.shtml' not in href:
                        continue
                    
                    title = a_tag.get_text(strip=True)
                    if len(title) < 10:
                        continue
                    
                    # 排除无关内容
                    if any(kw in title for kw in ['年度报表', '网站地图', '联系我们', '法律声明']):
                        continue
                    
                    # 提取日期
                    pub_date = ''
                    parent = a_tag.parent
                    if parent:
                        import re
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
                    
                    record = {
                        'id': doc_id,
                        'source_dept': '中国证监会',
                        'doc_no': self._extract_doc_no(title),
                        'title': title,
                        'url': full_url,
                        'publish_date': pub_date,
                        'source': 'csrc',
                        'category': 'news',
                        'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    self.all_records.append(record)
                    page_records += 1
                
                logger.info(f"  第 {page+1} 页采集到 {page_records} 条")
                
                if page_records == 0:
                    break
                
                time.sleep(0.5)  # 控制请求频率
                
            except Exception as e:
                logger.error(f"  第 {page+1} 页采集失败: {e}")
                break
        
        logger.info(f"证监会政策采集完成")
    
    def collect_gov(self, max_pages: int = 10):
        """采集国务院政策"""
        logger.info(f"开始采集国务院政策 (最多{max_pages}页)...")
        
        # 尝试多个URL
        urls = [
            "https://www.gov.cn/zhengce/zhengceku/index.htm",
            "https://www.gov.cn/zhengce/",
            "https://www.gov.cn/xinwen/lianbo/index.htm",
        ]
        
        for base_url in urls:
            try:
                logger.info(f"  尝试URL: {base_url}")
                
                for page in range(max_pages):
                    if page == 0:
                        url = base_url
                    else:
                        url = base_url.replace('.htm', f'_{page}.htm')
                    
                    response = safe_request(url, timeout=15)
                    if not response or response.status_code != 200:
                        break
                    
                    response.encoding = 'utf-8'
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    page_records = 0
                    
                    # 查找政策链接
                    for a_tag in soup.select('a'):
                        href = a_tag.get('href', '')
                        title = a_tag.get_text(strip=True)
                        
                        if not title or len(title) < 10:
                            continue
                        
                        # 只要政策相关链接
                        if not any(kw in href for kw in ['/zhengce/', '/xinwen/', '/gongbao/']):
                            continue
                        
                        # 排除导航链接
                        if any(kw in title for kw in ['更多', '加载', '下一页', '上一页']):
                            continue
                        
                        # 构建完整URL
                        if href.startswith('/'):
                            full_url = "https://www.gov.cn" + href
                        elif href.startswith('http'):
                            full_url = href
                        else:
                            continue
                        
                        doc_id = hashlib.md5(title.encode()).hexdigest()
                        
                        record = {
                            'id': doc_id,
                            'source_dept': '国务院',
                            'doc_no': self._extract_doc_no(title),
                            'title': title,
                            'url': full_url,
                            'publish_date': '',
                            'source': 'gov',
                            'category': 'policy',
                            'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        self.all_records.append(record)
                        page_records += 1
                    
                    logger.info(f"  第 {page+1} 页采集到 {page_records} 条")
                    
                    if page_records == 0:
                        break
                    
                    time.sleep(0.5)
                
                if page_records > 0:
                    break  # 成功采集，不再尝试其他URL
                
            except Exception as e:
                logger.error(f"  URL {base_url} 采集失败: {e}")
                continue
        
        logger.info(f"国务院政策采集完成")
    
    def collect_ndrc(self, max_pages: int = 10):
        """采集发改委政策"""
        logger.info(f"开始采集发改委政策 (最多{max_pages}页)...")
        
        # 发改委政策发布
        base_urls = [
            "https://www.ndrc.gov.cn/xxgk/zcfb/",
            "https://www.ndrc.gov.cn/xxgk/zcfb/tz/",  # 通知
            "https://www.ndrc.gov.cn/xxgk/zcfb/ghxwj/",  # 规划文件
        ]
        
        for base_url in base_urls:
            try:
                logger.info(f"  采集分类: {base_url}")
                
                for page in range(min(5, max_pages)):  # 每个分类最多5页
                    if page == 0:
                        url = base_url + "index.html"
                    else:
                        url = base_url + f"index_{page}.html"
                    
                    response = safe_request(url, timeout=15)
                    if not response or response.status_code != 200:
                        break
                    
                    response.encoding = 'utf-8'
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    page_records = 0
                    
                    for a_tag in soup.select('a'):
                        href = a_tag.get('href', '')
                        title = a_tag.get_text(strip=True)
                        
                        if not title or len(title) < 10:
                            continue
                        
                        # 发改委政策链接特征
                        if '.html' not in href:
                            continue
                        
                        # 构建完整URL
                        if href.startswith('/'):
                            full_url = "https://www.ndrc.gov.cn" + href
                        elif href.startswith('http'):
                            full_url = href
                        elif href.startswith('./'):
                            full_url = base_url + href[2:]
                        else:
                            continue
                        
                        doc_id = hashlib.md5(title.encode()).hexdigest()
                        
                        record = {
                            'id': doc_id,
                            'source_dept': '国家发改委',
                            'doc_no': self._extract_doc_no(title),
                            'title': title,
                            'url': full_url,
                            'publish_date': '',
                            'source': 'ndrc',
                            'category': 'policy',
                            'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        self.all_records.append(record)
                        page_records += 1
                    
                    logger.info(f"    第 {page+1} 页采集到 {page_records} 条")
                    
                    if page_records == 0:
                        break
                    
                    time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"  发改委分类 {base_url} 采集失败: {e}")
                continue
        
        logger.info(f"发改委政策采集完成")
    
    def collect_miit(self, max_pages: int = 5):
        """采集工信部政策"""
        logger.info(f"开始采集工信部政策 (最多{max_pages}页)...")
        
        base_url = "https://www.miit.gov.cn/zwgk/zcwj/"
        
        try:
            for page in range(max_pages):
                if page == 0:
                    url = base_url + "index.html"
                else:
                    url = base_url + f"index_{page}.html"
                
                response = safe_request(url, timeout=15)
                if not response or response.status_code != 200:
                    break
                
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                
                page_records = 0
                
                for a_tag in soup.select('a'):
                    href = a_tag.get('href', '')
                    title = a_tag.get_text(strip=True)
                    
                    if not title or len(title) < 10:
                        continue
                    
                    if '.html' not in href:
                        continue
                    
                    # 构建完整URL
                    if href.startswith('/'):
                        full_url = "https://www.miit.gov.cn" + href
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        continue
                    
                    doc_id = hashlib.md5(title.encode()).hexdigest()
                    
                    record = {
                        'id': doc_id,
                        'source_dept': '工业和信息化部',
                        'doc_no': self._extract_doc_no(title),
                        'title': title,
                        'url': full_url,
                        'publish_date': '',
                        'source': 'miit',
                        'category': 'policy',
                        'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    self.all_records.append(record)
                    page_records += 1
                
                logger.info(f"  第 {page+1} 页采集到 {page_records} 条")
                
                if page_records == 0:
                    break
                
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"工信部政策采集失败: {e}")
        
        logger.info(f"工信部政策采集完成")
    
    def collect_mof(self, max_pages: int = 5):
        """采集财政部政策"""
        logger.info(f"开始采集财政部政策 (最多{max_pages}页)...")
        
        base_url = "http://www.mof.gov.cn/zhengwuxinxi/zhengcefabu/"
        
        try:
            for page in range(max_pages):
                if page == 0:
                    url = base_url + "index.htm"
                else:
                    url = base_url + f"index_{page}.htm"
                
                response = safe_request(url, timeout=15)
                if not response or response.status_code != 200:
                    break
                
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.text, 'html.parser')
                
                page_records = 0
                
                for a_tag in soup.select('a'):
                    href = a_tag.get('href', '')
                    title = a_tag.get_text(strip=True)
                    
                    if not title or len(title) < 10:
                        continue
                    
                    # 财政部政策链接特征
                    if 'mof.gov.cn' not in href and not href.startswith('/'):
                        continue
                    
                    # 构建完整URL
                    if href.startswith('/'):
                        full_url = "http://www.mof.gov.cn" + href
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        continue
                    
                    doc_id = hashlib.md5(title.encode()).hexdigest()
                    
                    record = {
                        'id': doc_id,
                        'source_dept': '财政部',
                        'doc_no': self._extract_doc_no(title),
                        'title': title,
                        'url': full_url,
                        'publish_date': '',
                        'source': 'mof',
                        'category': 'policy',
                        'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    self.all_records.append(record)
                    page_records += 1
                
                logger.info(f"  第 {page+1} 页采集到 {page_records} 条")
                
                if page_records == 0:
                    break
                
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"财政部政策采集失败: {e}")
        
        logger.info(f"财政部政策采集完成")
    
    def _extract_doc_no(self, text: str) -> str:
        """提取发文字号"""
        import re
        patterns = [
            r'(证监[发办函]\s*[〔\[（(]\s*\d{4}\s*[〕\]）)]\s*\d+\s*号)',
            r'(国[发办函]\s*[〔\[（(]\s*\d{4}\s*[〕\]）)]\s*\d+\s*号)',
            r'(\w{2,6}[发办函]\s*[〔\[（(]\s*\d{4}\s*[〕\]）)]\s*\d+\s*号)',
            r'(\d{4}\s*年\s*第?\s*\d+\s*号\s*公告)',
            r'([\u4e00-\u9fa5]+令\s*第?\s*\d+\s*号)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                doc_no = match.group(1)
                doc_no = re.sub(r'\s+', '', doc_no)
                return doc_no
        
        return ''
    
    def save_results(self):
        """保存采集结果"""
        if not self.all_records:
            logger.warning("没有采集到任何数据")
            return
        
        # 转换为DataFrame
        df = pd.DataFrame(self.all_records)
        
        # 去重
        df = df.drop_duplicates(subset=['title'], keep='first')
        
        # 按来源排序
        df = df.sort_values(['source', 'create_time'])
        
        # 保存CSV
        date_str = datetime.now().strftime('%Y%m%d')
        csv_file = self.output_dir / 'csv' / f'policy_yearly_{date_str}.csv'
        csv_file.parent.mkdir(exist_ok=True)
        
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"数据已保存至: {csv_file}")
        
        # 保存JSONL（元数据）
        jsonl_file = self.output_dir / 'meta' / f'policy_yearly_{date_str}.jsonl'
        jsonl_file.parent.mkdir(exist_ok=True)
        
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for record in df.to_dict('records'):
                import json
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"元数据已保存至: {jsonl_file}")
        
        # 统计报告
        self._print_summary(df)
    
    def _print_summary(self, df: pd.DataFrame):
        """打印采集统计"""
        print("\n" + "="*60)
        print("采集结果统计")
        print("="*60)
        print(f"时间范围: {self.start_date} ~ {self.end_date}")
        print(f"总计: {len(df)} 条政策/文件")
        
        print("\n按来源统计:")
        for source, count in df['source'].value_counts().items():
            source_dept = df[df['source'] == source]['source_dept'].iloc[0]
            print(f"  - {source_dept} ({source}): {count} 条")
        
        print("\n发文字号提取统计:")
        has_doc_no = df['doc_no'].notna() & (df['doc_no'] != '')
        print(f"  - 已提取发文字号: {has_doc_no.sum()} 条 ({has_doc_no.sum()/len(df)*100:.1f}%)")
        print(f"  - 无发文字号: {(~has_doc_no).sum()} 条")
        
        print("\n发文字号样例:")
        doc_no_samples = df[has_doc_no]['doc_no'].head(10).tolist()
        for doc_no in doc_no_samples:
            print(f"  - {doc_no}")
        
        print("\n最新政策样例（前10条）:")
        for _, row in df.head(10).iterrows():
            print(f"  [{row['source_dept']}] {row['title'][:50]}...")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("近一年政策与监管文本数据采集")
    print("="*60)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 时间范围：近一年
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"时间范围: {start_date} ~ {end_date}")
    print("="*60)
    
    # 创建采集器
    collector = YearlyPolicyCollector(start_date, end_date)
    
    # 依次采集各数据源
    try:
        # 1. 证监会（重要性最高）
        collector.collect_csrc(max_pages=30)
        
        # 2. 国务院
        collector.collect_gov(max_pages=20)
        
        # 3. 发改委
        collector.collect_ndrc(max_pages=15)
        
        # 4. 工信部
        collector.collect_miit(max_pages=10)
        
        # 5. 财政部
        collector.collect_mof(max_pages=10)
        
    except KeyboardInterrupt:
        logger.warning("采集被用户中断")
    except Exception as e:
        logger.error(f"采集过程出错: {e}", exc_info=True)
    
    # 保存结果
    collector.save_results()
    
    print("\n采集完成！")
    return True


if __name__ == "__main__":
    # 创建日志目录
    Path("logs").mkdir(exist_ok=True)
    
    success = main()
    sys.exit(0 if success else 1)
