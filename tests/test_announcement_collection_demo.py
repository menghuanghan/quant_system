"""
上市公司公告采集 - 2025年Q1完整测试

采集2025年1月、2月、3月全市场公告数据
"""

import os
import sys
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect_q1_2025():
    """采集2025年Q1全市场公告"""
    
    from src.data_pipeline.collectors.unstructured.announcements.cninfo_crawler import (
        CninfoAnnouncementCrawler
    )
    
    # 输出目录
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'unstructured'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    crawler = CninfoAnnouncementCrawler()
    
    # 按月采集，避免单次请求过大
    months = [
        ('2025-01-01', '2025-01-31', '202501'),
        ('2025-02-01', '2025-02-28', '202502'),
        ('2025-03-01', '2025-03-31', '202503'),
    ]
    
    total_records = 0
    
    for start_date, end_date, month_label in months:
        logger.info("=" * 50)
        logger.info(f"采集 {month_label} 公告...")
        logger.info("=" * 50)
        
        for exchange in ['szse', 'sse']:
            logger.info(f"  采集 {exchange.upper()}...")
            
            df = crawler.collect(
                start_date=start_date,
                end_date=end_date,
                exchanges=[exchange]
            )
            
            if not df.empty:
                logger.info(f"  {exchange.upper()}: {len(df)} 条公告")
                total_records += len(df)
                
                # 保存数据
                output_file = os.path.join(
                    output_dir, 
                    f'announcement_{exchange}_{month_label}.csv'
                )
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
                logger.info(f"  已保存: {output_file}")
            else:
                logger.warning(f"  {exchange.upper()}: 无数据")
    
    logger.info("=" * 50)
    logger.info(f"采集完成！总计 {total_records} 条公告")
    logger.info("=" * 50)
    
    # 列出生成的文件
    logger.info("生成的文件:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.csv'):
            filepath = os.path.join(output_dir, f)
            size = os.path.getsize(filepath)
            logger.info(f"  {f}: {size/1024:.1f} KB")


if __name__ == '__main__':
    collect_q1_2025()
