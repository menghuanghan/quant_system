"""
上市公司公告采集功能快速测试

测试时间范围：2025年1月（仅15天）
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


def quick_test():
    """快速测试公告采集功能"""
    
    # 导入模块
    logger.info("正在导入采集模块...")
    from src.data_pipeline.collectors.unstructured.announcements.cninfo_crawler import (
        CninfoAnnouncementCrawler
    )
    logger.info("模块导入成功")
    
    # 输出目录
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'unstructured'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建爬虫
    crawler = CninfoAnnouncementCrawler()
    
    # 只采集2025年1月15-31日的深交所数据（测试用）
    logger.info("=" * 50)
    logger.info("采集深交所2025年1月15-31日公告")
    logger.info("=" * 50)
    
    df = crawler.collect(
        start_date='2025-01-15',
        end_date='2025-01-31',
        exchanges=['szse']
    )
    
    if not df.empty:
        logger.info(f"采集到 {len(df)} 条公告")
        
        # 按类别统计
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            logger.info("按类别统计:")
            for cat, count in category_counts.head(5).items():
                logger.info(f"  {cat}: {count} 条")
        
        # 保存数据
        output_file = os.path.join(output_dir, 'announcement_szse_20250115_31.csv')
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"数据已保存: {output_file}")
        
        # 显示样例
        logger.info("样例数据:")
        print(df[['ts_code', 'name', 'title', 'ann_date', 'category']].head(5).to_string())
    else:
        logger.warning("未采集到数据")
    
    logger.info("=" * 50)
    logger.info("快速测试完成")
    logger.info("=" * 50)


if __name__ == '__main__':
    quick_test()
