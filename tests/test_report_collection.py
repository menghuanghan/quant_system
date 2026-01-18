"""
研报与分析师观点采集 - 2025年Q1完整测试

采集内容：
- 券商研报（投资评级、目标价、PDF链接）
- 盈利预测（EPS）
- 分析师排名
"""

import os
import sys
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect_reports_q1_2025():
    """采集2025年Q1研报数据"""
    
    from src.data_pipeline.collectors.unstructured.reports import (
        EastMoneyReportCollector,
        AnalystCollector,
        get_eps_forecast,
    )
    import pandas as pd
    
    # 输出目录
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'unstructured', 'reports'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)
    logger.info("研报与分析师观点采集 - 2025年Q1")
    logger.info("=" * 60)
    
    # 热门股票列表（覆盖多行业）
    stock_list = [
        # 银行
        '000001', '600036', '601166', '601398', '601288',
        # 白酒
        '600519', '000858', '000568',
        # 科技
        '002415', '300059', '002230',
        # 新能源
        '300750', '002594', '601012',
        # 医药
        '600276', '000538', '002007',
        # 地产
        '000002', '600048', '001979',
        # 消费
        '000651', '600887', '002304',
        # 券商
        '600030', '601688', '601211',
    ]
    
    results = {}
    
    # ========== 1. 采集券商研报 ==========
    logger.info("")
    logger.info("=" * 60)
    logger.info("1. 采集券商研报")
    logger.info("   -> 投资评级、目标价、评级变化、PDF链接")
    logger.info("=" * 60)
    
    try:
        report_collector = EastMoneyReportCollector()
        
        # 分批采集
        batch_size = 8
        all_reports = []
        
        for i in range(0, len(stock_list), batch_size):
            batch = stock_list[i:i+batch_size]
            logger.info(f"  批次 {i//batch_size + 1}: {len(batch)} 只股票")
            
            df = report_collector.collect(
                start_date='2025-01-01',
                end_date='2025-03-31',
                stock_codes=batch
            )
            
            if not df.empty:
                all_reports.append(df)
        
        if all_reports:
            df_reports = pd.concat(all_reports, ignore_index=True)
            df_reports = df_reports.drop_duplicates(subset=['report_id'], keep='first')
            
            results['reports'] = len(df_reports)
            logger.info(f"  研报总计: {len(df_reports)} 条")
            
            # 统计分析
            logger.info("")
            logger.info("  [评级分布]")
            for rating, count in df_reports['rating'].value_counts().items():
                logger.info(f"    {rating}: {count}")
            
            logger.info("")
            logger.info("  [评级变化]")
            for change, count in df_reports['rating_change'].value_counts().items():
                logger.info(f"    {change}: {count}")
            
            logger.info("")
            logger.info("  [研报机构TOP5]")
            for broker, count in df_reports['broker'].value_counts().head(5).items():
                logger.info(f"    {broker}: {count}")
            
            # 保存
            output_file = os.path.join(output_dir, 'reports_2025Q1.csv')
            df_reports.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"  已保存: {output_file}")
        else:
            results['reports'] = 0
                
    except Exception as e:
        logger.error(f"研报采集失败: {e}")
        results['reports'] = 0
        import traceback
        traceback.print_exc()
    
    # ========== 2. 采集盈利预测（EPS） ==========
    logger.info("")
    logger.info("=" * 60)
    logger.info("2. 采集盈利预测（EPS）")
    logger.info("   -> 预测机构数、EPS最小值/均值/最大值、行业平均")
    logger.info("=" * 60)
    
    try:
        df_eps = get_eps_forecast(stock_list[:20])  # 采集前20只股票
        
        if not df_eps.empty:
            results['eps_forecast'] = len(df_eps)
            logger.info(f"  盈利预测: {len(df_eps)} 条")
            
            # 统计分析
            if 'year' in df_eps.columns:
                logger.info("")
                logger.info("  [按年份分布]")
                for year, count in df_eps['year'].value_counts().sort_index().items():
                    logger.info(f"    {year}: {count}")
            
            # 保存
            output_file = os.path.join(output_dir, 'eps_forecast_2025.csv')
            df_eps.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"  已保存: {output_file}")
        else:
            results['eps_forecast'] = 0
            logger.warning("  无盈利预测数据")
            
    except Exception as e:
        logger.error(f"盈利预测采集失败: {e}")
        results['eps_forecast'] = 0
        import traceback
        traceback.print_exc()
    
    # ========== 3. 采集分析师排名 ==========
    logger.info("")
    logger.info("=" * 60)
    logger.info("3. 采集分析师排名")
    logger.info("   -> 分析师业绩、收益率、研报数量")
    logger.info("=" * 60)
    
    try:
        analyst_collector = AnalystCollector()
        
        df_analysts = analyst_collector.collect_analyst_rank()
        
        if not df_analysts.empty:
            results['analysts'] = len(df_analysts)
            logger.info(f"  分析师: {len(df_analysts)} 条")
            
            # 统计分析
            if 'broker' in df_analysts.columns:
                logger.info("")
                logger.info("  [机构分布TOP5]")
                for broker, count in df_analysts['broker'].value_counts().head(5).items():
                    logger.info(f"    {broker}: {count}")
            
            output_file = os.path.join(output_dir, 'analyst_rank_2025Q1.csv')
            df_analysts.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"  已保存: {output_file}")
        else:
            results['analysts'] = 0
            logger.warning("  无分析师数据")
            
    except Exception as e:
        logger.error(f"分析师采集失败: {e}")
        results['analysts'] = 0
        import traceback
        traceback.print_exc()
    
    # ========== 汇总 ==========
    logger.info("")
    logger.info("=" * 60)
    logger.info("采集结果汇总")
    logger.info("=" * 60)
    
    for key, count in results.items():
        status = "✓" if count > 0 else "✗" 
        logger.info(f"  {status} {key}: {count} 条")
    
    logger.info("")
    logger.info("生成的文件:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.csv'):
            filepath = os.path.join(output_dir, f)
            size = os.path.getsize(filepath)
            logger.info(f"  {f}: {size/1024:.1f} KB")


if __name__ == '__main__':
    collect_reports_q1_2025()
