"""
预期与预测分析域 和 深度风险与质量因子域 数据采集脚本

采集2024年全年数据，样本股票30只
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from data_pipeline.collectors.structured.expectations import (
    EarningsForecastCollector,
    BrokerForecastCollector,
    ConsensusForecastCollector,
    InstitutionalRatingCollector,
    RatingSummaryCollector,
    InstitutionalSurveyCollector,
    AnalystRankCollector,
    AnalystDetailCollector,
    BrokerGoldStockCollector,
    ForecastRevisionCollector,
)

from data_pipeline.collectors.structured.deep_risk_quality import (
    APEPBEWMedianCollector,
    MarketCongestionCollector,
    StockBondSpreadCollector,
    BuffettIndicatorCollector,
    StockGoodwillCollector,
    GoodwillImpairmentCollector,
    BreakNetStockCollector,
    MSCIESGCollector,
    HZESGCollector,
    RefinitivESGCollector,
    ZhidingESGCollector,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据保存路径
RAW_DATA_DIR = project_root / 'data' / 'raw' / 'structured'
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 时间范围：2024年全年
START_DATE = '20240101'
END_DATE = '20241231'

# 样本股票：30只主流股票（覆盖不同行业和市值）
SAMPLE_STOCKS = [
    '600519.SH',  # 贵州茅台
    '000858.SZ',  # 五粮液
    '600036.SH',  # 招商银行
    '601318.SH',  # 中国平安
    '600276.SH',  # 恒瑞医药
    '000333.SZ',  # 美的集团
    '002594.SZ',  # 比亚迪
    '600030.SH',  # 中信证券
    '601012.SH',  # 隆基绿能
    '300750.SZ',  # 宁德时代
    '600887.SH',  # 伊利股份
    '000001.SZ',  # 平安银行
    '601888.SH',  # 中国中免
    '600809.SH',  # 山西汾酒
    '002475.SZ',  # 立讯精密
    '600031.SH',  # 三一重工
    '000568.SZ',  # 泸州老窖
    '603288.SH',  # 海天味业
    '600585.SH',  # 海螺水泥
    '601166.SH',  # 兴业银行
    '000002.SZ',  # 万科A
    '600690.SH',  # 海尔智家
    '002415.SZ',  # 海康威视
    '601398.SH',  # 工商银行
    '601939.SH',  # 建设银行
    '600900.SH',  # 长江电力
    '600028.SH',  # 中国石化
    '601857.SH',  # 中国石油
    '000725.SZ',  # 京东方A
    '601601.SH',  # 中国太保
]


def save_data(df, filename, domain_dir):
    """
    保存数据到文件
    
    Args:
        df: DataFrame数据
        filename: 文件名
        domain_dir: 数据域目录名
    """
    if df is None or df.empty:
        logger.warning(f"数据为空，跳过保存: {filename}")
        return False
    
    # 创建数据域目录
    save_dir = RAW_DATA_DIR / domain_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为CSV
    filepath = save_dir / filename
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    # 检查空列占比
    empty_cols = []
    total_cols = len(df.columns)
    for col in df.columns:
        null_ratio = df[col].isnull().sum() / len(df)
        if null_ratio > 0.9:  # 超过90%为空
            empty_cols.append(col)
    
    empty_ratio = len(empty_cols) / total_cols if total_cols > 0 else 0
    
    logger.info(f"已保存数据: {filepath}")
    logger.info(f"  行数: {len(df)}, 列数: {total_cols}")
    logger.info(f"  空列数: {len(empty_cols)} ({empty_ratio:.1%})")
    
    if empty_ratio > 0.2:
        logger.warning(f"  警告: 空列占比超过20%，空列: {empty_cols}")
        return False
    
    return True


def collect_expectations_forecasts():
    """采集预期与预测分析域数据"""
    logger.info("=" * 80)
    logger.info("开始采集【预期与预测分析域】数据")
    logger.info("=" * 80)
    
    results = {}
    domain_dir = 'expectations_forecasts'
    
    # 1. 业绩预告
    logger.info("\n【1/10】采集业绩预告数据...")
    try:
        collector = EarningsForecastCollector()
        df = collector.collect(start_date=START_DATE, end_date=END_DATE)
        results['earnings_forecast'] = save_data(df, 'earnings_forecast.csv', domain_dir)
    except Exception as e:
        logger.error(f"业绩预告采集失败: {e}", exc_info=True)
        results['earnings_forecast'] = False
    
    # 2. 券商盈利预测（按股票采集）
    logger.info("\n【2/10】采集券商盈利预测数据...")
    try:
        collector = BrokerForecastCollector()
        all_data = []
        for i, stock in enumerate(SAMPLE_STOCKS, 1):
            logger.info(f"  采集 {stock} ({i}/{len(SAMPLE_STOCKS)})...")
            df = collector.collect(ts_code=stock, start_date=START_DATE, end_date=END_DATE)
            if df is not None and not df.empty:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            results['broker_forecast'] = save_data(combined_df, 'broker_forecast.csv', domain_dir)
        else:
            logger.warning("券商盈利预测：无有效数据")
            results['broker_forecast'] = False
    except Exception as e:
        logger.error(f"券商盈利预测采集失败: {e}", exc_info=True)
        results['broker_forecast'] = False
    
    # 3. 一致预期数据（按股票采集）
    logger.info("\n【3/10】采集一致预期数据...")
    try:
        collector = ConsensusForecastCollector()
        all_data = []
        for i, stock in enumerate(SAMPLE_STOCKS, 1):
            logger.info(f"  采集 {stock} ({i}/{len(SAMPLE_STOCKS)})...")
            df = collector.collect(ts_code=stock, year='2024')
            if df is not None and not df.empty:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            results['consensus_forecast'] = save_data(combined_df, 'consensus_forecast.csv', domain_dir)
        else:
            logger.warning("一致预期：无有效数据")
            results['consensus_forecast'] = False
    except Exception as e:
        logger.error(f"一致预期采集失败: {e}", exc_info=True)
        results['consensus_forecast'] = False
    
    # 4. 机构评级（按股票采集）
    logger.info("\n【4/10】采集机构评级数据...")
    try:
        collector = InstitutionalRatingCollector()
        all_data = []
        for i, stock in enumerate(SAMPLE_STOCKS, 1):
            logger.info(f"  采集 {stock} ({i}/{len(SAMPLE_STOCKS)})...")
            df = collector.collect(ts_code=stock, start_date=START_DATE, end_date=END_DATE)
            if df is not None and not df.empty:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            results['institutional_rating'] = save_data(combined_df, 'institutional_rating.csv', domain_dir)
        else:
            logger.warning("机构评级：无有效数据")
            results['institutional_rating'] = False
    except Exception as e:
        logger.error(f"机构评级采集失败: {e}", exc_info=True)
        results['institutional_rating'] = False
    
    # 5. 评级汇总统计
    logger.info("\n【5/10】采集评级汇总统计数据...")
    try:
        collector = RatingSummaryCollector()
        df = collector.collect(start_date=START_DATE, end_date=END_DATE)
        results['rating_summary'] = save_data(df, 'rating_summary.csv', domain_dir)
    except Exception as e:
        logger.error(f"评级汇总统计采集失败: {e}", exc_info=True)
        results['rating_summary'] = False
    
    # 6. 机构调研
    logger.info("\n【6/10】采集机构调研数据...")
    try:
        collector = InstitutionalSurveyCollector()
        df = collector.collect(start_date=START_DATE, end_date=END_DATE)
        results['institutional_survey'] = save_data(df, 'institutional_survey.csv', domain_dir)
    except Exception as e:
        logger.error(f"机构调研采集失败: {e}", exc_info=True)
        results['institutional_survey'] = False
    
    # 7. 分析师排行
    logger.info("\n【7/10】采集分析师排行数据...")
    try:
        collector = AnalystRankCollector()
        df = collector.collect(year='2024')
        results['analyst_rank'] = save_data(df, 'analyst_rank.csv', domain_dir)
    except Exception as e:
        logger.error(f"分析师排行采集失败: {e}", exc_info=True)
        results['analyst_rank'] = False
    
    # 8. 分析师详情（按股票采集）
    logger.info("\n【8/10】采集分析师详情数据...")
    try:
        collector = AnalystDetailCollector()
        all_data = []
        for i, stock in enumerate(SAMPLE_STOCKS, 1):
            logger.info(f"  采集 {stock} ({i}/{len(SAMPLE_STOCKS)})...")
            df = collector.collect(ts_code=stock)
            if df is not None and not df.empty:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            results['analyst_detail'] = save_data(combined_df, 'analyst_detail.csv', domain_dir)
        else:
            logger.warning("分析师详情：无有效数据")
            results['analyst_detail'] = False
    except Exception as e:
        logger.error(f"分析师详情采集失败: {e}", exc_info=True)
        results['analyst_detail'] = False
    
    # 9. 券商金股组合
    logger.info("\n【9/10】采集券商金股组合数据...")
    try:
        collector = BrokerGoldStockCollector()
        df = collector.collect(year='2024')
        results['broker_gold_stock'] = save_data(df, 'broker_gold_stock.csv', domain_dir)
    except Exception as e:
        logger.error(f"券商金股组合采集失败: {e}", exc_info=True)
        results['broker_gold_stock'] = False
    
    # 10. 预测修正数据（按股票采集）
    logger.info("\n【10/10】采集预测修正数据...")
    try:
        collector = ForecastRevisionCollector()
        all_data = []
        for i, stock in enumerate(SAMPLE_STOCKS, 1):
            logger.info(f"  采集 {stock} ({i}/{len(SAMPLE_STOCKS)})...")
            df = collector.collect(ts_code=stock, start_date=START_DATE, end_date=END_DATE)
            if df is not None and not df.empty:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            results['forecast_revision'] = save_data(combined_df, 'forecast_revision.csv', domain_dir)
        else:
            logger.warning("预测修正：无有效数据")
            results['forecast_revision'] = False
    except Exception as e:
        logger.error(f"预测修正采集失败: {e}", exc_info=True)
        results['forecast_revision'] = False
    
    return results


def collect_deep_risk_quality():
    """采集深度风险与质量因子域数据"""
    logger.info("\n" + "=" * 80)
    logger.info("开始采集【深度风险与质量因子域】数据")
    logger.info("=" * 80)
    
    results = {}
    domain_dir = 'deep_risk_quality'
    
    # 1. A股等权重与中位数市盈率/市净率
    logger.info("\n【1/11】采集A股等权重与中位数PE/PB数据...")
    try:
        collector = APEPBEWMedianCollector()
        df = collector.collect(start_date=START_DATE, end_date=END_DATE)
        results['a_pe_pb_ew_median'] = save_data(df, 'a_pe_pb_ew_median.csv', domain_dir)
    except Exception as e:
        logger.error(f"A股PE/PB采集失败: {e}", exc_info=True)
        results['a_pe_pb_ew_median'] = False
    
    # 2. 大盘拥挤度
    logger.info("\n【2/11】采集大盘拥挤度数据...")
    try:
        collector = MarketCongestionCollector()
        df = collector.collect(start_date=START_DATE, end_date=END_DATE)
        results['market_congestion'] = save_data(df, 'market_congestion.csv', domain_dir)
    except Exception as e:
        logger.error(f"大盘拥挤度采集失败: {e}", exc_info=True)
        results['market_congestion'] = False
    
    # 3. 股债利差
    logger.info("\n【3/11】采集股债利差数据...")
    try:
        collector = StockBondSpreadCollector()
        df = collector.collect(start_date=START_DATE, end_date=END_DATE)
        results['stock_bond_spread'] = save_data(df, 'stock_bond_spread.csv', domain_dir)
    except Exception as e:
        logger.error(f"股债利差采集失败: {e}", exc_info=True)
        results['stock_bond_spread'] = False
    
    # 4. 巴菲特指标
    logger.info("\n【4/11】采集巴菲特指标数据...")
    try:
        collector = BuffettIndicatorCollector()
        df = collector.collect(start_date=START_DATE, end_date=END_DATE)
        results['buffett_indicator'] = save_data(df, 'buffett_indicator.csv', domain_dir)
    except Exception as e:
        logger.error(f"巴菲特指标采集失败: {e}", exc_info=True)
        results['buffett_indicator'] = False
    
    # 5. 个股商誉明细（按股票采集）
    logger.info("\n【5/11】采集个股商誉明细数据...")
    try:
        collector = StockGoodwillCollector()
        all_data = []
        for i, stock in enumerate(SAMPLE_STOCKS, 1):
            logger.info(f"  采集 {stock} ({i}/{len(SAMPLE_STOCKS)})...")
            df = collector.collect(ts_code=stock, period='2024')
            if df is not None and not df.empty:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            results['stock_goodwill'] = save_data(combined_df, 'stock_goodwill.csv', domain_dir)
        else:
            logger.warning("个股商誉：无有效数据")
            results['stock_goodwill'] = False
    except Exception as e:
        logger.error(f"个股商誉采集失败: {e}", exc_info=True)
        results['stock_goodwill'] = False
    
    # 6. 商誉减值预期明细
    logger.info("\n【6/11】采集商誉减值预期数据...")
    try:
        collector = GoodwillImpairmentCollector()
        df = collector.collect(year='2024')
        results['goodwill_impairment'] = save_data(df, 'goodwill_impairment.csv', domain_dir)
    except Exception as e:
        logger.error(f"商誉减值采集失败: {e}", exc_info=True)
        results['goodwill_impairment'] = False
    
    # 7. 破净股统计
    logger.info("\n【7/11】采集破净股统计数据...")
    try:
        collector = BreakNetStockCollector()
        df = collector.collect(date=END_DATE)
        results['break_net_stock'] = save_data(df, 'break_net_stock.csv', domain_dir)
    except Exception as e:
        logger.error(f"破净股统计采集失败: {e}", exc_info=True)
        results['break_net_stock'] = False
    
    # 8. MSCI-ESG评级（按股票采集）
    logger.info("\n【8/11】采集MSCI-ESG评级数据...")
    try:
        collector = MSCIESGCollector()
        all_data = []
        for i, stock in enumerate(SAMPLE_STOCKS, 1):
            logger.info(f"  采集 {stock} ({i}/{len(SAMPLE_STOCKS)})...")
            df = collector.collect(symbol=stock)
            if df is not None and not df.empty:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            results['esg_msci'] = save_data(combined_df, 'esg_msci.csv', domain_dir)
        else:
            logger.warning("MSCI-ESG：无有效数据")
            results['esg_msci'] = False
    except Exception as e:
        logger.error(f"MSCI-ESG采集失败: {e}", exc_info=True)
        results['esg_msci'] = False
    
    # 9. 华证指数-ESG评级（按股票采集）
    logger.info("\n【9/11】采集华证指数-ESG评级数据...")
    try:
        collector = HZESGCollector()
        all_data = []
        for i, stock in enumerate(SAMPLE_STOCKS, 1):
            logger.info(f"  采集 {stock} ({i}/{len(SAMPLE_STOCKS)})...")
            df = collector.collect(symbol=stock)
            if df is not None and not df.empty:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            results['esg_hz'] = save_data(combined_df, 'esg_hz.csv', domain_dir)
        else:
            logger.warning("华证ESG：无有效数据")
            results['esg_hz'] = False
    except Exception as e:
        logger.error(f"华证ESG采集失败: {e}", exc_info=True)
        results['esg_hz'] = False
    
    # 10. 路孚特-ESG评级（按股票采集）
    logger.info("\n【10/11】采集路孚特-ESG评级数据...")
    try:
        collector = RefinitivESGCollector()
        all_data = []
        for i, stock in enumerate(SAMPLE_STOCKS, 1):
            logger.info(f"  采集 {stock} ({i}/{len(SAMPLE_STOCKS)})...")
            df = collector.collect(symbol=stock)
            if df is not None and not df.empty:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            results['esg_refinitiv'] = save_data(combined_df, 'esg_refinitiv.csv', domain_dir)
        else:
            logger.warning("路孚特ESG：无有效数据")
            results['esg_refinitiv'] = False
    except Exception as e:
        logger.error(f"路孚特ESG采集失败: {e}", exc_info=True)
        results['esg_refinitiv'] = False
    
    # 11. 秩鼎-ESG评级（按股票采集）
    logger.info("\n【11/11】采集秩鼎-ESG评级数据...")
    try:
        collector = ZhidingESGCollector()
        all_data = []
        for i, stock in enumerate(SAMPLE_STOCKS, 1):
            logger.info(f"  采集 {stock} ({i}/{len(SAMPLE_STOCKS)})...")
            df = collector.collect(symbol=stock)
            if df is not None and not df.empty:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            results['esg_zhiding'] = save_data(combined_df, 'esg_zhiding.csv', domain_dir)
        else:
            logger.warning("秩鼎ESG：无有效数据")
            results['esg_zhiding'] = False
    except Exception as e:
        logger.error(f"秩鼎ESG采集失败: {e}", exc_info=True)
        results['esg_zhiding'] = False
    
    return results


def print_summary(expectations_results, deep_risk_results):
    """打印采集汇总"""
    logger.info("\n" + "=" * 80)
    logger.info("数据采集汇总")
    logger.info("=" * 80)
    
    logger.info("\n【预期与预测分析域】采集结果:")
    success_count = sum(1 for v in expectations_results.values() if v)
    total_count = len(expectations_results)
    for name, success in expectations_results.items():
        status = "✓ 成功" if success else "✗ 失败/空列过多"
        logger.info(f"  {name:30s}: {status}")
    logger.info(f"  成功率: {success_count}/{total_count} ({success_count/total_count:.1%})")
    
    logger.info("\n【深度风险与质量因子域】采集结果:")
    success_count = sum(1 for v in deep_risk_results.values() if v)
    total_count = len(deep_risk_results)
    for name, success in deep_risk_results.items():
        status = "✓ 成功" if success else "✗ 失败/空列过多"
        logger.info(f"  {name:30s}: {status}")
    logger.info(f"  成功率: {success_count}/{total_count} ({success_count/total_count:.1%})")
    
    # 统计需要修复的采集器
    failed_collectors = []
    for name, success in {**expectations_results, **deep_risk_results}.items():
        if not success:
            failed_collectors.append(name)
    
    if failed_collectors:
        logger.warning("\n需要修复的采集器:")
        for name in failed_collectors:
            logger.warning(f"  - {name}")
    
    logger.info("\n数据已保存至: " + str(RAW_DATA_DIR))


def main():
    """主函数"""
    logger.info(f"开始数据采集任务")
    logger.info(f"时间范围: {START_DATE} - {END_DATE}")
    logger.info(f"样本股票数量: {len(SAMPLE_STOCKS)}")
    logger.info(f"数据保存路径: {RAW_DATA_DIR}")
    
    # 采集预期与预测分析域数据
    expectations_results = collect_expectations_forecasts()
    
    # 采集深度风险与质量因子域数据
    deep_risk_results = collect_deep_risk_quality()
    
    # 打印汇总
    print_summary(expectations_results, deep_risk_results)
    
    logger.info("\n数据采集任务完成!")


if __name__ == '__main__':
    main()
