"""
预期与预测分析域数据采集脚本

采集范围：
1. 盈利预测：业绩预告、券商盈利预测、一致预期
2. 机构评级：机构评级、评级汇总、机构调研
3. 研究员指数：分析师排行、分析师详情、券商金股

时间范围：2024-01-01 至 2024-12-31
样本股票：30只主流股票
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import logging
from datetime import datetime
import pandas as pd

from src.data_pipeline.collectors.structured.expectations import (
    get_earnings_forecast,
    get_broker_forecast,
    get_consensus_forecast,
    get_inst_rating,
    get_rating_summary,
    get_inst_survey,
    get_analyst_rank,
    get_analyst_detail,
    get_broker_gold_stock,
    get_forecast_revision,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据存储路径
DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'structured' / 'expectations'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 时间范围
START_DATE = '20240101'
END_DATE = '20241231'

# 样本股票（30只主流股票）
SAMPLE_STOCKS = [
    '600519.SH',  # 贵州茅台
    '000858.SZ',  # 五粮液
    '600036.SH',  # 招商银行
    '601318.SH',  # 中国平安
    '000001.SZ',  # 平安银行
    '600000.SH',  # 浦发银行
    '601166.SH',  # 兴业银行
    '000002.SZ',  # 万科A
    '600030.SH',  # 中信证券
    '601688.SH',  # 华泰证券
    '600887.SH',  # 伊利股份
    '000333.SZ',  # 美的集团
    '000651.SZ',  # 格力电器
    '002594.SZ',  # 比亚迪
    '300750.SZ',  # 宁德时代
    '688981.SH',  # 中芯国际
    '600276.SH',  # 恒瑞医药
    '000661.SZ',  # 长春高新
    '300015.SZ',  # 爱尔眼科
    '002475.SZ',  # 立讯精密
    '601012.SH',  # 隆基绿能
    '600809.SH',  # 山西汾酒
    '000568.SZ',  # 泸州老窖
    '603259.SH',  # 药明康德
    '300059.SZ',  # 东方财富
    '002415.SZ',  # 海康威视
    '600031.SH',  # 三一重工
    '601888.SH',  # 中国中免
    '002352.SZ',  # 顺丰控股
    '600900.SH',  # 长江电力
]


def check_empty_columns(df: pd.DataFrame, data_type: str) -> dict:
    """检查空列占比"""
    if df.empty:
        return {'empty': True, 'total_rows': 0}
    
    total_rows = len(df)
    empty_info = {}
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_ratio = null_count / total_rows
        if null_ratio > 0.2:  # 超过20%为空
            empty_info[col] = {
                'null_count': null_count,
                'null_ratio': f'{null_ratio*100:.1f}%'
            }
    
    return {
        'empty': False,
        'total_rows': total_rows,
        'empty_columns': empty_info
    }


def save_data(df: pd.DataFrame, filename: str, data_type: str):
    """保存数据并检查空列"""
    if df.empty:
        logger.warning(f"❌ {data_type}: 数据为空")
        return
    
    filepath = DATA_DIR / filename
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    # 检查空列
    check_result = check_empty_columns(df, data_type)
    
    if check_result['empty_columns']:
        logger.warning(f"⚠️  {data_type}: 保存 {check_result['total_rows']} 条数据，但存在空列问题：")
        for col, info in check_result['empty_columns'].items():
            logger.warning(f"   - {col}: {info['null_count']} 条空值 ({info['null_ratio']})")
    else:
        logger.info(f"✅ {data_type}: 成功保存 {check_result['total_rows']} 条数据 -> {filename}")


def collect_earnings_forecast():
    """采集盈利预测数据"""
    logger.info("\n" + "="*80)
    logger.info("开始采集盈利预测数据")
    logger.info("="*80)
    
    # 1. 业绩预告（按时间范围 - 修正：Tushare要求必须有ann_date或ts_code）
    logger.info("\n【1/3】采集业绩预告...")
    # 尝试采集2024年全年的，由于接口限制，我们分月采集或按特定公告日期
    df_forecast = get_earnings_forecast(
        ann_date='20240131' # 样例：1月底的公告
    )
    save_data(df_forecast, 'earnings_forecast.csv', '业绩预告')
    
    # 2. 券商盈利预测（样本股票）
    logger.info("\n【2/3】采集券商盈利预测 (由于频率限制，增加延迟)...")
    import time
    all_broker_forecasts = []
    for i, ts_code in enumerate(SAMPLE_STOCKS[:10], 1): # 减少到10只以提高成功率
        logger.info(f"  采集 {i}/10: {ts_code}")
        try:
            df = get_broker_forecast(
                ts_code=ts_code,
                start_date=START_DATE,
                end_date=END_DATE
            )
            if not df.empty:
                all_broker_forecasts.append(df)
            time.sleep(31) # Tushare限制每分钟2次，所以休眠约30秒
        except Exception as e:
            logger.error(f"  采集失败 {ts_code}: {e}")
    
    if all_broker_forecasts:
        df_broker = pd.concat(all_broker_forecasts, ignore_index=True)
        save_data(df_broker, 'broker_forecast.csv', '券商盈利预测')
    else:
        logger.warning("❌ 券商盈利预测: 无数据")
    
    # 3. 一致预期（样本股票）
    logger.info("\n【3/3】采集一致预期数据...")
    all_consensus = []
    for i, ts_code in enumerate(SAMPLE_STOCKS[:10], 1):  # 限制10只以节省时间
        logger.info(f"  采集 {i}/10: {ts_code}")
        try:
            df = get_consensus_forecast(ts_code=ts_code)
            if not df.empty:
                all_consensus.append(df)
        except Exception as e:
            logger.error(f"  采集失败 {ts_code}: {e}")
    
    if all_consensus:
        df_consensus = pd.concat(all_consensus, ignore_index=True)
        save_data(df_consensus, 'consensus_forecast.csv', '一致预期')
    else:
        logger.warning("❌ 一致预期: 无数据")


def collect_institutional_rating():
    """采集机构评级数据"""
    logger.info("\n" + "="*80)
    logger.info("开始采集机构评级数据")
    logger.info("="*80)
    
    # 1. 机构评级（按时间范围）
    logger.info("\n【1/3】采集机构评级...")
    df_rating = get_inst_rating(
        start_date=START_DATE,
        end_date=END_DATE
    )
    save_data(df_rating, 'institutional_rating.csv', '机构评级')
    
    # 2. 评级汇总统计
    logger.info("\n【2/3】采集评级汇总统计...")
    try:
        df_summary = get_rating_summary(
            start_date=START_DATE,
            end_date=END_DATE
        )
        save_data(df_summary, 'rating_summary.csv', '评级汇总')
    except Exception as e:
        logger.error(f"评级汇总采集失败: {e}")
    
    # 3. 机构调研（样本股票）
    logger.info("\n【3/3】采集机构调研...")
    all_surveys = []
    for i, ts_code in enumerate(SAMPLE_STOCKS[:15], 1):  # 限制15只
        logger.info(f"  采集 {i}/15: {ts_code}")
        try:
            df = get_inst_survey(
                ts_code=ts_code,
                start_date=START_DATE,
                end_date=END_DATE
            )
            if not df.empty:
                all_surveys.append(df)
        except Exception as e:
            logger.error(f"  采集失败 {ts_code}: {e}")
    
    if all_surveys:
        df_survey = pd.concat(all_surveys, ignore_index=True)
        save_data(df_survey, 'institutional_survey.csv', '机构调研')
    else:
        logger.warning("❌ 机构调研: 无数据")


def collect_analyst_index():
    """采集研究员指数数据"""
    logger.info("\n" + "="*80)
    logger.info("开始采集研究员指数数据")
    logger.info("="*80)
    
    # 1. 分析师排行
    logger.info("\n【1/3】采集分析师排行...")
    try:
        df_rank = get_analyst_rank()
        save_data(df_rank, 'analyst_rank.csv', '分析师排行')
    except Exception as e:
        logger.error(f"分析师排行采集失败: {e}")
    
    # 2. 券商金股
    logger.info("\n【2/3】采集券商金股...")
    try:
        df_gold = get_broker_gold_stock()
        save_data(df_gold, 'broker_gold_stock.csv', '券商金股')
    except Exception as e:
        logger.error(f"券商金股采集失败: {e}")
    
    # 3. 预测修正
    logger.info("\n【3/3】采集预测修正...")
    try:
        df_revision = get_forecast_revision()
        save_data(df_revision, 'forecast_revision.csv', '预测修正')
    except Exception as e:
        logger.error(f"预测修正采集失败: {e}")


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("预期与预测分析域数据采集")
    logger.info(f"时间范围: {START_DATE} - {END_DATE}")
    logger.info(f"样本股票: {len(SAMPLE_STOCKS)} 只")
    logger.info(f"数据存储: {DATA_DIR}")
    logger.info("="*80)
    
    start_time = datetime.now()
    
    # 采集各类数据
    collect_earnings_forecast()
    collect_institutional_rating()
    collect_analyst_index()
    
    # 汇总统计
    logger.info("\n" + "="*80)
    logger.info("采集完成汇总")
    logger.info("="*80)
    
    csv_files = list(DATA_DIR.glob('*.csv'))
    total_issues = 0
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            check_result = check_empty_columns(df, csv_file.stem)
            
            status = "✅"
            issue_count = len(check_result.get('empty_columns', {}))
            if issue_count > 0:
                status = "⚠️"
                total_issues += 1
            
            logger.info(f"{status} {csv_file.name}: {check_result['total_rows']} 条数据, {issue_count} 个空列问题")
        except Exception as e:
            logger.error(f"❌ {csv_file.name}: 读取失败 - {e}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*80)
    logger.info(f"总耗时: {duration:.1f} 秒")
    logger.info(f"生成文件: {len(csv_files)} 个")
    logger.info(f"存在空列问题的文件: {total_issues} 个")
    logger.info("="*80)


if __name__ == '__main__':
    main()
