"""
深度风险与质量因子域数据采集脚本

运行此脚本以采集深度风险与质量因子相关数据
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.collectors.structured.deep_risk_quality import (
    # 估值扩散与拥挤度
    get_a_pe_pb_ew_median,
    get_market_congestion,
    get_stock_bond_spread,
    get_buffett_indicator,
    
    # 资产质量异常
    get_stock_goodwill,
    get_goodwill_impairment,
    get_break_net_stock,
    
    # ESG评级
    get_esg_msci,
    get_esg_hz,
    get_esg_refinitiv,
    get_esg_zhiding,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


def save_to_csv(df: pd.DataFrame, filename: str, data_dir: str = 'data/raw/deep_risk_quality'):
    """
    保存DataFrame到CSV文件
    
    Args:
        df: 要保存的DataFrame
        filename: 文件名
        data_dir: 数据目录
    """
    if df.empty:
        logger.warning(f"{filename}: 数据为空，跳过保存")
        return
    
    # 创建目录
    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存文件
    filepath = output_dir / filename
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    # 检查空列
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        logger.warning(f"{filename}: 以下列全为空值: {empty_cols}")
    
    logger.info(f"已保存 {filename}: {len(df)} 条记录, {len(df.columns)} 列")


def collect_valuation_dispersion_data(start_date: str, end_date: str):
    """
    采集估值扩散与拥挤度数据
    
    Args:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)
    """
    logger.info("=" * 80)
    logger.info("开始采集估值扩散与拥挤度数据")
    logger.info("=" * 80)
    
    # A股市盈率/市净率
    logger.info("采集A股市盈率/市净率...")
    try:
        pe_pb_data = get_a_pe_pb_ew_median(start_date=start_date, end_date=end_date)
        save_to_csv(pe_pb_data, 'a_pe_pb_ew_median.csv', 'data/raw/deep_risk_quality/valuation')
    except Exception as e:
        logger.error(f"采集A股市盈率/市净率失败: {e}")
    
    # 大盘拥挤度
    logger.info("采集大盘拥挤度...")
    try:
        congestion_data = get_market_congestion(start_date=start_date, end_date=end_date)
        save_to_csv(congestion_data, 'market_congestion.csv', 'data/raw/deep_risk_quality/valuation')
    except Exception as e:
        logger.error(f"采集大盘拥挤度失败: {e}")
    
    # 股债利差
    logger.info("采集股债利差...")
    try:
        spread_data = get_stock_bond_spread(start_date=start_date, end_date=end_date)
        save_to_csv(spread_data, 'stock_bond_spread.csv', 'data/raw/deep_risk_quality/valuation')
    except Exception as e:
        logger.error(f"采集股债利差失败: {e}")
    
    # 巴菲特指标
    logger.info("采集巴菲特指标...")
    try:
        buffett_data = get_buffett_indicator(start_date=start_date, end_date=end_date)
        save_to_csv(buffett_data, 'buffett_indicator.csv', 'data/raw/deep_risk_quality/valuation')
    except Exception as e:
        logger.error(f"采集巴菲特指标失败: {e}")


def collect_asset_quality_data():
    """采集资产质量异常数据"""
    logger.info("=" * 80)
    logger.info("开始采集资产质量异常数据")
    logger.info("=" * 80)
    
    # 个股商誉明细
    logger.info("采集个股商誉明细...")
    try:
        goodwill_data = get_stock_goodwill()
        save_to_csv(goodwill_data, 'stock_goodwill.csv', 'data/raw/deep_risk_quality/asset_quality')
    except Exception as e:
        logger.error(f"采集个股商誉明细失败: {e}", exc_info=False)
    
    # 商誉减值预期
    logger.info("采集商誉减值预期...")
    try:
        impairment_data = get_goodwill_impairment()
        save_to_csv(impairment_data, 'goodwill_impairment.csv', 'data/raw/deep_risk_quality/asset_quality')
    except Exception as e:
        logger.error(f"采集商誉减值预期失败: {e}", exc_info=False)
    
    # 破净股统计（此接口可能有网络问题,跳过）
    logger.info("采集破净股统计...")
    try:
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("采集超时")
        
        # 设置30秒超时（仅Unix系统）
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            break_net_data = get_break_net_stock()
            signal.alarm(0)
            save_to_csv(break_net_data, 'break_net_stock.csv', 'data/raw/deep_risk_quality/asset_quality')
        except AttributeError:
            # Windows系统没有SIGALRM,直接采集
            break_net_data = get_break_net_stock()
            save_to_csv(break_net_data, 'break_net_stock.csv', 'data/raw/deep_risk_quality/asset_quality')
    except (TimeoutError, KeyboardInterrupt) as e:
        logger.warning(f"采集破净股统计超时或被中断,跳过: {e}")
    except Exception as e:
        logger.error(f"采集破净股统计失败: {e}", exc_info=False)


def collect_esg_ratings_data():
    """采集ESG评级数据"""
    logger.info("=" * 80)
    logger.info("开始采集ESG评级数据")
    logger.info("=" * 80)
    
    # MSCI ESG评级
    logger.info("采集MSCI ESG评级...")
    try:
        msci_data = get_esg_msci()
        save_to_csv(msci_data, 'esg_msci.csv', 'data/raw/deep_risk_quality/esg')
    except Exception as e:
        logger.error(f"采集MSCI ESG评级失败: {e}", exc_info=False)
    
    # 华证ESG评级
    logger.info("采集华证ESG评级...")
    try:
        hz_data = get_esg_hz()
        save_to_csv(hz_data, 'esg_hz.csv', 'data/raw/deep_risk_quality/esg')
    except Exception as e:
        logger.error(f"采集华证ESG评级失败: {e}", exc_info=False)
    
    # 路孚特ESG评级
    logger.info("采集路孚特ESG评级...")
    try:
        refinitiv_data = get_esg_refinitiv()
        save_to_csv(refinitiv_data, 'esg_refinitiv.csv', 'data/raw/deep_risk_quality/esg')
    except Exception as e:
        logger.error(f"采集路孚特ESG评级失败: {e}", exc_info=False)
    
    # 秩鼎ESG评级
    logger.info("采集秩鼎ESG评级...")
    try:
        zhiding_data = get_esg_zhiding()
        save_to_csv(zhiding_data, 'esg_zhiding.csv', 'data/raw/deep_risk_quality/esg')
    except Exception as e:
        logger.error(f"采集秩鼎ESG评级失败: {e}", exc_info=False)


def create_collection_summary(data_dir: str = 'data/raw/deep_risk_quality'):
    """
    创建采集汇总信息
    
    Args:
        data_dir: 数据目录
    """
    summary_list = []
    data_path = Path(data_dir)
    
    # 遍历所有CSV文件
    for csv_file in data_path.rglob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            summary_list.append({
                '文件路径': str(csv_file.relative_to(data_path)),
                '数据类型': csv_file.stem,
                '记录数': len(df),
                '字段数': len(df.columns),
                '采集时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        except Exception as e:
            logger.warning(f"读取 {csv_file} 失败: {e}")
    
    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        summary_path = data_path / 'collection_summary.csv'
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        logger.info(f"\n采集汇总:\n{summary_df.to_string()}")
        logger.info(f"汇总信息已保存到: {summary_path}")


def main():
    """主函数"""
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("深度风险与质量因子域数据采集开始")
    logger.info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # 设置日期范围（最近一年）
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = '20250101'  # 2025年开始
    
    try:
        # 1. 采集估值扩散与拥挤度数据
        collect_valuation_dispersion_data(start_date, end_date)
        
        # 2. 采集资产质量异常数据
        collect_asset_quality_data()
        
        # 3. 采集ESG评级数据
        collect_esg_ratings_data()
        
        # 4. 创建采集汇总
        create_collection_summary()
        
    except Exception as e:
        logger.error(f"采集过程发生错误: {e}", exc_info=True)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("=" * 80)
    logger.info("深度风险与质量因子域数据采集完成")
    logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总耗时: {duration:.2f} 秒")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
