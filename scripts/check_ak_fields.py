import akshare as ak
import pandas as pd

def check_ak(name, **kwargs):
    print(f"\nChecking AkShare {name}...")
    try:
        api_func = getattr(ak, name)
        df = api_func(**kwargs)
        if not df.empty:
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  Row 0 sample: {df.iloc[0].to_dict()}")
        else:
            print("  Empty")
    except Exception as e:
        print(f"  Error: {e}")

# 东财盈利预测
check_ak('stock_profit_forecast_em', symbol='600519')
# 东财个股研报详情
check_ak('stock_institute_recommend_detail', symbol='600519')
# 分析师指数排行
check_ak('stock_analyst_rank_em')
