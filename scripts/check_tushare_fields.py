import tushare as ts
import os
from dotenv import load_dotenv

load_dotenv()
pro = ts.pro_api(os.getenv('TUSHARE_TOKEN'))

def check_interface(name, **kwargs):
    print(f"\nChecking {name}...")
    try:
        api_func = getattr(pro, name)
        df = api_func(**kwargs)
        if not df.empty:
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  Row 0: {df.iloc[0].to_dict()}")
        else:
            print("  Empty")
    except Exception as e:
        print(f"  Error: {e}")

check_interface('strategy_gold_stock', month='202401')
check_interface('broker_recommend', month='202401')
check_interface('report_rc', ts_code='600519.SH', limit=1)
