import pandas as pd
from pathlib import Path
import os

dwd_dir = Path('data/processed/structured/dwd')
out_dir = Path('data/temp/dwd')
out_dir.mkdir(parents=True, exist_ok=True)

tables = [
    'dwd_stock_price', 'dwd_stock_fundamental', 'dwd_stock_status',
    'dwd_money_flow', 'dwd_chip_structure', 'dwd_stock_industry',
    'dwd_event_signal', 'dwd_macro_env'
]

N = 200000

for name in tables:
    path = dwd_dir / f'{name}.parquet'
    df = pd.read_parquet(path)
    total = len(df)
    
    # 前 200000 行
    head_df = df.head(N)
    head_path = out_dir / f'{name}_head_{N}.csv'
    head_df.to_csv(head_path, index=False)
    
    # 中间连续 200000 行
    mid_start = max(0, total // 2 - N // 2)
    mid_df = df.iloc[mid_start:mid_start + N]
    mid_path = out_dir / f'{name}_mid_{N}.csv'
    mid_df.to_csv(mid_path, index=False)
    
    # 后 200000 行
    tail_df = df.tail(N)
    tail_path = out_dir / f'{name}_tail_{N}.csv'
    tail_df.to_csv(tail_path, index=False)
    
    # 输出文件信息
    for p in [head_path, mid_path, tail_path]:
        size_kb = p.stat().st_size / 1024
        print(f'  {p.name}: {size_kb:.0f} KB')
    
    print(f'✓ {name} 完成 (总行数: {total:,})')
    print()

print('全部完成！')