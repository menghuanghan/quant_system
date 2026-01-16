import pandas as pd
from pathlib import Path

data_dir = Path('data/raw/structured/expectations')
files = list(data_dir.glob('*.csv'))

print('采集到的文件:')
for f in files:
    df = pd.read_csv(f)
    print(f'\n{f.name}: {len(df)} 条数据')
    print('列名:', df.columns.tolist())
    
    nulls = df.isnull().sum()
    high_nulls = nulls[nulls / len(df) > 0.2]
    if len(high_nulls) > 0:
        print(f'空列 (>20%):')
        for col in high_nulls.index:
            ratio = high_nulls[col] / len(df) * 100
            print(f'  - {col}: {ratio:.1f}%')
