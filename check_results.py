import pandas as pd
from pathlib import Path

files = sorted(Path('data/raw/unstructured').rglob('*.parquet'))
print(f'Total files: {len(files)}\n')

for f in files:
    df = pd.read_parquet(f)
    rel_path = f.relative_to(Path('data/raw/unstructured'))
    print(f'{str(rel_path): <50} {len(df):>8} rows')

print(f'\nTotal records: {sum(len(pd.read_parquet(f)) for f in files)}')
