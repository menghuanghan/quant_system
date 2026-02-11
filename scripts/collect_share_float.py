#!/usr/bin/env python3
"""
采集 share_float（限售解禁）全量数据
按月分批 + offset 分页 + 截断自动拆分
"""
import os, sys, signal, time, calendar, argparse
from datetime import datetime, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import tushare as ts

ts.set_token(os.environ['TUSHARE_TOKEN'])
pro = ts.pro_api()
print('API ready')

PAGE_LIMIT = 6000      # Tushare 单次上限
MAX_OFFSET = 96000     # 服务端 offset 上限（保守值）
OUT_PATH = 'data/raw/structured/fundamental/share_float.parquet'


def timeout_handler(signum, frame):
    raise TimeoutError('API call hung!')

signal.signal(signal.SIGALRM, timeout_handler)


def fetch_with_pagination(start_date, end_date):
    """带分页的 share_float 采集。返回 (DataFrame, is_truncated)"""
    all_chunks = []
    offset = 0
    truncated = False

    for _ in range(20):
        if offset >= MAX_OFFSET:
            truncated = True
            break
        signal.alarm(30)
        try:
            df = pro.share_float(
                start_date=start_date,
                end_date=end_date,
                limit=PAGE_LIMIT,
                offset=offset
            )
            signal.alarm(0)
        except TimeoutError:
            signal.alarm(0)
            break
        except Exception:
            signal.alarm(0)
            if offset > 0:
                truncated = True
            break

        if df is None or df.empty:
            break
        all_chunks.append(df)
        if len(df) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
        time.sleep(0.3)

    result = pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()
    return result, truncated


def fetch_period(sd, ed, depth=0):
    """获取一个时间段的数据，截断时自动拆分成更小的段"""
    indent = '    ' * depth
    df, truncated = fetch_with_pagination(sd, ed)
    rows = len(df)

    if not truncated or depth >= 3:
        print(f'{indent}  {sd}-{ed}: {rows:,} rows')
        return df

    # 截断了，拆成两半
    d1 = datetime.strptime(sd, '%Y%m%d')
    d2 = datetime.strptime(ed, '%Y%m%d')
    if (d2 - d1).days <= 1:
        print(f'{indent}  {sd}-{ed}: {rows:,} rows (无法再拆分)')
        return df

    mid = d1 + (d2 - d1) // 2
    mid_str = mid.strftime('%Y%m%d')
    mid_next = (mid + timedelta(days=1)).strftime('%Y%m%d')

    print(f'{indent}  {sd}-{ed}: {rows:,} rows (截断, 拆分...)')
    df1 = fetch_period(sd, mid_str, depth + 1)
    time.sleep(0.3)
    df2 = fetch_period(mid_next, ed, depth + 1)

    parts = [p for p in [df, df1, df2] if not p.empty]
    if parts:
        return pd.concat(parts, ignore_index=True).drop_duplicates()
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description='采集 share_float 数据')
    parser.add_argument('--start-date', default='20190101')
    parser.add_argument('--end-date', default='20251231')
    parser.add_argument('--append', action='store_true', help='追加到现有文件')
    args = parser.parse_args()

    START, END = args.start_date, args.end_date
    print(f'采集范围: {START} ~ {END}')

    all_dfs = []

    # 追加模式
    if args.append and os.path.exists(OUT_PATH):
        existing = pd.read_parquet(OUT_PATH)
        all_dfs.append(existing)
        print(f'追加模式: 已加载 {len(existing):,} 行')

    start_y, start_m = int(START[:4]), int(START[4:6])
    end_y, end_m = int(END[:4]), int(END[4:6])
    y, m = start_y, start_m

    while (y, m) <= (end_y, end_m):
        last_day = calendar.monthrange(y, m)[1]
        sd = f'{y}{m:02d}01'
        ed = f'{y}{m:02d}{last_day:02d}'
        if sd < START:
            sd = START
        if ed > END:
            ed = END

        df = fetch_period(sd, ed)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(0.3)

        m += 1
        if m > 12:
            m = 1
            y += 1

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        n_before = len(result)
        result = result.drop_duplicates()
        n_after = len(result)
        print(f'\nTotal: {n_before:,} -> {n_after:,} after dedup')
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        result.to_parquet(OUT_PATH, index=False, engine='pyarrow', compression='snappy')
        print(f'Saved to {OUT_PATH}')
        print(f'float_date range: {result["float_date"].min()} ~ {result["float_date"].max()}')
        print(f'Unique ts_code: {result["ts_code"].nunique()}')
    else:
        print('\nNo data collected!')


if __name__ == '__main__':
    main()
