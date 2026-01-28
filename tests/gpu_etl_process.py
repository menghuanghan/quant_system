import cudf
import time
import os

# --- 路径配置 ---
# 确保这个路径指向你那 5000 多个小文件的文件夹
RAW_DIR = "data/raw/structured/market_data/stock_daily"
OUTPUT_FILE = "data/processed/all_stocks_clean.parquet"

def fast_gpu_batch_process():
    print(f"🚀 正在启动 RTX 5070 (Blackwell) 处理引擎...")
    print(f"📂 扫描目录: {RAW_DIR}")
    t0 = time.time()

    # 1. 批量并行读取 (GPU 核心优势)
    try:
        # cuDF 会自动并发读取目录下所有 Parquet 并合并
        df = cudf.read_parquet(RAW_DIR)
        print(f"✅ 批量加载成功! 规模: {len(df):,} 行")
    except Exception as e:
        print(f"❌ 读取失败，请检查路径。错误: {e}")
        return

    # 2. 稳健的日期转换 (解决 ValueError 的核心逻辑)
    print("⏳ 正在进行日期格式标准化...")
    try:
        # 方案：先强制转为日期对象（自动处理 2023-01-01 或 2023/01/01）
        temp_date = cudf.to_datetime(df['trade_date'], errors='coerce')
        
        # 转换为 YYYYMMDD 格式的整数，方便后续极速检索
        df['trade_date'] = (temp_date.dt.year * 10000 + 
                            temp_date.dt.month * 100 + 
                            temp_date.dt.day).astype('int32')
        
        # 剔除掉日期非法的行
        df = df.dropna(subset=['trade_date'])
        print("✅ 日期清洗完成 (格式: YYYYMMDD)")
    except Exception as e:
        print(f"⚠️ 日期处理出现小插曲，尝试备用方案... 错误: {e}")
        # 备用方案：直接去掉非数字字符
        df['trade_date'] = df['trade_date'].str.replace('-', '').astype('int32')

    # 3. 基础价格清洗
    # 剔除开盘价或收盘价为空的无效交易日
    df = df.dropna(subset=['close', 'open'])

    # 4. 并行计算后复权 (GPU 向量化乘法)
    if 'adj_factor' in df.columns:
        print("⚡ 正在执行全市场后复权计算...")
        # 直接计算，RTX 5070 处理这几百万行只需不到 0.1 秒
        df['adj_close'] = df['close'] * df['adj_factor']
        df['adj_open'] = df['open'] * df['adj_factor']
        df['adj_high'] = df['high'] * df['adj_factor']
        df['adj_low'] = df['low'] * df['adj_factor']
    else:
        print("⚠️ 未发现 adj_factor 列，跳过复权计算")

    # 5. 导出最终大表
    print(f"💾 正在导出合并后的清洗数据至: {OUTPUT_FILE}")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 使用 Snappy 压缩，兼顾速度和空间
    df.to_parquet(OUTPUT_FILE, compression='snappy')
    
    t_end = time.time()
    print("-" * 50)
    print(f"🏁 全流程处理结束！")
    print(f"⏱️ 总耗时: {t_end - t0:.2f} 秒")
    print(f"📊 最终有效数据: {len(df):,} 行")
    print("-" * 50)

if __name__ == "__main__":
    fast_gpu_batch_process()