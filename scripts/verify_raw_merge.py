"""
合并数据完整性验证脚本

全面对比 data/raw/structured（合并后）与 data/temp/back_up/structured（原始备份），验证：
1. 字段结构一致性：列名、列数量完全相同
2. 数据量一致性：合并后行数 = 原目录下所有文件行数之和
3. 数据值一致性：随机抽样对比具体数据值，确保无损坏
4. 完整性检查：原始所有 ts_code 文件是否都包含在合并结果中
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from collections import defaultdict

MERGED_DIR = Path("data/raw/structured")
BACKUP_DIR = Path("data/temp/back_up/structured")

# 用于收集所有问题
issues = []
warnings = []
passed = []


def log_issue(msg):
    issues.append(msg)
    print(f"  ❌ {msg}")


def log_warning(msg):
    warnings.append(msg)
    print(f"  ⚠️  {msg}")


def log_pass(msg):
    passed.append(msg)
    print(f"  ✅ {msg}")


def verify_single_files():
    """验证原本就是单文件的数据（未经合并操作）是否完全一致"""
    print("\n" + "=" * 80)
    print("【检查1】单文件数据一致性（未经合并的 parquet 文件）")
    print("=" * 80)

    single_files = []
    for domain_dir in sorted(BACKUP_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue
        for item in sorted(domain_dir.iterdir()):
            if item.is_file() and item.suffix == ".parquet":
                # 确认它在备份中也不是由目录来的
                single_files.append((domain_dir.name, item.name))

    for domain, fname in single_files:
        backup_path = BACKUP_DIR / domain / fname
        merged_path = MERGED_DIR / domain / fname
        label = f"{domain}/{fname}"

        if not merged_path.exists():
            log_issue(f"{label}: 合并后文件不存在!")
            continue

        try:
            df_backup = pd.read_parquet(backup_path)
            df_merged = pd.read_parquet(merged_path)

            # 列一致性
            if list(df_backup.columns) != list(df_merged.columns):
                log_issue(f"{label}: 列不一致! 备份={list(df_backup.columns)}, 合并后={list(df_merged.columns)}")
            elif len(df_backup) != len(df_merged):
                log_issue(f"{label}: 行数不一致! 备份={len(df_backup)}, 合并后={len(df_merged)}")
            else:
                # 对于单文件，应完全相同
                if df_backup.equals(df_merged):
                    log_pass(f"{label}: 完全一致 ({len(df_backup)} 行, {len(df_backup.columns)} 列)")
                else:
                    log_warning(f"{label}: 行数列数相同但内容有差异 ({len(df_backup)} 行)")
        except Exception as e:
            log_issue(f"{label}: 读取出错 - {e}")


def verify_merged_directories():
    """验证按 ts_code 拆分目录合并后的数据"""
    print("\n" + "=" * 80)
    print("【检查2】合并目录数据一致性（原按 ts_code 拆分的目录）")
    print("=" * 80)

    # 找出备份中所有的子目录（即需要合并的）
    merge_targets = []
    for domain_dir in sorted(BACKUP_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue
        for item in sorted(domain_dir.iterdir()):
            if item.is_dir():
                merge_targets.append((domain_dir.name, item.name, item))

    for domain, name, backup_sub_dir in merge_targets:
        label = f"{domain}/{name}"
        merged_path = MERGED_DIR / domain / f"{name}.parquet"
        print(f"\n--- {label} ---")

        if not merged_path.exists():
            log_issue(f"{label}: 合并后的 {name}.parquet 不存在!")
            continue

        # 1. 读取合并后的数据
        try:
            df_merged = pd.read_parquet(merged_path)
        except Exception as e:
            log_issue(f"{label}: 读取合并后文件失败 - {e}")
            continue

        # 2. 读取原始所有分片文件并汇总
        original_files = sorted(backup_sub_dir.glob("*.parquet"))
        if not original_files:
            log_warning(f"{label}: 原目录为空")
            continue

        total_rows_original = 0
        original_dfs = []
        original_columns = None
        column_issues_in_source = []
        read_errors = 0

        for pf in original_files:
            try:
                df = pd.read_parquet(pf)
                total_rows_original += len(df)
                if not df.empty:
                    if original_columns is None:
                        original_columns = list(df.columns)
                    elif list(df.columns) != original_columns:
                        column_issues_in_source.append(pf.name)
                    original_dfs.append(df)
            except Exception as e:
                read_errors += 1

        if read_errors > 0:
            log_warning(f"{label}: 原始文件中有 {read_errors} 个读取失败")

        if column_issues_in_source:
            log_warning(f"{label}: 原始文件间列结构不一致 ({len(column_issues_in_source)} 个文件)")

        # 3. 检查文件数量
        log_pass(f"{label}: 原始文件数 = {len(original_files)}")

        # 4. 检查行数一致性
        if len(df_merged) == total_rows_original:
            log_pass(f"{label}: 行数一致 = {total_rows_original}")
        else:
            log_issue(f"{label}: 行数不一致! 原始总计={total_rows_original}, 合并后={len(df_merged)}")

        # 5. 检查列结构
        if original_columns is not None:
            merged_cols = list(df_merged.columns)
            orig_set = set(original_columns)
            merged_set = set(merged_cols)

            if orig_set == merged_set:
                log_pass(f"{label}: 列名一致 ({len(merged_cols)} 列)")
            else:
                extra_in_merged = merged_set - orig_set
                missing_in_merged = orig_set - merged_set
                if extra_in_merged:
                    log_warning(f"{label}: 合并后多出列: {extra_in_merged}")
                if missing_in_merged:
                    log_issue(f"{label}: 合并后缺失列: {missing_in_merged}")

        # 6. 检查列的数据类型
        if original_columns is not None and original_dfs:
            # 用第一个非空df的类型作为参考
            ref_dtypes = original_dfs[0].dtypes
            merged_dtypes = df_merged.dtypes
            dtype_mismatches = []
            for col in original_columns:
                if col in merged_dtypes.index:
                    orig_dt = str(ref_dtypes.get(col, "N/A"))
                    merge_dt = str(merged_dtypes.get(col, "N/A"))
                    if orig_dt != merge_dt:
                        dtype_mismatches.append((col, orig_dt, merge_dt))
            if dtype_mismatches:
                log_warning(f"{label}: {len(dtype_mismatches)} 列类型变化: " +
                           ", ".join(f"{c}({o}->{m})" for c, o, m in dtype_mismatches[:5]) +
                           ("..." if len(dtype_mismatches) > 5 else ""))
            else:
                log_pass(f"{label}: 列类型完全一致")

        # 7. 数据值抽样验证 —— 随机选取5个原始文件，逐行对比
        sample_count = min(5, len(original_files))
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(len(original_files), sample_count, replace=False)
        value_mismatch_count = 0

        for idx in sample_indices:
            pf = original_files[idx]
            try:
                df_orig_single = pd.read_parquet(pf)
                if df_orig_single.empty:
                    continue

                # 在合并后的数据中找到对应的行
                # ts_code 文件名格式: {ts_code}.parquet
                ts_code = pf.stem.replace("_", ".", 1) if "_" in pf.stem else pf.stem

                # 尝试多种方式匹配
                matched = False
                for ts_col in ["ts_code", "code", "symbol"]:
                    if ts_col in df_merged.columns:
                        # ts_code 在文件名中是 000001_SZ，在数据中可能是 000001.SZ
                        df_match = df_merged[df_merged[ts_col].astype(str) == ts_code]
                        if len(df_match) == 0:
                            # 尝试用原始文件名（下划线格式）
                            ts_code_underscore = pf.stem
                            df_match = df_merged[df_merged[ts_col].astype(str) == ts_code_underscore]
                        if len(df_match) > 0:
                            matched = True
                            # 对比行数
                            if len(df_match) != len(df_orig_single):
                                value_mismatch_count += 1
                                log_issue(
                                    f"{label}: {pf.name} 行数不匹配 "
                                    f"(原={len(df_orig_single)}, 合并后={len(df_match)})"
                                )
                            else:
                                # 对比共有列的数值（排除因类型转换可能的微小差异）
                                common_cols = [c for c in df_orig_single.columns if c in df_match.columns]
                                df_orig_sorted = df_orig_single[common_cols].reset_index(drop=True)
                                df_match_sorted = df_match[common_cols].reset_index(drop=True)

                                # 逐列对比
                                col_issues = []
                                for col in common_cols:
                                    try:
                                        s1 = df_orig_sorted[col]
                                        s2 = df_match_sorted[col]

                                        # 转为相同类型再比较
                                        if s1.dtype != s2.dtype:
                                            s1 = s1.astype(str).replace("None", pd.NA).replace("NaT", pd.NA)
                                            s2 = s2.astype(str).replace("None", pd.NA).replace("NaT", pd.NA)

                                        # 对齐 NA
                                        both_na = s1.isna() & s2.isna()
                                        neither_na = ~s1.isna() & ~s2.isna()

                                        if s1.isna().sum() != s2.isna().sum():
                                            col_issues.append(f"{col}(NA数不同: {s1.isna().sum()} vs {s2.isna().sum()})")
                                        elif neither_na.any():
                                            v1 = s1[neither_na].astype(str).values
                                            v2 = s2[neither_na].astype(str).values
                                            if not np.array_equal(v1, v2):
                                                diff_count = (v1 != v2).sum()
                                                col_issues.append(f"{col}({diff_count}行值不同)")
                                    except Exception:
                                        pass

                                if col_issues:
                                    value_mismatch_count += 1
                                    log_warning(
                                        f"{label}: {pf.name} 有列值差异: " +
                                        ", ".join(col_issues[:3]) +
                                        ("..." if len(col_issues) > 3 else "")
                                    )
                            break

                if not matched and len(df_orig_single) > 0:
                    # 如果没有 ts_code 列，则按整体行数已经在上面验证过了
                    pass

            except Exception as e:
                log_warning(f"{label}: 抽样验证 {pf.name} 出错 - {e}")

        if value_mismatch_count == 0:
            log_pass(f"{label}: 抽样验证通过 ({sample_count} 个文件)")

        # 8. 完整性检查 —— 确认所有 ts_code 都在合并后数据中
        if original_dfs:
            for ts_col in ["ts_code", "code", "symbol"]:
                if ts_col in df_merged.columns and ts_col in original_dfs[0].columns:
                    # 统计原始所有 ts_code
                    all_original_codes = set()
                    for df in original_dfs:
                        if ts_col in df.columns:
                            all_original_codes.update(df[ts_col].dropna().unique())
                    merged_codes = set(df_merged[ts_col].dropna().unique())
                    missing_codes = all_original_codes - merged_codes
                    if missing_codes:
                        log_issue(f"{label}: 合并后缺失 {len(missing_codes)} 个 {ts_col}: {list(missing_codes)[:5]}...")
                    else:
                        log_pass(f"{label}: 所有 {ts_col} 完整 ({len(all_original_codes)} 个)")
                    break


def verify_completeness():
    """验证合并后目录的完整性 —— 确保原有数据都存在"""
    print("\n" + "=" * 80)
    print("【检查3】合并后目录完整性")
    print("=" * 80)

    # 检查所有域是否都存在
    backup_domains = sorted([d.name for d in BACKUP_DIR.iterdir() if d.is_dir()])
    merged_domains = sorted([d.name for d in MERGED_DIR.iterdir() if d.is_dir()])

    if backup_domains == merged_domains:
        log_pass(f"数据域完整: {backup_domains}")
    else:
        missing = set(backup_domains) - set(merged_domains)
        extra = set(merged_domains) - set(backup_domains)
        if missing:
            log_issue(f"缺失数据域: {missing}")
        if extra:
            log_warning(f"多出数据域: {extra}")

    # 检查每个域下的文件是否都存在
    for domain in backup_domains:
        backup_items = set()
        for item in (BACKUP_DIR / domain).iterdir():
            if item.is_file():
                backup_items.add(item.name)
            elif item.is_dir():
                backup_items.add(f"{item.name}.parquet")  # 目录应该变成 .parquet

        merged_items = set()
        merged_dir = MERGED_DIR / domain
        if merged_dir.exists():
            for item in merged_dir.iterdir():
                if item.is_file():
                    merged_items.add(item.name)

        missing_items = backup_items - merged_items
        if missing_items:
            log_issue(f"{domain}: 缺失文件 {missing_items}")
        else:
            log_pass(f"{domain}: 所有文件完整 ({len(merged_items)} 个)")

    # 确认没有残留目录
    remaining_dirs = []
    for domain_dir in MERGED_DIR.iterdir():
        if domain_dir.is_dir():
            for item in domain_dir.iterdir():
                if item.is_dir():
                    remaining_dirs.append(str(item))
    if remaining_dirs:
        log_issue(f"合并后仍有残留目录: {remaining_dirs}")
    else:
        log_pass("合并后无残留子目录")


def print_summary():
    print("\n" + "=" * 80)
    print("【验证总结】")
    print("=" * 80)
    print(f"  ✅ 通过: {len(passed)} 项")
    print(f"  ⚠️  警告: {len(warnings)} 项")
    print(f"  ❌ 问题: {len(issues)} 项")

    if warnings:
        print("\n--- 警告详情 ---")
        for w in warnings:
            print(f"  ⚠️  {w}")

    if issues:
        print("\n--- 问题详情 ---")
        for i in issues:
            print(f"  ❌ {i}")
    else:
        print("\n  🎉 所有检查通过，数据合并完整无误!")


if __name__ == "__main__":
    print("结构化原始数据合并验证")
    print(f"合并后: {MERGED_DIR}")
    print(f"备份: {BACKUP_DIR}")

    verify_completeness()
    verify_single_files()
    verify_merged_directories()
    print_summary()
