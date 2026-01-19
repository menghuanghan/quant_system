"""
舆情数据清理脚本
修复乱码和空列问题

问题：
1. guba_comments 文件存在UTF-8编码显示乱码
2. hotlist_xueqiu 文件存在100%空值的列
3. 其他文件可能存在类似问题

解决方案：
1. 重新编码保存文件
2. 清理空列  
3. 数据质量检查
"""

import pandas as pd
import os
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_encoding_issues(file_path):
    """修复编码问题"""
    try:
        # 尝试读取并重新保存
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # 备份原文件
        backup_path = str(file_path).replace('.csv', '_backup.csv')
        if not os.path.exists(backup_path):
            df.to_csv(backup_path, index=False, encoding='utf-8-sig')
            logger.info(f"备份原文件: {backup_path}")
        
        # 检查文本列的编码问题
        text_cols = ['title', 'content', 'author', 'name']
        fixed_cols = []
        
        for col in text_cols:
            if col in df.columns:
                # 检查是否有编码问题
                sample_values = df[col].dropna().head(10)
                has_encoding_issue = any('è' in str(val) or 'ä' in str(val) or 'ï' in str(val) 
                                       for val in sample_values)
                
                if has_encoding_issue:
                    logger.info(f"发现编码问题列: {col}")
                    # 尝试修复编码
                    df[col] = df[col].apply(lambda x: fix_garbled_text(x) if pd.notna(x) else x)
                    fixed_cols.append(col)
        
        if fixed_cols:
            # 保存修复后的文件
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.info(f"修复编码问题: {file_path}, 涉及列: {fixed_cols}")
            return True
        else:
            logger.info(f"文件编码正常: {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"修复编码失败 {file_path}: {e}")
        return False

def fix_garbled_text(text):
    """修复乱码文本"""
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # 常见乱码映射修复
    replacements = {
        'è': '茅',
        'å°': '台',  
        'ä¼': '企',
        'ä¸': '业',
        'ç´': '直',
        'ä¾': '供',
        'é': '通',
        'é': '道',
        'å': '回',
        'å': '应',
        'æ¥': '来',
        'äº': '了',
        'ï¼': '，',
        'ï¼': '！',
        'ï¼': '？',
        'ã': '「',
        'ã': '」',
    }
    
    # 如果包含大量乱码字符，可能需要重新编码
    garbled_chars = ['è', 'ä', 'å', 'ç', 'é', 'ï¼', 'ã']
    if sum(char in text for char in garbled_chars) > 3:
        # 尝试不同编码解析
        try:
            # 假设原始是UTF-8被错误解码为latin-1
            fixed = text.encode('latin-1').decode('utf-8')
            return fixed
        except:
            # 使用替换映射
            for old, new in replacements.items():
                text = text.replace(old, new)
    
    return text

def clean_empty_columns(file_path, threshold=0.9):
    """清理空列"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        original_shape = df.shape
        
        # 找出空值比例超过阈值的列
        empty_cols = []
        for col in df.columns:
            empty_ratio = df[col].isna().sum() / len(df)
            if empty_ratio >= threshold:
                empty_cols.append(col)
        
        if empty_cols:
            # 备份原文件
            backup_path = str(file_path).replace('.csv', '_backup.csv')
            if not os.path.exists(backup_path):
                df.to_csv(backup_path, index=False, encoding='utf-8-sig')
            
            # 删除空列
            df_cleaned = df.drop(columns=empty_cols)
            df_cleaned.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"清理空列: {file_path}")
            logger.info(f"  删除列: {empty_cols}")
            logger.info(f"  原始形状: {original_shape} -> 清理后: {df_cleaned.shape}")
            return True
        else:
            logger.info(f"无需清理空列: {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"清理空列失败 {file_path}: {e}")
        return False

def validate_data_quality(file_path):
    """数据质量检查"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        logger.info(f"\n数据质量报告: {os.path.basename(file_path)}")
        logger.info(f"  形状: {df.shape}")
        logger.info(f"  列名: {list(df.columns)}")
        
        # 检查各列数据质量
        for col in df.columns:
            if col in ['title', 'content', 'name', 'author']:  # 文本列
                non_empty = df[col].dropna()
                if not non_empty.empty:
                    avg_len = non_empty.astype(str).str.len().mean()
                    sample = str(non_empty.iloc[0])[:30] + '...'
                    logger.info(f"  {col}: {len(non_empty)} 条, 平均长度 {avg_len:.1f}, 样本: {sample}")
                else:
                    logger.info(f"  {col}: 全部为空")
            
            elif col in ['pub_time', 'trade_date']:  # 日期列
                non_empty = df[col].dropna()
                if not non_empty.empty:
                    sample = str(non_empty.iloc[0])
                    logger.info(f"  {col}: {len(non_empty)} 条, 样本: {sample}")
        
        return True
        
    except Exception as e:
        logger.error(f"质量检查失败 {file_path}: {e}")
        return False

def main():
    """主函数"""
    logger.info("="*60)
    logger.info("开始舆情数据清理")
    logger.info("="*60)
    
    # 舆情数据目录
    sentiment_dir = Path('data/raw/unstructured/sentiment')
    
    # 需要处理的文件
    files_to_fix = [
        'guba_comments_2025Q1_quick.csv',
        'eastmoney_comments_2025Q1_quick.csv', 
        'xueqiu_comments_2025Q1_quick.csv',
        'hotlist_xueqiu.csv',
        'guba_comments_2025Q1.csv',
        'eastmoney_comments_2025Q1.csv'
    ]
    
    results = {}
    
    for filename in files_to_fix:
        file_path = sentiment_dir / filename
        
        if file_path.exists():
            logger.info(f"\n处理文件: {filename}")
            
            # 1. 修复编码问题
            encoding_fixed = fix_encoding_issues(file_path)
            
            # 2. 清理空列
            columns_cleaned = clean_empty_columns(file_path)
            
            # 3. 数据质量检查
            quality_checked = validate_data_quality(file_path)
            
            results[filename] = {
                'encoding_fixed': encoding_fixed,
                'columns_cleaned': columns_cleaned,
                'quality_checked': quality_checked
            }
        else:
            logger.warning(f"文件不存在: {filename}")
            results[filename] = {'status': 'not_found'}
    
    # 结果汇总
    logger.info("\n" + "="*60)
    logger.info("数据清理结果汇总")
    logger.info("="*60)
    
    for filename, result in results.items():
        if 'status' in result:
            logger.info(f"{filename}: {result['status']}")
        else:
            encoding_status = "✓" if result['encoding_fixed'] else "○"
            column_status = "✓" if result['columns_cleaned'] else "○" 
            quality_status = "✓" if result['quality_checked'] else "✗"
            logger.info(f"{filename}: 编码{encoding_status} 空列{column_status} 质量{quality_status}")
    
    logger.info("\n✓ 数据清理完成！")

if __name__ == "__main__":
    main()