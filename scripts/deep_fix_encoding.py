"""
深度修复舆情数据乱码问题

处理编码误解析导致的乱码文本
"""

import pandas as pd
import re
from pathlib import Path

def deep_fix_encoding(text):
    """深度修复编码问题"""
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # 如果文本很短且只有乱码，尝试替换为合理内容
    if len(text.strip()) <= 3 and any(char in text for char in ['ï¼', 'è', 'ä', 'å']):
        return '[空]'
    
    # 常见乱码修复映射
    fixes = {
        'ï¼': '，',
        'ï¼': '。',
        'ï¼': '？',  
        'ï¼': '！',
        'ã': '「',
        'ã': '」',
        'è´µå·èå°': '贵州茅台',
        'ä¸­å½å¹³å®': '中国平安',
        'æç': '招商',
        'è¿æ¥æ¬¢': '远景',
        'æ¸¸ç ': '海康',
        'éå': '顺丰',
    }
    
    # 应用修复
    for bad, good in fixes.items():
        text = text.replace(bad, good)
    
    # 如果仍然有大量乱码，尝试重新编码
    garbled_chars = ['è', 'ä', 'å', 'ç', 'é', 'ï', 'ã', 'â']
    if sum(char in text for char in garbled_chars) >= 3:
        try:
            # 假设是UTF-8被错误解析为ISO-8859-1
            fixed = text.encode('iso-8859-1').decode('utf-8')
            return fixed
        except:
            # 如果重编码失败，返回清理后的文本
            for char in garbled_chars:
                text = text.replace(char, '')
    
    return text.strip()

def fix_sentiment_files():
    """修复舆情文件"""
    sentiment_dir = Path('data/raw/unstructured/sentiment')
    
    files = [
        'guba_comments_2025Q1_quick.csv',
        'eastmoney_comments_2025Q1_quick.csv'
    ]
    
    for filename in files:
        file_path = sentiment_dir / filename
        if not file_path.exists():
            continue
            
        print(f'\n修复文件: {filename}')
        
        # 读取数据
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # 修复文本列
        text_columns = ['title', 'author', 'content']
        for col in text_columns:
            if col in df.columns:
                print(f'  修复列: {col}')
                df[col] = df[col].apply(deep_fix_encoding)
        
        # 保存修复后的文件
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f'  ✓ 保存完成')
        
        # 显示修复效果
        if 'title' in df.columns:
            print('  修复后样本:')
            for i, title in enumerate(df['title'].head(3), 1):
                print(f'    {i}. {title}')

def main():
    print('='*50)
    print('深度修复舆情数据乱码')
    print('='*50)
    
    fix_sentiment_files()
    
    print('\n✅ 深度修复完成！')

if __name__ == "__main__":
    main()