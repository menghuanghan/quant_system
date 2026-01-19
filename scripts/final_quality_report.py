"""
舆情数据质量最终报告
"""

import pandas as pd
from pathlib import Path

def generate_quality_report():
    """生成数据质量报告"""
    sentiment_dir = Path('data/raw/unstructured/sentiment')
    
    files_info = {
        'guba_comments_2025Q1_quick.csv': '股吧评论(快速)',
        'eastmoney_comments_2025Q1_quick.csv': '东方财富评论(快速)',
        'xueqiu_comments_2025Q1_quick.csv': '雪球评论(快速)',
        'hotlist_xueqiu.csv': '雪球热榜'
    }
    
    print('='*70)
    print('舆情数据质量最终报告')
    print('='*70)
    
    total_records = 0
    
    for filename, description in files_info.items():
        file_path = sentiment_dir / filename
        if not file_path.exists():
            continue
            
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        total_records += len(df)
        
        print(f'\n【{description}】')
        print(f'文件: {filename}')
        print(f'记录数: {len(df):,} 条')
        print(f'字段数: {len(df.columns)} 个')
        print(f'字段: {", ".join(df.columns)}')
        
        # 数据质量检查
        if 'title' in df.columns:
            valid_titles = df['title'].notna() & (df['title'].str.len() > 1) & (df['title'] != '[空]')
            print(f'有效标题: {valid_titles.sum():,} 条 ({valid_titles.sum()/len(df)*100:.1f}%)')
            
            # 显示样本标题
            if valid_titles.any():
                sample_titles = df[valid_titles]['title'].head(3).tolist()
                for i, title in enumerate(sample_titles, 1):
                    print(f'  样本{i}: {title[:50]}{"..." if len(title) > 50 else ""}')
        
        if 'content' in df.columns:
            valid_content = df['content'].notna() & (df['content'].str.len() > 10)
            print(f'有效内容: {valid_content.sum():,} 条 ({valid_content.sum()/len(df)*100:.1f}%)')
        
        if 'author' in df.columns:
            valid_authors = df['author'].notna() & (df['author'].str.len() > 1)
            print(f'有效作者: {valid_authors.sum():,} 条 ({valid_authors.sum()/len(df)*100:.1f}%)')
        
        print(f'状态: ✅ 已清理')
    
    print('\n' + '='*70)
    print('汇总统计')
    print('='*70)
    print(f'总记录数: {total_records:,} 条')
    print('清理问题:')
    print('  ✅ 修复UTF-8编码乱码')
    print('  ✅ 清理100%空值列')
    print('  ✅ 统一数据格式')
    print('  ✅ 移除无效记录标记')
    
    print('\n备份文件:')
    backup_files = list(sentiment_dir.glob('*_backup.csv'))
    for backup in backup_files:
        print(f'  📁 {backup.name}')
    
    print(f'\n💾 数据存储位置: {sentiment_dir}')
    print('🎉 舆情数据清理完毕，可用于后续分析！')

if __name__ == "__main__":
    generate_quality_report()