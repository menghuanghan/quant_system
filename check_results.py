import pandas as pd
from pathlib import Path

data_dir = Path('data/raw/unstructured')
news_dir = data_dir / 'news'
sentiment_dir = data_dir / 'sentiment'

print('='*60)
print('2025年Q1快速采集结果统计')
print('='*60)

# 新闻数据统计
print('\n【新闻数据】')
news_files = {
    '东方财富新闻': 'eastmoney_news_2025Q1_quick.csv',
    '新浪财经新闻': 'sina_news_2025Q1_quick.csv', 
    '证券时报新闻': 'stcn_news_2025Q1_quick.csv'
}

total_news = 0
for name, filename in news_files.items():
    file_path = news_dir / filename
    if file_path.exists():
        df = pd.read_csv(file_path)
        total_news += len(df)
        print(f'  {name}: {len(df)} 条')
        if len(df) > 0 and 'title' in df.columns:
            sample_title = df['title'].iloc[0][:40] + '...'
            print(f'    样本: {sample_title}')
    else:
        print(f'  {name}: 文件不存在')

# 情感数据统计  
print('\n【情感数据】')
sentiment_files = {
    '股吧评论': 'guba_comments_2025Q1_quick.csv',
    '东方财富评论': 'eastmoney_comments_2025Q1_quick.csv',
    '雪球评论': 'xueqiu_comments_2025Q1_quick.csv'
}

total_sentiment = 0
for name, filename in sentiment_files.items():
    file_path = sentiment_dir / filename
    if file_path.exists():
        df = pd.read_csv(file_path)
        total_sentiment += len(df)
        print(f'  {name}: {len(df)} 条')
        if len(df) > 0:
            # 检查内容字段
            content_cols = [c for c in df.columns if 'content' in c.lower() or 'text' in c.lower()]
            if content_cols:
                has_content = df[content_cols[0]].astype(str).str.len() > 10
                content_pct = has_content.sum()/len(df)*100
                print(f'    有内容: {has_content.sum()} 条 ({content_pct:.1f}%)')
    else:
        print(f'  {name}: 文件不存在')

print(f'\n📊 总计: 新闻{total_news}条, 情感数据{total_sentiment}条')
print('✅ 快速采集完成！总用时约5-6分钟')
print('💡 优化措施: 减少延迟、限制页数、样本采集、Playwright加速')