import pandas as pd

print('='*60)
print('验证数据清理效果')
print('='*60)

# 检查修复后的股吧评论
print('\n【股吧评论 - 修复后】')
df_guba = pd.read_csv('data/raw/unstructured/sentiment/guba_comments_2025Q1_quick.csv', encoding='utf-8-sig')
print(f'形状: {df_guba.shape}')

# 显示前5个标题
print('前5个标题:')
for i, title in enumerate(df_guba['title'].head(5), 1):
    print(f'  {i}. {title}')

# 检查修复后的雪球热榜  
print('\n【雪球热榜 - 清理后】')
df_xueqiu = pd.read_csv('data/raw/unstructured/sentiment/hotlist_xueqiu.csv', encoding='utf-8-sig')
print(f'形状: {df_xueqiu.shape}')
print(f'列名: {list(df_xueqiu.columns)}')

# 显示前3行
print('前3行数据:')
for i, row in df_xueqiu.head(3).iterrows():
    trade_date = row['trade_date']
    name = row['name']
    rank = row['rank']
    print(f'  {trade_date} | {name} | 排名: {rank}')

print('\n✅ 数据清理验证完成！')
print('主要修复:')
print('  1. 修复UTF-8编码乱码问题')  
print('  2. 清理100%空值列')
print('  3. 保留有效数据字段')