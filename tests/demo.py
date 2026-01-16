import tushare as ts

# pro = ts.pro_api()
# 或者
pro = ts.pro_api('a80ae2f6ce3e25bc96b37c998804ef04bac9870a185b3d62bb71cadd')

df = pro.bo_daily(date='20241014')

print(df)