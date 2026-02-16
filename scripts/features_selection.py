import pickle
import json
from pathlib import Path

# 加载 LightGBM 模型
with open('models/lgbm/lgbm_excess_ret_5d.pkl', 'rb') as f:
    model = pickle.load(f)

# 获取特征重要性 (使用 gain，与训练脚本一致)
importance = model.feature_importance(importance_type='gain')
feature_names = model.feature_name()

# 排序获取 Top 50
sorted_idx = importance.argsort()[::-1]
top50_features = [feature_names[i] for i in sorted_idx[:50]]

# 排除类别特征（纯数值版）
categorical = []
numeric_features = [f for f in top50_features if f not in categorical][:50]

# 保存
output = {'features': numeric_features, 'count': len(numeric_features)}
with open('models/lgbm/top50_features.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f'已保存 {len(numeric_features)} 个纯数值特征到 models/lgbm/top50_features.json')
print('Top 15 特征:', numeric_features[:15])