
# Step 1: 載入訓練集和測試集
import pandas as pd
from pycaret.classification import *

# 載入資料
train_df = pd.read_csv('train.csv')  # 替換為訓練集路徑
test_df = pd.read_csv('test.csv')    # 替換為測試集路徑

# Step 2: 資料前處理（訓練集和測試集）
# 處理訓練集
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna('S', inplace=True)
train_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'])

# 處理測試集
test_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
test_df['FareBand'] = pd.qcut(test_df['Fare'].fillna(train_df['Fare'].median()), 4, labels=['Low', 'Medium', 'High', 'Very High'])

# Step 3: 初始化 PyCaret 環境並訓練模型
clf1 = setup(data=train_df, target='Survived', session_id=123, categorical_features=['Sex', 'Embarked', 'FareBand'])
best_model = compare_models()

# Step 4: 使用最佳模型對測試集進行預測
predictions = predict_model(best_model, data=test_df)

# Step 5: 生成符合 Kaggle 提交格式的文件
submission = predictions[['PassengerId', 'prediction_label']]
submission.rename(columns={'prediction_label': 'Survived'}, inplace=True)

# 確保 Survived 是整數
submission['Survived'] = submission['Survived'].astype(int)

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print("提交檔案已保存為 submission.csv")

# 顯示前幾行結果
print(submission.head())
