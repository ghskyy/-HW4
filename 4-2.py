
# Step 1: 載入資料
import pandas as pd
from pycaret.classification import *

# 載入訓練集和測試集
train_df = pd.read_csv('train.csv')  # 替換為你的訓練集實際路徑
test_df = pd.read_csv('test.csv')    # 替換為你的測試集實際路徑

# Step 2: 資料前處理
# 處理訓練數據
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna('S', inplace=True)
train_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'])

# 處理測試數據
test_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
test_df['FareBand'] = pd.qcut(test_df['Fare'].fillna(train_df['Fare'].median()), 4, labels=['Low', 'Medium', 'High', 'Very High'])

# Step 3: 初始化 PyCaret 環境並訓練模型
clf1 = setup(
    data=train_df,
    target='Survived',
    session_id=123,
    categorical_features=['Sex', 'Embarked', 'FareBand']
)

# 自動比較模型，選出前 3 名
best_models = compare_models(n_select=3)

# 使用最佳模型進行集成（集成模型）
ensemble_model = blend_models(best_models)

# 調整最佳模型的超參數
tuned_model = tune_model(best_models[0])  # 針對第一名模型調參

# Step 4: 預測測試集
predictions = predict_model(tuned_model, data=test_df)

# 整理提交格式
submission = predictions[['PassengerId', 'prediction_label']]
submission.rename(columns={'prediction_label': 'Survived'}, inplace=True)
submission['Survived'] = submission['Survived'].astype(int)

# Step 5: 保存模型與提交文件
# 保存最佳模型
save_model(tuned_model, 'optimized_titanic_model')

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print("提交檔案已保存為 submission.csv")

# 顯示提交檔案的前幾行
print(submission.head())
