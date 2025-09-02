# 计算单列
# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#
# # 读取 CSV 文件
# df = pd.read_csv('./result/benchmark_pigruTZ_TZ.csv')
#
# # 提取标签和输出列
# labels = df['label'].values
# outputs = df['output'].values
#
# # 计算评价指标
# # 计算均方误差
# mse = mean_squared_error(labels, outputs)
#
# mae = mean_absolute_error(labels, outputs)
# mape = np.mean(np.abs((labels - outputs) / labels)) * 100
# rmse = np.sqrt(mse)
# r2 = r2_score(labels, outputs)
#
# # 打印评价指标
# print(f'MAE: {mae:.4f}')
# print(f'MAPE: {mape:.4f}%')
# print(f'RMSE: {rmse:.4f}')
# print(f'R²: {r2:.4f}')


# 计算多列
import pandas as pd
import numpy as np

# 读取 CSV 文件
df = pd.read_csv('./result/benchmark_all_TZ.csv')

# 提取 label 列
label = df['label']

# 初始化字典来存储 MAE 和 RMSE 结果
mae_results = {}
rmse_results = {}

# 遍历每一列计算 MAE 和 RMSE，忽略 label 列本身
for column in df.columns:
    if column != 'label':
        # 计算 MAE
        mae = np.mean(np.abs(df[column] - label))
        mae_results[column] = mae

        # 计算 RMSE
        rmse = np.sqrt(np.mean((df[column] - label) ** 2))
        rmse_results[column] = rmse

# 输出结果
print("Mean Absolute Error (MAE):")
for column, mae in mae_results.items():
    print(f"{column}: {mae}")

print("\nRoot Mean Squared Error (RMSE):")
for column, rmse in rmse_results.items():
    print(f"{column}: {rmse}")
