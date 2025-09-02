import pandas as pd
import numpy as np
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 读取数据
df = pd.read_csv('./data/matlab/split_datav3/test_matlab_TZ211230-220105_10min.csv')

# 删除时间列用于数据处理
# df.drop('timestamp', axis=1, inplace=True)
data = df.values
datax = data[:, :-1]
datay = data[:, -1].reshape(-1, 1)


# 数据预处理
scalerx = StandardScaler()
datax_normalized = scalerx.fit_transform(datax)

scalery = StandardScaler()
datay_normalized = scalery.fit_transform(datay)


def create_sequences(datax, datay, sequence_length, pred_length):
    x, y = [], []
    for i in range((len(datax) - sequence_length)//sequence_length):
        x.append(datax[i*sequence_length:(i*sequence_length + sequence_length), :])
        y.append(datay[(i*sequence_length + sequence_length):(i*sequence_length + sequence_length + pred_length)])
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    y = y.reshape(y.shape[0], pred_length)
    return x, y

class MyDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y) if y is not None else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.y is not None:
            return self.x[index], self.y[index]
        return self.x[index]

sequence_length = 12
pred_length = 12
batch_size = 32

# 创建数据集和数据加载器
x, y = create_sequences(datax_normalized, datay_normalized, sequence_length, pred_length)

dataset = MyDataset(x, y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)



class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # 定义 gru 层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # 定义输出层
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播 GRU
        out, _ = self.gru(x, h0)

        # 取 GRU 输出的最后一步
        out = self.linear(out[:, -1, :])

        # 调整输出形状为 (batch_size, sequence_length, output_dim)
        return out.view(x.size(0), -1, self.output_size)


model = GRU(input_size=4, hidden_size=128, num_layers=2, output_size=12,).to(device)
model.load_state_dict(torch.load('bestmodel/matlab_split6&4/trainset_CQ0804-0810/gru_10min_history_forward_model_best_v3_CQ08.pth'))
model.to(device)

# 训练开始前的时间
start_time = time.time()

# 评估模型
model.eval()
all_predictions = []
all_labels = []
losses = []

criterion = nn.MSELoss()
print('x：', x.shape)
with torch.no_grad():
    predictions = model(torch.Tensor(x)).view(-1,1)
    all_predictions = np.array(predictions)
print(all_predictions.shape)

# 训练和评估完成后的时间
end_time = time.time()
# 计算并显示所花费的总时间
total_time = end_time - start_time
print(f'Total computation time: {total_time:.2f} seconds')


# 反归一化
all_predictions = scalery.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
all_labels = scalery.inverse_transform(y.reshape(-1, 1)).flatten()
print(all_predictions.shape)
print(all_labels.shape)

# 保存结果到CSV文件
results_df = pd.DataFrame({
    'Actual': all_labels,
    'Prediction': all_predictions
})
# results_df.to_csv('./results/matlab_split6&4/trainset_TZ/gru_CQ07.csv', index=False)

# 可视化结果
plt.figure(figsize=(10, 5))
plt.plot(all_labels, label='Actual Data', color='red')
plt.plot(all_predictions, label='GRU Predicted Data', color='blue')
plt.title('Comparison of Actual and GRU Prediction')
plt.xlabel('Index')
plt.ylabel('Data Value')
plt.legend()
plt.tight_layout()
# plt.savefig('testGRU_forward_history_comparison_actual_predicted.png')
plt.show()

# 计算评估指标
mae = mean_absolute_error(all_labels, all_predictions)
mape = np.mean(np.abs((all_labels - all_predictions) / all_labels)) * 100
rmse = np.sqrt(mean_squared_error(all_labels, all_predictions))
r2 = r2_score(all_labels, all_predictions)

# print(f'Loss: {loss.item():.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.4f}%')
print(f'RMSE: {rmse:.4f}')
print(f'R2: {r2:.4f}')


# with torch.no_grad():
#     for seq_inputs, seq_labels in data_loader:
#         seq_inputs = seq_inputs.to(device)
#         seq_labels = seq_labels.to(device)
#         outputs = model(seq_inputs)
#         loss = criterion(outputs, seq_labels)
#         losses.append(loss.item())
#
#         all_predictions.extend(outputs.cpu().numpy())
#         all_labels.extend(seq_labels.cpu().numpy())
#
# # 确保所有数组的长度一致
# min_length = min(len(all_predictions), len(all_labels))
# all_predictions = np.array(all_predictions[:min_length]).flatten()
# all_labels = np.array(all_labels[:min_length]).flatten()
# print(len(all_predictions))
# print(len(all_labels))
# # 反归一化
# all_predictions = scalery.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
# all_labels = scalery.inverse_transform(all_labels.reshape(-1, 1)).flatten()

# # 将预测结果保存到 CSV 文件中
# results_df = pd.DataFrame({
#     'Actual': all_labels,
#     'Prediction': all_predictions
# })
# # results_df.to_csv('evalRNN_forward_TZ.csv', index=False)
#
# # 计算评估指标
# mae = mean_absolute_error(all_labels, all_predictions)
# mape = np.mean(np.abs((all_labels - all_predictions) / all_labels)) * 100
# rmse = np.sqrt(mean_squared_error(all_labels, all_predictions))
# r2 = r2_score(all_labels, all_predictions)
#
# print(f'MAE: {mae:.4f}')
# print(f'MAPE: {mape:.4f}%')
# print(f'RMSE: {rmse:.4f}')
# print(f'R2: {r2:.4f}')
#
# # 可视化结果
# plt.figure(figsize=(10, 5))
# plt.plot(range(len(all_labels)), all_labels, label='Actual Data', color='red', linewidth=0.5)
# plt.plot(range(len(all_predictions)), all_predictions, label='RNN Predicted Data', color='blue', linewidth=0.5)
# # plt.plot(all_labels, label='Actual Data', color='red',linewidth=0.5)
# # plt.plot(all_predictions, label='RNN Predicted Data', color='blue',linewidth=0.5)
# plt.title('Comparison of Actual and RNN Predicted Data')
# plt.xlabel('Index')
# plt.ylabel('Data Value')
# plt.legend()
# plt.tight_layout()
# # plt.savefig('evalRNN_forward_TZ.png')
# plt.show()
#
# # # 绘制损失图
# # plt.figure(figsize=(10, 5))
# # plt.plot(range(len(losses)), losses, label='Loss')
# # plt.xlabel('Batch')
# # plt.ylabel('Loss')
# # plt.title('RNN_Loss over Batches')
# # plt.legend()
# # plt.savefig('evalRNN_forward_loss_over_batches.png')
# # plt.show()
#
# # 打印平均损失
# print(f'Average Loss: {np.mean(loss):.4f}')



# import pandas as pd
# import numpy as np
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import matplotlib.pyplot as plt
#
# # 读取数据
# df = pd.read_csv('./data/10min/oneweek_history_noise/CQ230804-230810_v1.csv')
#
# # 删除时间列用于数据处理
# df.drop('timestamp', axis=1, inplace=True)
# data = df.values
# X = data[:, :-1]
# Y = data[:, -1].reshape(-1, 1)
#
# # 数据构造函数
# def create_sequences(datax, datay, sequence_length, pred_length):
#     x, y = [], []
#     for i in range((len(datax) - sequence_length) // sequence_length):
#         x.append(datax[i*sequence_length:(i*sequence_length + sequence_length), :])
#         y.append(datay[(i*sequence_length + sequence_length):(i*sequence_length + sequence_length + pred_length)])
#     x = np.array(x, dtype=np.float32)
#     y = np.array(y, dtype=np.float32)
#     y = y.reshape(y.shape[0], pred_length)
#     return x, y
#
# sequence_length = 12
# pred_length = 12
#
# X_seq, Y_seq = create_sequences(X, Y, sequence_length, pred_length)
# print('X:',X_seq.shape)
# print('Y:',Y_seq.shape)
# # 数据预处理
# scaler_x = StandardScaler()
# scaler_y = StandardScaler()
# X_scaled = scaler_x.fit_transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
# Y_scaled = scaler_y.fit_transform(Y_seq)
# print('X_scaled:',X_scaled.shape)
# print('Y_scaled:',Y_scaled.shape)
#
# # PLS 回归模型
# pls = PLSRegression(n_components=2)
#
# # 拟合模型
# n_samples, n_timesteps, n_features = X_scaled.shape
# X_scaled_reshaped = X_scaled.reshape(n_samples, -1)  # 展平时间步和特征维度
# print('X_scaled_reshaped：', X_scaled_reshaped.shape)
# pls.fit(X_scaled_reshaped, Y_scaled)
#
# # 拟合结果
# Y_pred_scaled = pls.predict(X_scaled_reshaped)
# Y_pred = scaler_y.inverse_transform(Y_pred_scaled)
#
# # 计算评价指标
# mse = mean_squared_error(Y_seq, Y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(Y_seq, Y_pred)
# r2 = r2_score(Y_seq, Y_pred)
#
# print(f'PLS MSE: {mse:.4f}')
# print(f'PLS RMSE: {rmse:.4f}')
# print(f'PLS MAE: {mae:.4f}')
# print(f'PLS R²: {r2:.4f}')
#
# # 绘制拟合效果图
# plt.figure(figsize=(10, 5))
# plt.plot(Y_seq.flatten(), label='Actual Data', color='red', linewidth=0.5)
# plt.plot(Y_pred.flatten(), label='PLS Predicted Data', color='blue', linewidth=0.5)
# plt.title('Comparison of Actual and PLS Predicted Data')
# plt.xlabel('Index')
# plt.ylabel('Data Value')
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# # 未来预测
# future_steps = 12
# last_known_data = X[-sequence_length:]
#
# last_known_data_scaled = scaler_x.transform(last_known_data.reshape(1, -1))
# future_pred_scaled = pls.predict(last_known_data_scaled)
# future_pred = scaler_y.inverse_transform(future_pred_scaled)
# predictions = future_pred.flatten()
#
# # 打印预测结果
# print("未来预测值：", predictions)
