import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 读取数据
df = pd.read_csv('./data/matlab/split_datav3/test_matlab_CQ230713-0720_10min.csv')

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
#

def create_sequences(datax, datay, sequence_length, pred_length):
    x, y = [], []
    for i in range(len(datax) - sequence_length - pred_length+ 1):
        x_seq = datax[i:i + sequence_length]
        y_seq = datay[i + sequence_length:i + sequence_length + pred_length]
        x.append(x_seq)
        y.append(y_seq)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
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
print('y:', y.shape)
dataset = MyDataset(x, y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # 定义 LSTM 层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # 定义输出层
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播 RNN，注意这里只需要 h0
        out, _ = self.rnn(x, h0)

        # 取 RNN 输出的最后一步
        out = self.linear(out[:, -1, :])

        # 调整输出形状为 (batch_size, sequence_length, output_dim)
        return out.view(x.size(0), -1, self.output_size)

model = RNN(input_size=4, hidden_size=128, num_layers=1, output_size=12).to(device)
model.load_state_dict(torch.load('bestmodel/matlab_split6&4/trainset_CQ0713-0720/rnn_10min_history_forward_model_best_v3_CQ07.pth'))
model.to(device)

# 训练开始前的时间
start_time = time.time()

# 评估模型
model.eval()
all_predictions = []
all_labels = []
losses = []

criterion = nn.MSELoss()


with torch.no_grad():
    # predictions = model(torch.Tensor(x)).view(-1,1)
    predictions = model(torch.Tensor(x))
    all_predictions = np.array(predictions)

# 训练和评估完成后的时间
end_time = time.time()
# 计算并显示所花费的总时间
total_time = end_time - start_time
print(f'Total computation time: {total_time:.2f} seconds')


print('all_predictions:', all_predictions.shape)
print('all_labels:', y.shape)

# 对于 all_predictions 形状为 (973, 1, 12)，选择每组中的第一个值
all_predictions = all_predictions[:, 0, 0].reshape(-1, 1)  # 选择第一个预测值，重新形状为 (973, 1)
# 对于 all_labels 形状为 (973, 12, 1)，选择每组中的第一个值
all_labels = y[:, 0, 0].reshape(-1, 1)  # 选择第一个标签值，重新形状为 (973, 1)

# 反归一化
# all_predictions = scalery.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
all_predictions = scalery.inverse_transform(all_predictions)
# all_labels = scalery.inverse_transform(y.reshape(-1, 1)).flatten()
all_labels = scalery.inverse_transform(all_labels)
print('all_predictions:', all_predictions.shape)
print('all_labels:', all_labels.shape)

all_predictions = np.squeeze(all_predictions, axis=1)
all_labels = np.squeeze(all_labels, axis=1)
# 保存结果到CSV文件
results_df = pd.DataFrame({
    'Actual': all_labels,
    'Prediction': all_predictions
})
# results_df.to_csv('./results/matlab_split6&4/trainset_TZ/rnn_CQ08.csv', index=False)

# 可视化结果
plt.figure(figsize=(10, 5))
plt.plot(all_labels, label='Actual Data', color='red')
plt.plot(all_predictions, label='RNN Predicted Data', color='blue')
plt.title('Comparison of Actual and RNN Prediction')
plt.xlabel('Index')
plt.ylabel('Data Value')
plt.legend()
plt.tight_layout()
# plt.savefig('testRNN_forward_history_comparison_actual_predicted.png')
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


