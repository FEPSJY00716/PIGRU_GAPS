# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#
# # 设置设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# # 读取数据
# df = pd.read_csv('./data/matlab/split_datav3/train_matlab_TZ211230-220105_10min.csv')
#
# # 删除时间列用于数据处理
# # df.drop('timestamp', axis=1, inplace=True)
# data = df.values
# datax = data[:, :-1]
# datay = data[:, -1].reshape(-1, 1)
# print('datax', datax.shape)
# print('datay', datay.shape)
#
# # 数据预处理
# scalerx = StandardScaler()
# datax_normalized = scalerx.fit_transform(datax)
#
# scalery = StandardScaler()
# datay_normalized = scalery.fit_transform(datay)
#
#
# def create_sequences(datax, datay, sequence_length, pred_length):
#     x, y = [], []
#     for i in range(len(datax) - sequence_length - pred_length+ 1):
#         x_seq = datax[i:i + sequence_length]
#         y_seq = datay[i + sequence_length:i + sequence_length + pred_length]
#         x.append(x_seq)
#         y.append(y_seq)
#     x = np.array(x, dtype=np.float32)
#     y = np.array(y, dtype=np.float32)
#     return x, y
#
#
# class MyDataset(Dataset):
#     def __init__(self, x, y=None):
#         self.x = torch.from_numpy(x)
#         self.y = torch.from_numpy(y) if y is not None else None
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, index):
#         if self.y is not None:
#             return self.x[index], self.y[index]
#         return self.x[index]
#
# sequence_length = 12
# pred_length = 12
# batch_size = 32
#
# # 创建数据集和数据加载器
# x, y = create_sequences(datax_normalized, datay_normalized, sequence_length, pred_length)
#
# train_size = int(len(x) * 0.6)
# test_size = len(x) - train_size
#
# trainx, testx = x[:train_size], x[train_size:]
# trainy, testy = y[:train_size], y[train_size:]
# print("trainx:", trainx.shape)
# print("trainy:", trainy.shape)
# print('testx:', testx.shape)
# print('testy:', testy.shape)
# train_dataset = MyDataset(trainx, trainy)
# test_dataset = MyDataset(testx, testy)
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
#
#
# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.output_size = output_size
#
#         # 定义 LSTM 层
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#
#         # 定义输出层
#         self.linear = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         # 初始化隐藏状态和细胞状态
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#
#         # 前向传播 LSTM
#         out, _ = self.lstm(x, (h0, c0))
#
#         # 取 LSTM 输出的最后一步
#         out = self.linear(out[:, -1, :])
#
#         # 输出形状调整为 (batch_size, output_dim)
#         return out
#
#
#
# # 初始化模型和其他参数
# model = LSTM(input_size=4, hidden_size=128, num_layers=2, output_size=12,).to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.MSELoss(reduction="mean")
# num_epochs = 300
#
# # 训练模型
# epoch_losses = []
# best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大
#
# for epoch in range(num_epochs):
#     model.train()
#     batch_losses = []
#     for train_x, train_y in train_loader:
#         train_x, train_y = train_x.to(device), train_y.to(device)
#         optimizer.zero_grad()
#         y_pred = model(train_x)
#         data_loss = criterion(y_pred, train_y)
#         data_loss.backward()
#         optimizer.step()
#         batch_losses.append(data_loss.item())
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses.append(epoch_loss)
#     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
#
#     # 验证过程
#     model.eval()
#     val_losses = []
#     with torch.no_grad():
#         for testx, testy in test_loader:
#             testx, testy = testx.to(device), testy.to(device)
#             val_pred = model(testx)
#             val_loss = criterion(val_pred, testy)
#             val_losses.append(val_loss.item())
#
#     # 计算验证损失
#     mean_val_loss = np.mean(val_losses)
#     print(f'Validation Loss: {mean_val_loss:.4f}')
#
#     # 如果这一轮的验证损失是到目前为止最低的，则保存模型
#     if mean_val_loss < best_val_loss:
#         best_val_loss = mean_val_loss
#         torch.save(model.state_dict(), 'bestmodel/matlab_split6&4/trainset_TZ1230-0105/lstm_10min_history_forward_model_best_v3_TZ.pth')
#         print("Best model updated and saved.")
#
#
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 读取数据
df = pd.read_csv('./data/matlab/split_datav3/train_matlab_CQ230804-0810_10min.csv')

# 删除时间列用于数据处理
# df.drop('timestamp', axis=1, inplace=True)
data = df.values
datax = data[:, :-1]
datay = data[:, -1].reshape(-1, 1)
print('datax', datax.shape)
print('datay', datay.shape)

# 数据预处理
scalerx = StandardScaler()
datax_normalized = scalerx.fit_transform(datax)

scalery = StandardScaler()
datay_normalized = scalery.fit_transform(datay)

def create_sequences(datax, datay, sequence_length, pred_length):
    x, y = [], []
    for i in range(len(datax) - sequence_length - pred_length + 1):
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

train_size = int(len(x) * 0.6)
test_size = len(x) - train_size

trainx, testx = x[:train_size], x[train_size:]
trainy, testy = y[:train_size], y[train_size:]
print("trainx:", trainx.shape)
print("trainy:", trainy.shape)
print('testx:', testx.shape)
print('testy:', testy.shape)
train_dataset = MyDataset(trainx, trainy)
test_dataset = MyDataset(testx, testy)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 定义输出层
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))

        # 取 LSTM 输出的最后一步
        out = self.linear(out[:, -1, :])

        # 输出形状调整为 (batch_size, output_dim)
        # return out
        return out.view(x.size(0), -1, self.output_size)


# 初始化模型和其他参数
model = LSTM(input_size=4, hidden_size=128, num_layers=2, output_size=12).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss(reduction="mean")
num_epochs = 300

# 记录训练时间和损失
start_time = time.time()
epoch_losses = []
best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大

for epoch in range(num_epochs):
    model.train()
    batch_losses = []
    for train_x, train_y in train_loader:
        train_x, train_y = train_x.to(device), train_y.to(device)
        optimizer.zero_grad()
        y_pred = model(train_x)
        data_loss = criterion(y_pred, train_y)
        data_loss.backward()
        optimizer.step()
        batch_losses.append(data_loss.item())
    epoch_loss = np.mean(batch_losses)
    epoch_losses.append(epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # 验证过程
    model.eval()
    val_losses = []
    with torch.no_grad():
        for testx, testy in test_loader:
            testx, testy = testx.to(device), testy.to(device)
            val_pred = model(testx)
            val_loss = criterion(val_pred, testy)
            val_losses.append(val_loss.item())

    # 计算验证损失
    mean_val_loss = np.mean(val_losses)
    print(f'Validation Loss: {mean_val_loss:.4f}')

    # 如果这一轮的验证损失是到目前为止最低的，则保存模型
    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        torch.save(model.state_dict(), 'bestmodel/matlab_split6&4/trainset_CQ0804-0810/lstm_10min_history_forward_model_best_v3_CQ08.pth')
        print("Best model updated and saved.")

end_time = time.time()
training_time = end_time - start_time
print(f'Training time: {training_time:.2f} seconds')

# 保存损失记录到文件
losses_df = pd.DataFrame({'epoch': range(1, num_epochs+1), 'loss': epoch_losses})
losses_df.to_csv('results/matlab_split6&4/trainset_CQ08/lstm_training_losses.csv', index=False)

# 绘制训练损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses_df['epoch'], losses_df['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_curve.png')
plt.show()
