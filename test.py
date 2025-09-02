import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from normalize import TrainData, SlidingWindowDataGenerator
import matplotlib.pyplot as plt
import sys
from model import GRU
import time


torch.set_printoptions(threshold=sys.maxsize)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mm = TrainData()

tr_data_file = r'./data/train_TZ211230-220105_10min.csv'
te_data_file = r'./data/test_CQ230713-0720_10min.csv'
te_data_file2 = r'./data/test_CQ230804-0810_10min.csv'
te_data_file3 = r'./data/test_TZ211230-220105_10min.csv'


tr_data, te_data, te_data2, te_data3 = np.array(pd.read_csv(tr_data_file, index_col=None,)), np.array(pd.read_csv(te_data_file, index_col=None,)), np.array(pd.read_csv(te_data_file2, index_col=None,)), np.array(pd.read_csv(te_data_file3, index_col=None,))
tr_data, te_data, te_data2, te_data3 = torch.tensor(tr_data, dtype=torch.float), torch.tensor(te_data, dtype=torch.float), torch.tensor(te_data2, dtype=torch.float), torch.tensor(te_data3, dtype=torch.float)
y,yte, yte2, yte3 = tr_data[:, 4:], te_data[:, 7].unsqueeze(1), te_data2[:, 7].unsqueeze(1), te_data3[:, 7].unsqueeze(1)
max,_ = torch.max(y, axis=0)
min,_ = torch.min(y, axis=0)

tr_data = torch.cat((mm.train_data(tr_data)[:, :4], y,), 1)
te_data = torch.cat((mm.test_data(te_data)[:, :4], yte,), 1).reshape(32,12,5)
te_data2 = torch.cat((mm.test_data(te_data2)[:, :4], yte2,), 1).reshape(32,12,5)
teoneday1 = torch.cat((mm.test_data(te_data3)[:, :4], yte3,), 1).reshape(32,12,5)

# 初始化模型、损失函数和优化器
model = torch.load('./best_model/train_TZ211230-220105_10min.pth', map_location=torch.device('cpu'))

criterion = nn.MSELoss()

start_time = time.time()
_, _, Tair_future = model(te_data[:, :, :4].clone().detach().to(device), max, min)
end_time = time.time()
label = te_data[:, :, 4].to(device)
losste = criterion(Tair_future, label)
print("CQ07 Loss:", losste)
print("CQ07 Time:", end_time - start_time, "seconds")

start_time = time.time()
_, _, Tair_future2 = model(te_data2[:, :, :4].clone().detach().to(device), max, min)
end_time = time.time()
label2 = te_data2[:, :, 4].to(device)
losste2 = criterion(Tair_future2, label2)
print("CQ08 Loss:", losste)
print("CQ08 Time:", end_time - start_time, "seconds")

start_time = time.time()
_, _, Tair_future3 = model(teoneday1[:, :, :4].clone().detach().to(device), max, min)
end_time = time.time()
label3 = teoneday1[:, :, 4].to(device)
losste3 = criterion(Tair_future3, label3)
print("TZ Loss:", losste)
print("TZ Time:", end_time - start_time, "seconds")


label = label.cpu().detach().numpy().reshape(32*12, 1)
output = Tair_future.cpu().detach().numpy().reshape(32*12, 1)
label2 = label2.cpu().detach().numpy().reshape(32*12, 1)
output2 = Tair_future2.cpu().detach().numpy().reshape(32*12, 1)
label3 = label3.cpu().detach().numpy().reshape(32*12, 1)
output3 = Tair_future3.cpu().detach().numpy().reshape(32*12, 1)


# df = pd.DataFrame({'label': label.flatten(), 'output': output.flatten()})
# df.to_csv('CQ07.csv', index=False)
# df = pd.DataFrame({'label': label2.flatten(), 'output': output2.flatten()})
# df.to_csv('CQ08.csv', index=False)
# df = pd.DataFrame({'label': label3.flatten(), 'output': output3.flatten()})
# df.to_csv('TZ.csv', index=False)

x = np.arange(32*12)
plt.figure(figsize=(10, 6))
plt.plot(x, output, label='output')
plt.plot(x, label, label='label')
# plt.plot(x, y2, label='y2')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Regression Results')
plt.legend()
plt.show()

x = np.arange(32*12)
plt.figure(figsize=(10, 6))
plt.plot(x, output2, label='output2')
plt.plot(x, label2, label='label2')
# plt.plot(x, y2, label='y2')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Regression Results')
plt.legend()
plt.show()

x = np.arange(32*12)
plt.figure(figsize=(10, 6))
plt.plot(x, output3, label='output3')
plt.plot(x, label3, label='label3')
# plt.plot(x, y2, label='y2')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Regression Results')
plt.legend()
plt.show()
