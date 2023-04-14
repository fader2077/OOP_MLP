import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix#加入confussion matrix
import seaborn as sns

# 總訓練次數
EPOCHS = 26 
# 設定訓練裝置，預設為GPU，沒有就用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# Load data
Train_data = pd.read_csv('BC_Train.csv', header=None)
Test_data = pd.read_csv('BC_Test.csv', header=None)
# show ground truth value counts
Train_data[6].value_counts()
mms = MinMaxScaler()
Train_data = mms.fit_transform(Train_data)
Test_data = mms.transform(Test_data)
X_train = Train_data[:, 0:6]
y_train = Train_data[:, 6]
X_test = Test_data[:, 0:6]
y_test = Test_data[:, 6]
#one-hot encoding
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()
y_train = torch.nn.functional.one_hot(y_train, 2)   
y_test = torch.nn.functional.one_hot(y_test, 2)
X_train = torch.from_numpy(X_train).float().to(DEVICE)
X_test = torch.from_numpy(X_test).float().to(DEVICE)
class data_loader(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)#建立全連接層，將輸入的特徵數量映射成64個元素
        self.fc2 = nn.Linear(64, 128)#將64個元素映射成128個元素
        self.fc3 = nn.Linear(128, 64)#將128個元素映射成64個元素
        self.fc4 = nn.Linear(64, 2)#將64個元素映射成2個元素
        self.dropout = nn.Dropout(0.2)#建立Dropout層，每次訓練隨機丟棄20%的神經元

    def forward(self, x):
        x = F.relu(self.fc1(x))#將輸入資料x經過第一層全連接層轉換成64個元素
        x = self.dropout(x)#對第一層全連接層的輸出進行Dropout
        x = F.relu(self.fc2(x))#將經過Dropout的輸出經過第二層全連接層轉換成128個元素
        x = self.dropout(x)#進行Dropout
        x = F.relu(self.fc3(x))# 將經過Dropout的輸出經過第三層全連接層轉換成64個元素
        x = self.fc4(x)#將經過第三層全連接層的輸出經過第四層全連接層轉換成2個元素
        return x
train_data_loader = data_loader(X_train, y_train)
train_data_loader = torch.utils.data.DataLoader(train_data_loader, batch_size=8, shuffle=True)
test_data_loader = data_loader(X_test, y_test)
test_data_loader = torch.utils.data.DataLoader(test_data_loader, batch_size=8, shuffle=True)
def train(model, device, train_loader, optimizer, epoch):
    model.train()#將模型設置為訓練模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)#將數據和標籤發送到指定的裝置上
        optimizer.zero_grad()#對優化器進行參數更新
        output = model(data)#通過模型進行前向傳播
        loss = F.cross_entropy(output, target.argmax(1))#計算輸出和標籤之間的交叉熵損失
        loss.backward()#計算梯度
        optimizer.step()#更新模型參數
        if batch_idx % 500 == 0:#每500次迭代輸出訓練狀態
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
def test(model,device,optimizer,epoch,test_loader):
    model.eval()    #將模型設置為驗證模式
    test_loss = 0  #初始化測試損失和正確預測數量
    correct = 0
    with torch.no_grad():    #設置 torch.no_grad()避免計算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)#將data和target發送到指定的裝置上
            output = model(data)#通過模型進行前向傳播
            test_loss += F.cross_entropy(output, target.argmax(1), reduction='sum').item() #計算輸出和目標之間的交叉損失
            pred = output.argmax(1, keepdim=True)#獲取最高概率預測類的索引
            correct += pred.eq(target.argmax(1, keepdim=True).view_as(pred)).sum().item()#與真實類比較並更新正確預測數
            
    test_loss /= len(test_loader.dataset)#計算平均測試損失
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, (100. * correct / len(test_loader.dataset))


model = MLP().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)#這是一個學習率調整器，每50個epoch調整一次學習率，乘上0.1


test_loss = []
test_acc = []
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_data_loader, optimizer, epoch)
    scheduler.step()
    loss, acc = test(model, DEVICE, optimizer, epoch, test_data_loader)
    test_loss.append(loss)
    test_acc.append(acc)
#confusion matrix

y_pred = model(X_test.to(DEVICE)).argmax(dim=1, keepdim=True)
y_true = y_test.argmax(1, keepdim=True).view_as(y_pred)
cm = confusion_matrix(y_true.cpu(), y_pred.cpu())
print('confusion matrix')
print(cm)
#輸出loss圖
plt.plot(test_loss)
plt.title('Test loss')
plt.legend(['loss'])
plt.show()
#輸出acc圖
plt.plot(test_acc)
plt.title('Test acc')
plt.legend(['acc'])
plt.show()
