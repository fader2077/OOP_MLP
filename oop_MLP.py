import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix#加入confussion matrix


# 總訓練次數
EPOCHS = 30 
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
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
train_data_loader = data_loader(X_train, y_train)
train_data_loader = torch.utils.data.DataLoader(train_data_loader, batch_size=8, shuffle=True)
test_data_loader = data_loader(X_test, y_test)
test_data_loader = torch.utils.data.DataLoader(test_data_loader, batch_size=8, shuffle=True)
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target.argmax(1))
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
def test(model,device,optimizer,epoch,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target.argmax(1), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.argmax(1, keepdim=True).view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

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
#輸出

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