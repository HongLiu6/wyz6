import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import ToTensor

# 下载数据集到本地
train_ds = torchvision.datasets.MNIST('data/', train=True, transform=ToTensor(), download=True)
test_ds = torchvision.datasets.MNIST('data/', train=False, transform=ToTensor(), download=True)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256)

vision = True
if vision:
    images, labels = (train_ds._load_data())
    for i, c in enumerate(np.random.randint(0, 1000, 16)):
        plt.subplot(4, 4, i+1)
        plt.tight_layout()
        plt.imshow(images[c], interpolation='none')
        plt.title("label:{}".format(labels[c]))
    plt.savefig('./image.jpg')


# imgs.shape
# torch.Size([64, 1, 28, 28])
# 第1维 是图片的数量，第2维是channel，第3维是高，第4维是宽。 Minis数据集是黑白图片，因此channel为1

# 创建模型
class Model(nn.Module):  # 所有的pytorch都继承至nn.Module
    def __init__(self):  # 初始化3个层
        super().__init__()  # 继承父类属性   # nn.Linear, 全连接层，输入数据要求是一维的.（batch,features） 无论多少输入，后面都展平成一维 features
        self.liner_1 = nn.Linear(28 * 28, 120)  # 第一层输入28*28（1*28*28 展平后的长度 28*28）， 输出120个单元
        self.liner_2 = nn.Linear(120, 84)  # 第二层输出84个单元，输入是上一层的输出神经元个数。 这里的120/84都是自己选的，超参数
        self.liner_3 = nn.Linear(84, 10)  # 第三层输出10个分类。 softmax 模型输出C个可能值上的概率。C表示类别总数

    def forward(self, input):  # 具体实现，用forward方法
        x = input.view(-1, 28 * 28)  # 用view方法，将输入展平为-1，28*28. 第一维-1 为batch
        x = F.relu(self.liner_1(x))  # 每个中间层都要进行激活
        x = F.relu(self.liner_2(x))
        x = self.liner_3(x)  # 此处的x应该叫logits，未激活前的输出。 softmax归一化处理后，才输出概率值。 如果没有用softmax，则用argmax也可以取出概率分布最大的值。
        return x


device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
# print("Using {} device".format(device))
model = Model().to(device)
loss_fn = torch.nn.CrossEntropyLoss()  # 损失函数，输入应该是未经激活的输出。输出targets就是分类索引，而非独热编码方式。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # 优化：根据计算得到的损失，调整模型参数，从而降低损失的过程


# train函数
def train(dataloader, model, loss_fn, optimizer):
    size = len(
        dataloader.dataset)  # 获取当前数据集的总样本数。 dataloader.dataset 获取转换为dataloader之前的dataset。num_batches = len(dl) 返回迭代的批次数
    train_loss, correct = 0, 0  # train_loss会累计所有批次的损失之和； correct 累计预测正确的样本数
    for X, y in dataloader:  # X代表输入，y代表target标签
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)  # 返回一个批次所有样本的平均损失

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # argmax(0)是batch 位。argmax(1)是实际预测的值
            train_loss += loss.item()
    train_loss /= size  # 每个样本的平均loss
    correct /= size  # 正确率
    return train_loss, correct


# 测试函数
def test(dataloader, model):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size

    return test_loss, correct


epochs = 50

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    epoch_loss, epoch_acc = train(train_dl, model, loss_fn, optimizer)
    epoch_test_loss, epoch_test_acc = test(test_dl, model)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

    template = ("epoch:{:2d}, train_loss: {:.5f}, train_acc: {:.1f}% , test_loss: {:.5f}, test_acc: {:.1f}%")
    print(template.format(epoch, epoch_loss, epoch_acc * 100, epoch_test_loss, epoch_test_acc * 100))

print("Done!")
