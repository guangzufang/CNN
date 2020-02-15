import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def plot_images_labels_prediction(images,  # 图像列表
                                  labels,  # 标签列表
                                  prediction,  # 预测值列表
                                  index,  # 从第INDEX个开始显示
                                  num=10):  # 缺省一次显示10幅
    fig = plt.gcf()  # 获取当前图标
    fig.set_size_inches(10, 12)  # 1英寸等于2.54cm
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(np.reshape(images[index], (28, 28)), cmap='binary')
        title = "label=" + str(labels[i])
        if len(prediction) > 0:
            title += ",predict=" + str(prediction[index])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        index += 1
    plt.show()
#搭建网络模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(4*4*50, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def main():
    use_cuda =  torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 1000
    EPOCH = 20

    train_data = torchvision.datasets.FashionMNIST(root='./data', train=True,
                      download=False, transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.FashionMNIST(root='./data', train=False,
                      download=False, transform=torchvision.transforms.ToTensor())

    train_loader = DataLoader(train_data,
                      batch_size=TRAIN_BATCH_SIZE,  # mini batch size
                      shuffle=True,  # random shuffle for training
                         **kwargs
                      )
    test_loader = DataLoader(test_data,
                      batch_size=TEST_BATCH_SIZE,  # mini batch size
                      shuffle=True,  # random shuffle for training
                      **kwargs
                      )

    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(1, EPOCH+1):
        model.train()
        train_accracy = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predicton = model(batch_x)
            loss = loss_func(predicton, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = predicton.argmax(dim=1, keepdim=True)
            train_accracy += pred.eq(batch_y.view_as(pred)).sum().item()
            print('Train_epoch:{}({:.0f})%| step:{}| loss:{:.6f}'
                  .format(epoch, 100. * step / (len(train_data)/TRAIN_BATCH_SIZE)
                  ,step,loss.item()))
        train_accracy /= len(train_data)
        print('Average Train Accracy:{:.3f}%'
              .format(100. * train_accracy))

        #testing
        model.eval()
        test_loss = 0
        accracy = 0
        with torch.no_grad():
            for (test_x, target) in (test_loader):
                test_x, target = test_x.to(device), target.to(device)
                output = model(test_x)
                test_loss += loss_func(output, target).item()
                test_pred = output.argmax(dim=1,keepdim=True)
                accracy += test_pred.eq(target.view_as(test_pred)).sum().item()

        test_loss /= (len(test_data)/TEST_BATCH_SIZE)
        accracy /= len(test_data)
        print('Average Loss:{:.6f} | Average Accracy:{:.3f}%'
               .format(test_loss,100. * accracy))

    torch.save(model, 'net.pkl')  # save entire net
    torch.save(model.state_dict(), 'net_params.pkl')  # save only the parameters

    # print 10 predictions from test data
    test_x = (torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255.).to(device)
    test_y = test_data.targets.numpy()
    test_output = model(test_x[:10])
    pred_y = test_output.argmax(dim=1,keepdim=True).cpu()
    plot_images_labels_prediction(test_data.data[:10],
                                  test_y[:10],
                                  pred_y.numpy(),0,10)
if __name__ == '__main__':
    main()
