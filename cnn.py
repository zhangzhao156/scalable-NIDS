import torch

# KDD 99
class CNN(torch.nn.Module):
    def __init__(self,N_class):
        super(CNN,self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.max_pool1=torch.nn.MaxPool1d(kernel_size=4,stride=2)
        self.conv2 = torch.nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.max_pool2 = torch.nn.MaxPool1d(kernel_size=4, stride=2)
        # KDD
        self.fn1 = torch.nn.Linear(256,128)
        self.fn2 = torch.nn.Linear(128,N_class)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU(inplace=True)
        # self.dropout = torch.nn.Dropout(0.3)
        # self.batchnorm = torch.nn.BatchNorm1d(128)
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x = x.view(x.size(0),-1)
        x = self.fn1(x)
        # x = self.batchnorm(x)
        x = self.relu(x)
        # x = self.dropout(x)
        av = self.fn2(x)
        x = self.sigmoid(av)
        return av,x



class SharedCNN(torch.nn.Module):
    def __init__(self,N_class):
        super(SharedCNN,self).__init__()
        self.sharedNet = cnn(N_class)

    def forward(self, indata, outdata):
        av_indata, pred_indata = self.sharedNet(indata)
        av_outdata, pred_outdata = self.sharedNet(outdata)
        return av_indata, pred_indata, av_outdata, pred_outdata


def cnn(N_class):
    model = CNN(N_class)
    return model




class CNN2(torch.nn.Module):
    def __init__(self,N_class):
        super(CNN2,self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.max_pool1=torch.nn.MaxPool1d(kernel_size=4,stride=2)
        self.conv2 = torch.nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.max_pool2 = torch.nn.MaxPool1d(kernel_size=4, stride=2)
        self.fn1 = torch.nn.Linear(256,128)
        self.fn2 = torch.nn.Linear(128,N_class)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x = x.view(x.size(0),-1)
        # print(x.size())
        x = self.fn1(x)
        x = self.relu(x)
        x = self.fn2(x)
        x = self.sigmoid(x)
        return x