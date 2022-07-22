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
        # UNSW
        # self.fn1 = torch.nn.Linear(288, 128)
        # # self.fn1 = torch.nn.Linear(224, 128)
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

# UNSW
# class CNN(torch.nn.Module):
#     def __init__(self,N_class):
#         super(CNN,self).__init__()
#         self.conv1 = torch.nn.Conv1d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=0)
#         self.max_pool1=torch.nn.MaxPool1d(kernel_size=2,stride=2)
#         # self.conv2 = torch.nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
#         # self.max_pool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
#         # UNSW
#         self.fn1 = torch.nn.Linear(1280,128)
#         # CICIDS2017
#         # self.fn1 = torch.nn.Linear(2432, 128)
#         self.fn2 = torch.nn.Linear(128, N_class)
#         # self.fn2 = torch.nn.Linear(1280,N_class)
#         self.sigmoid = torch.nn.Sigmoid()
#         self.relu = torch.nn.ReLU(inplace=True)
#         self.dropout = torch.nn.Dropout(0.5)
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.max_pool1(x)
#         # x = self.conv2(x)
#         # x = self.relu(x)
#         # x = self.max_pool2(x)
#         x = x.view(x.size(0),-1)
#
#         # print(x.size())
#         x = self.fn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         av = self.fn2(x)
#         x = self.sigmoid(av)
#         return av,x


# # CICIDS2017 80 dimension features
# class CNN(torch.nn.Module):
#     def __init__(self,N_class):
#         super(CNN,self).__init__()
#         self.conv1 = torch.nn.Conv1d(in_channels=1,out_channels=128,kernel_size=2,stride=1,padding=0)
#         self.max_pool1=torch.nn.MaxPool1d(kernel_size=2,stride=2)
#         self.conv2 = torch.nn.Conv1d(in_channels=128,out_channels=64,kernel_size=2,stride=1,padding=0)
#         self.max_pool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv3 = torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2, stride=1, padding=0)
#         self.max_pool3 = torch.nn.MaxPool1d(kernel_size=2, stride=2)
#         self.fn1 = torch.nn.Linear(256, N_class)
#         # self.fn1 = torch.nn.Linear(4736, 1024)
#         # self.fn2 = torch.nn.Linear(1024, 512)
#         # self.fn3 = torch.nn.Linear(128,N_class)
#         self.sigmoid = torch.nn.Sigmoid()
#         self.relu = torch.nn.ReLU(inplace=True)
#         self.dropout = torch.nn.Dropout(0.5)
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.max_pool1(x)
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.max_pool2(x)
#         # CICIDS2017 80 dimension features
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.max_pool3(x)
#         x = self.dropout(x)
#         x = x.view(x.size(0),-1)
#         # print(x.size())
#         av = self.fn1(x)
#         # x = self.relu(x)
#         # x = self.fn2(x)
#         # x = self.relu(x)
#         # av = self.fn3(x)
#         x = self.sigmoid(av)
#         return av,x

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


# class CNN_NORMAL(torch.nn.Module):
#     """docstring for CNN_NORMAL"""
#     def __init__(self, N_class=4):
#         super(CNN_NORMAL, self).__init__()
#         self.avg_kernel_size = 4
#         self.i_size = 16
#         self.num_class = N_class
#         self.input_space = None
#         self.input_size = (self.i_size, self.i_size, 1)
#         self.conv1 = torch.nn.Sequential(
#             torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 16*16*32
#             torch.nn.ReLU(inplace=True),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, )  # 8*8*16
#         )
#         self.conv2 = torch.nn.Sequential(
#             torch.nn.Conv2d(32, 128, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 8*8*128
#             torch.nn.ReLU(inplace=True),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, )  # 4*4*128
#         )
#         self.avg_pool = torch.nn.AvgPool2d(kernel_size=self.avg_kernel_size, stride=2, ceil_mode=False)  # 1*1*128
#         self.fc = torch.nn.Sequential(
#             torch.nn.BatchNorm1d(1 * 1 * 128),
#             torch.nn.Dropout(0.5),
#             torch.nn.Linear(1 * 1 * 128, self.num_class, bias=True)
#
#             # torch.nn.Linear(1 * 1 * 128, 32, bias=True),
#             # torch.nn.ReLU(),
#             # torch.nn.Linear(32, self.num_class, bias=True)
#         )
#         self.sigmoid = torch.nn.Sigmoid()
#
#     def features(self, input_data):
#         x = self.conv1(input_data)
#         x = self.conv2(x)
#         return x
#
#     def logits(self, input_data):
#         x = self.avg_pool(input_data)
#         x = x.view(x.size(0), -1)
#         # x = input_data.view(input_data.size(0), -1)
#         x = self.fc(x)
#         return x
#
#     def forward(self, input_data):
#         x = self.features(input_data)
#         av = self.logits(x)
#         x = self.sigmoid(av)
#         return av, x


class CNN_NORMAL(torch.nn.Module):
    """docstring for CNN_NORMAL"""
    def __init__(self, N_class=4):
        super(CNN_NORMAL, self).__init__()
        self.avg_kernel_size = 4
        self.i_size = 16
        self.num_class = N_class
        self.input_space = None
        self.input_size = (self.i_size, self.i_size, 1)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 16*16*32
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, )  # 8*8*16
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 128, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 8*8*128
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, )  # 4*4*128
        )
        # self.conv3 = torch.nn.Sequential(
        #     torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 4*4*128
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.MaxPool2d(kernel_size=4, stride=4, padding=0, )  # 1*1*128
        # )
        # self.avg_pool = torch.nn.AvgPool2d(kernel_size=self.avg_kernel_size, stride=2, ceil_mode=False)  # 1*1*128
        self.fc = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(4*4*128),
            # # torch.nn.Dropout(0.5),
            # # torch.nn.Linear(1 * 1 * 128, self.num_class, bias=True)
            # # torch.nn.Dropout(0.5),
            # torch.nn.Linear(4*4*128, 128, bias=True),
            # torch.nn.ReLU(inplace=True),

            torch.nn.BatchNorm1d(4 * 4 * 128),
            torch.nn.Linear(4 * 4 * 128, self.num_class, bias=True)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def features(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        # x = self.conv3(x)
        return x

    def logits(self, input_data):
        # x = self.avg_pool(input_data)
        # x = x.view(x.size(0), -1)
        x = input_data.view(input_data.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self,input_data):
        x = self.features(input_data)
        av = self.logits(x)
        x = self.sigmoid(av)
        return av, x


# ---------------------
class SharedCNN(torch.nn.Module):
    def __init__(self,N_class):
        super(SharedCNN,self).__init__()
        self.sharedNet = cnn(N_class)

    def forward(self, indata, outdata):
        av_indata, pred_indata = self.sharedNet(indata)
        av_outdata, pred_outdata = self.sharedNet(outdata)
        return av_indata, pred_indata, av_outdata, pred_outdata


class CNN_LSTM(torch.nn.Module):
    def __init__(self, N_class=4):
        super(CNN_LSTM, self).__init__()
        self.num_class = N_class
        self.input_space = None
        self.num_layers = 2
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),  # 16*16*16
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4, stride=2)  # 8*8*16
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),  # 8*8*64
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4, stride=2)  # 4*4*64
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 16 * 16, bias=True)
        )
        self.lstm = torch.nn.LSTM(input_size=16, hidden_size=64, num_layers=self.num_layers, batch_first=True,
                                  dropout=0.5)
        self.classifier = torch.nn.Linear(64, self.num_class, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def features(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input_data):
        x = self.features(input_data)
        x = x.view(x.size(0), 16, 16)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        # out = self.sigmoid(out)
        return out

class CNN_LSTM_NORMAL(torch.nn.Module):
    """docstring for ClassName"""
    def __init__(self, N_class=4):
        super(CNN_LSTM_NORMAL, self).__init__()
        self.num_class =N_class
        self.input_space = None
        self.input_size = (16,16,1)
        self.num_layers = 2
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,16,kernel_size=3,stride=1,dilation=1,padding=1,bias=True),#16*16*16
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0,)#8*8*16
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,64,kernel_size=3,stride=1,dilation=1,padding=1,bias=True),#8*8*64
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0,)#4*4*64
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4*4*64,16*16,bias=True)
        )
        self.lstm = torch.nn.LSTM(input_size=16,hidden_size=64,num_layers=self.num_layers,batch_first=True,dropout=0.5)
        self.classifier = torch.nn.Linear(64,self.num_class,bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def features(self,input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    def forward(self,input_data):
        x = self.features(input_data)
        x = x.view(x.size(0),16,16)
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        out = self.classifier(out)
        # out = self.sigmoid(out)
        return out


class LSTM(torch.nn.Module):
    def __init__(self, N_class=4):
        super(LSTM, self).__init__()
        self.num_class = N_class
        self.input_size = (16, 16, 1)
        self.num_layers = 1
        self.lstm = torch.nn.LSTM(input_size=16, hidden_size=128, num_layers=self.num_layers, batch_first=True, dropout=0.5)
        self.classifier = torch.nn.Linear(128, self.num_class, bias=True)
    def forward(self, input_data):
        x = input_data.view(input_data.size(0), 16, 16)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out


def cnn(N_class):
    # KDD Dataset
    # model = CNN(N_class)
    # new dataset CICIDS2017
    model = CNN_NORMAL(N_class)
    return model

def cnnlstm(N_class):
    # KDD
    # model = CNN_LSTM(N_class)
    # CICIDS2017 256(16*16)
    model = CNN_LSTM_NORMAL(N_class)
    return model

def lstm(N_class):
    model = LSTM(N_class)
    return model

