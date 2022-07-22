import torch






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



def cnn(N_class):
    # new dataset CICIDS2017
    model = CNN_NORMAL(N_class)
    return model

