import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import cnn as models
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import math
import mmd
import cal_metrics
import argparse
from collections import Iterable                            # < py38
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# convert a list of list to a list [[],[],[]]->[,,]
def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def readdataset_known():
    # read header
    mydata_benign = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_Monday-Benign.csv')  # 62639
    mydata_DDoS = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_DDoS.csv')  # 261226
    mydata_hulk = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_DoS-Hulk.csv')  # 474656
    mydata_portscan_1 = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_PortScan_1.csv')  # 755
    mydata_portscan_2 = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_PortScan_2.csv')  # 318881
    # mydata_sshpatator = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_SSH-Patator.csv')  # 27545
    # mydata_glodeneye = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_DoS-GlodenEye.csv')  # 20543
    # mydata_slowloris = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_DoS-Slowloris.csv')  # 10537
    # mydata_webattack = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_WebAttack.csv')  # 10537
    mydata_ftppatator = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_FTP-Patator.csv')  # 19941

    benign = mydata_benign.values[0:60000,1:]
    ddos = mydata_DDoS.values[0:50000, 1:]
    hulk = mydata_hulk.values[0:50000, 1:]
    portscan_1 = mydata_portscan_1.values[0:700, 1:]
    portscan_2 = mydata_portscan_2.values[0:49300, 1:]
    # ssh_patator = mydata_sshpatator.values[0:20000, 1:]
    # glodeneye = mydata_glodeneye.values[:, 1:]#20453
    # slowloris = mydata_slowloris.values[:, 1:]# 10537
    # webattack = mydata_webattack.values[:, 1:]# 10537
    ftp_patator = mydata_ftppatator.values[:, 1:]# 19941

    x_benign = benign[:,:-1]
    x_ddos = ddos[:, :-1]
    x_hulk = hulk[:, :-1]
    x_portscan_1 = portscan_1[:, :-1]
    x_portscan_2 = portscan_2[:, :-1]
    # x_sshpatator = ssh_patator[:, :-1]
    # x_glodeneye = glodeneye[:, :-1]
    # x_slowloris = slowloris[:, :-1]
    # x_webattack = webattack[:, :-1]
    x_ftppatator = ftp_patator[:, :-1]

    y_benign = benign[:,-1]
    y_ddos = ddos[:, -1]
    y_hulk = hulk[:, -1]
    y_portscan_1 = portscan_1[:, -1]
    y_portscan_2 = portscan_2[:, -1]
    # y_sshpatator = ssh_patator[:, -1]
    # y_glodeneye = glodeneye[:, -1]
    # y_slowloris = slowloris[:, -1]
    # y_webattack = webattack[:, -1]
    y_ftppatator = ftp_patator[:, -1]

    x_tr_benign, x_te_benign, y_tr_benign, y_te_benign = train_test_split(x_benign, y_benign, test_size=0.2, random_state=1)
    x_tr_ddos, x_te_ddos, y_tr_ddos, y_te_ddos = train_test_split(x_ddos, y_ddos, test_size=0.2, random_state=1)
    x_tr_hulk, x_te_hulk, y_tr_hulk, y_te_hulk = train_test_split(x_hulk, y_hulk, test_size=0.2, random_state=1)
    x_tr_portscan_1, x_te_portscan_1, y_tr_portscan_1, y_te_portscan_1 = train_test_split(x_portscan_1, y_portscan_1,test_size=0.2, random_state=1)
    x_tr_portscan_2, x_te_portscan_2, y_tr_portscan_2, y_te_portscan_2 = train_test_split(x_portscan_2, y_portscan_2,test_size=0.2, random_state=1)
    # x_tr_sshpatator, x_te_sshpatator, y_tr_sshpatator, y_te_sshpatator = train_test_split(x_sshpatator, y_sshpatator,test_size=0.2, random_state=1)
    # x_tr_glodeneye, x_te_glodeneye, y_tr_glodeneye, y_te_glodeneye = train_test_split(x_glodeneye, y_glodeneye,test_size=0.2, random_state=1)
    # x_tr_slowloris, x_te_slowloris, y_tr_slowloris, y_te_slowloris = train_test_split(x_slowloris, y_slowloris,test_size=0.2, random_state=1)
    # x_tr_webattack, x_te_webattack, y_tr_webattack, y_te_webattack = train_test_split(x_webattack, y_webattack,test_size=0.2, random_state=1)
    x_tr_ftppatator, x_te_ftppatator, y_tr_ftppatator, y_te_ftppatator = train_test_split(x_ftppatator, y_ftppatator,test_size=0.2, random_state=1)

    x_tr_portscan = np.concatenate((x_tr_portscan_1, x_tr_portscan_2), axis=0)
    y_tr_portscan = np.concatenate((y_tr_portscan_1, y_tr_portscan_2))
    x_te_portscan = np.concatenate((x_te_portscan_1, x_te_portscan_2), axis=0)
    y_te_portscan = np.concatenate((y_te_portscan_1, y_te_portscan_2))

    y_tr_benign = np.array([0] * len(y_tr_benign))
    y_tr_ddos = np.array([1] * len(y_tr_ddos))
    y_tr_hulk = np.array([2] * len(y_tr_hulk))
    y_tr_portscan = np.array([3] * len(y_tr_portscan))
    # y_tr_sshpatator = np.array([3] * len(y_tr_sshpatator))
    # y_tr_glodeneye = np.array([3] * len(y_tr_glodeneye))
    # y_tr_slowloris = np.array([4] * len(y_tr_slowloris))
    # y_tr_webattack = np.array([3] * len(y_tr_webattack))
    y_tr_ftppatator = np.array([4] * len(y_tr_ftppatator))

    y_te_benign = np.array([0] * len(y_te_benign))
    y_te_ddos = np.array([1] * len(y_te_ddos))
    y_te_hulk = np.array([2] * len(y_te_hulk))
    y_te_portscan = np.array([3] * len(y_te_portscan))
    # y_te_sshpatator = np.array([3] * len(y_te_sshpatator))
    # y_te_glodeneye = np.array([3] * len(y_te_glodeneye))
    # y_te_slowloris = np.array([4] * len(y_te_slowloris))
    # y_te_webattack = np.array([3] * len(y_te_webattack))
    y_te_ftppatator = np.array([4] * len(y_te_ftppatator))

    x_train = np.concatenate((x_tr_benign, x_tr_ddos, x_tr_hulk, x_tr_portscan, x_tr_ftppatator))#, x_tr_sshpatator,x_tr_glodeneye, x_tr_slowloris
    y_train = np.concatenate((y_tr_benign, y_tr_ddos, y_tr_hulk, y_tr_portscan, y_tr_ftppatator))

    x_test = np.concatenate((x_te_benign, x_te_ddos, x_te_hulk, x_te_portscan, x_te_ftppatator))
    y_test = np.concatenate((y_te_benign, y_te_ddos, y_te_hulk, y_te_portscan, y_te_ftppatator))

    return x_train,y_train,x_test,y_test

def readdataset_unknown(unknown_attack_type):
    if unknown_attack_type == 'heartbleed':
        mydata_heartbleed = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_Heartbleed-Port.csv')  # 9859
        heartbleed = mydata_heartbleed.values[:, 1:]
        x_heartbleed = heartbleed[:, :-1]
        y_heartbleed = heartbleed[:, -1]
        x = x_heartbleed
        y = np.array([5]*len(y_heartbleed))
    elif unknown_attack_type == 'infiltration':
        mydata_infiltration = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_Infiltration-2.csv')  #5126
        infiltration = mydata_infiltration.values[:, 1:]
        x_infiltration = infiltration[:, :-1]
        y_infiltration = infiltration[:, -1]
        x = x_infiltration
        y = np.array([5]*len(y_infiltration))
    elif unknown_attack_type == 'botnet':
        mydata_botnet = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_Botnet.csv')  # 2075
        botnet = mydata_botnet.values[:, 1:]
        x_botnet = botnet[:, :-1]
        y_botnet = botnet[:, -1]
        x = x_botnet
        y = np.array([5]*len(y_botnet))
    elif unknown_attack_type == 'slowhttp':
        mydata_slowhttp = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_DoS-Slowhttptest.csv')  # 2075
        slowhttp = mydata_slowhttp.values[:, 1:]
        x_slowhttp = slowhttp[:, :-1]
        y_slowhttp = slowhttp[:, -1]
        x = x_slowhttp
        y = np.array([5]*len(y_slowhttp))
    elif unknown_attack_type == 'glodeneye':
        mydata_glodeneye = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_DoS-GlodenEye.csv')  #20543
        glodeneye = mydata_glodeneye.values[:, 1:]
        x_glodeneye = glodeneye[:, :-1]
        y_glodeneye = glodeneye[:, -1]
        x = x_glodeneye
        y = np.array([5]*len(y_glodeneye))
    elif unknown_attack_type == 'sshpatator':
        mydata_sshpatator = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_SSH-Patator.csv')  # 27545
        sshpatator = mydata_sshpatator.values[:, 1:]
        x_sshpatator = sshpatator[:, :-1]
        y_sshpatator = sshpatator[:, -1]
        x = x_sshpatator
        y = np.array([5]*len(y_sshpatator))
    elif unknown_attack_type == 'slowloris':
        mydata_slowloris = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_DoS-Slowloris.csv')  # 10537
        slowloris = mydata_slowloris.values[:, 1:]
        x_slowloris = slowloris[:, :-1]
        y_slowloris = slowloris[:, -1]
        x = x_slowloris
        y = np.array([5]*len(y_slowloris))
    elif unknown_attack_type == 'webattack':
        mydata_webattack = pd.read_csv('./imbalanced_flow_data/flow_labeled/labeld_WebAttack.csv')  # 10537
        webattack = mydata_webattack.values[:, 1:]
        x_webattack = webattack[:, :-1]
        y_webattack = webattack[:, -1]
        x = x_webattack
        y = np.array([5]*len(y_webattack))
    return x, y

def get_source_loader( data_source_np0, label_sources_np0):
    data_source = torch.from_numpy(data_source_np0)
    label_source = torch.from_numpy(label_sources_np0)
    torch_source_dataset = Data.TensorDataset(data_source, label_source)
    source_loader = Data.DataLoader(
        dataset=torch_source_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多线程来读数据
    )
    return source_loader



def cal_centroid(label,pred,N_class,last_centroids):
    label_pred = torch.argmax(pred,1).view(-1).type(torch.LongTensor)
    correct = torch.eq(label_pred,label)
    correct_index = torch.nonzero(correct).contiguous().view(-1)
    pred_correct = torch.index_select(pred, 0, correct_index)
    label_correct = torch.index_select(label,0,correct_index)
    # print('label_correct',label_correct)
    # print('pred_correct',pred_correct.size())
    label_correct = label_correct.view(-1,1)
    if label_correct.size(0)>0:
        class_count = torch.zeros(N_class, 1)
        for col in range(N_class):
            # k = torch.ones(label_correct.size())*col
            # print(k)
            kk = torch.tensor([col]).expand_as(label_correct)
            # print(col)
            # print(kk)
            # kk = (torch.tensor([col-1]).unsqueeze(0).expand(label_correct.size(),1))
            class_count[col,] = torch.eq(label_correct, kk.type(torch.LongTensor)).sum(0)
            # 对于每个batch来说，label_correct不一定包含所有类别
            # class_count = torch.zeros(N_class,1).scatter_add(0,label_correct,torch.ones(label_correct.size()))
        positive_class_count = torch.max(class_count, torch.ones(class_count.size()))
        scatter_index = label_correct.expand(label_correct.size(0), N_class)
        centroid = torch.zeros(N_class, N_class).scatter_add(0,scatter_index.type(torch.LongTensor),pred_correct)
        # print('centroid sum',centroid)
        # print('pasitivie_class_count',positive_class_count)
        mean_centroid = centroid / positive_class_count
        # print('mean_centroid',mean_centroid)
        current_centroids = mean_centroid
        for i in range(0, mean_centroid.size(0)):
            if positive_class_count[i] == 1:
                current_centroids[i,] = last_centroids[i,]
                # print('using one class centroids')
                # if torch.equal(mean_centroid[i,],torch.zeros(N_class,1)):
                #     current_centroids[i,] = last_centroids[i,]
                #     print('AAA')
            else:
                current_centroids[i,] = 0.5*last_centroids[i,]+0.5*current_centroids[i,]
    else:
        current_centroids = last_centroids
        # print('all use last_centroids')

    return current_centroids

# calculate every smaple to all the class centroids's distance,return size (N,N_class)
def cal_dist_to_centroids(pred,centroid):
    dist = []
    for i in range(centroid.size(0)):
        dist_to_centroid = ((pred-centroid[i,])**2).sum(1)
        dist.append(dist_to_centroid)
    dist_to_centroids = torch.stack(dist,dim=1)
    return dist_to_centroids

# calculate the cosine similarity to all the class centroids's distance,return size (N,N_class)
def cal_cos_dist_to_centroids(pred,centroid):
    cos_dist = []
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    for i in range(centroid.size(0)):
        # print('centroid i size',centroid[i,].size())
        cos_dist_to_centroid = cos(pred,centroid[i,].view(1,-1))
        # print('cos_dist_to_centroid size',cos_dist_to_centroid.size())
        cos_dist.append(cos_dist_to_centroid)
    cos_dist_to_centroids = torch.stack(cos_dist,dim=1)
    return cos_dist_to_centroids


def cal_min_dis_to_centroid(pred,centroid):
    all_distances = cal_dist_to_centroids(pred=pred, centroid=centroid)
    dist_to_its_centriod = torch.min(all_distances, dim=1)[0]
    min_dist_class_index = torch.min(all_distances,dim=1)[1]
    return dist_to_its_centriod,min_dist_class_index

def cal_max_cos_dist_to_centroid(pred,centroid):
    all_cos_distances = cal_cos_dist_to_centroids(pred=pred,centroid=centroid)
    cos_dist_to_its_centroid = torch.max(all_cos_distances,dim=1)[0]
    max_cos_dist_class_index = torch.max(all_cos_distances,dim=1)[1]
    return cos_dist_to_its_centroid,max_cos_dist_class_index

def cal_threshold(label,pred,centroid,rank_rate):
    label_pred = torch.argmax(pred, 1).view(-1).type(torch.LongTensor)
    correct = torch.eq(label_pred, label)
    correct_index = torch.nonzero(correct).contiguous().view(-1)
    pred_correct = torch.index_select(pred, 0, correct_index)
    dist = []
    threshold = torch.zeros(centroid.size(0))
    if pred_correct.size(0)>0:
        centroid_index = torch.argmax(pred_correct, 1).type(torch.LongTensor)
        for i in range(pred_correct.size(0)):
            dist_to_its_centriod = ((pred_correct[i,] - centroid[centroid_index[i]]) ** 2).sum()
            dist.append(dist_to_its_centriod)
        dist_to_its_centriod = torch.stack(dist)
        for j in range(centroid.size(0)):
            class_j_index = (centroid_index==j).nonzero().view(-1)
            if class_j_index.size(0)>0:
                dist_to_j_centroid = torch.gather(dist_to_its_centriod, 0, class_j_index)
                ascend_dist = torch.sort(dist_to_j_centroid)[0]
                threshold_index = torch.floor(dist_to_j_centroid.size(0) * torch.tensor(rank_rate)).type(
                    torch.LongTensor)
                threshold[j] = ascend_dist[threshold_index]
    # else:
    #     threshold = 0
    return threshold


def cal_inter_dist(dist_to_centroids, label, N_class):
    label = label.view(-1, 1)
    class_count = torch.zeros(N_class, 1).scatter_add(0, label, torch.ones(label.size()))
    positive_class_count = torch.max(class_count, torch.ones(class_count.size()))
    distCC = torch.zeros(N_class,N_class).scatter_add(0,label.expand(label.size(0),N_class),dist_to_centroids)
    mean_distCC = distCC/positive_class_count
    distC2 = torch.zeros(N_class,2).scatter_add(1,torch.eye(N_class,N_class).type(torch.LongTensor),mean_distCC)
    intra_inter = distC2.sum(0)
    # intra_inter size=(2,1) intra_inter[0]=sum(pred-the other class centroid) intra_inter[1]=sum(pred-its class centroid)
    return intra_inter

def intra_spread_loss(pred,label,centroid,N_class):
    dist_to_centroids = cal_dist_to_centroids(pred,centroid)
    intra_inter = cal_inter_dist(dist_to_centroids, label, N_class)
    intra_spread = intra_inter[1]
    return intra_spread

def inter_distance(A):
    s0 = int(A.size(0))
    s1 = int(A.size(1))
    # unsqueeze 指定维度上进行扩充1维
    aa = A.unsqueeze(0).expand(s0,s0,s1).contiguous().view(-1,s1)
    aat = A.unsqueeze(1).expand(s0,s0,s1).contiguous().view(-1,s1)
    dist = ((aa-aat)**2).sum(1).contiguous().view(s0,s0)
    # print('inter_centroid: {:.4f}',dist)
    return dist


def find_nonezero_min(dist):
    dist1 = dist.contiguous().view(-1)
    ascend_dist = torch.sort(dist1)[0]
    if ascend_dist[-1] != 0:
        # 存在非0的距离
        j = dist.size(0)
        while j < ascend_dist.size(0):
            if ascend_dist[j] != 0:
                min_dist = ascend_dist[j]
                # print('minmin')
                break
            else:
                j += 1
    else:
        min_dist = ascend_dist[-1]
        # print('000')
    return min_dist


def inter_spread_loss(centroid):
    dist = inter_distance(centroid)
    min = find_nonezero_min(dist)
    inter_spread = min
    return inter_spread


def fisher_loss(pred,centroid):
    class_index = torch.argmax(pred,1).view(-1).type(torch.LongTensor)
    # print('class_index',class_index)
    sum_Sw = 0
    for i in range(pred.size(0)):
        sum_Sw += ((pred[i,]-centroid[class_index[i],])**2).sum()
    Sw = sum_Sw/pred.size(0)
    sum_Sb = inter_distance(centroid).sum()
    Sb = sum_Sb/(centroid.size(0)*(centroid.size(0)-1))
    return Sw,Sb


# train with fisher loss and KL loss
def train_sharedcnn(epoch,model,rank_rate,max_threshold,data,label,N_class):
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)
    correct = 0
    loss = torch.nn.CrossEntropyLoss()
    model.train()
    train_loader = get_source_loader(data,label)
    len_train_dataset = len(train_loader.dataset)
    len_train_loader = len(train_loader)
    iter_train = iter(train_loader)
    num_iter = len_train_loader
    last_centroids = torch.load('CICIDS_centroids.pt')
    for i in range(1, num_iter):
        data_train, label_train = iter_train.next()
        noise = torch.FloatTensor(data_train.size(0),data_train.size(1), data_train.size(2), data_train.size(3)).normal_(0, 1)
        out_data = data_train.float() + noise
        data_train, label_train = Variable(data_train).float().to(device), Variable(label_train).type(
            torch.LongTensor).to(device)
        out_data = Variable(out_data).float().to(device)
        with torch.no_grad():
            last_centroids = Variable(last_centroids)
        optimizer.zero_grad()
        train_av, train_pred, outdata_av, outdata_pred = model(data_train,out_data)
        centroid = cal_centroid(label_train.cpu(), train_av.cpu(), N_class, last_centroids)
        last_centroids.data = centroid
        Sw, Sb = fisher_loss(pred=train_av.cpu(), centroid=centroid)
        threshold = cal_threshold(label=label_train.cpu(), pred=train_av.cpu(), centroid=centroid, rank_rate=rank_rate)
        cross_loss = loss(train_pred, label_train)
        sw_lamda= (lamda.to(device)) * (Sw.to(device))
        sb_alpha = ((lamda * alpha).to(device)) * (Sb.to(device))
        fisherloss = sw_lamda - sb_alpha
        KL_loss_nobeta = mmd.mmd_rbf_noaccelerate(outdata_av,train_av)
        KL_loss = (beta.to(device))* KL_loss_nobeta
        Loss = cross_loss + fisherloss - KL_loss
        pred = train_pred.max(1)[1]
        correct += pred.eq(label_train.data.view_as(pred)).cpu().sum()
        Loss.backward()
        optimizer.step()
    for m in range(threshold.size(0)):
        if threshold[m] > max_threshold[m]:
            max_threshold[m] = threshold[m]
    torch.save(last_centroids.data, 'CICIDS_centroids.pt')
    Accuracy = 100. * correct.type(torch.FloatTensor) / len_train_dataset
    print(
        'Train Epoch:{}\tLoss: {:.6f}\tcross_loss: {:.6f}\tfisherloss: {:.6f}\tSw: {:.6f}\tSb: {:.6f}\tSw_lamda: {:.6f}\tSb_alpha: {:.6f}\tMMD: {:.6f}\tKLloss: {:.6f}\tAccuracy: {:.4f}'.format(
            epoch, Loss, cross_loss, fisherloss, Sw, Sb, sw_lamda, sb_alpha, KL_loss_nobeta, KL_loss, Accuracy))
    logging.info(
        'Train Epoch:{}\tLoss: {:.6f}\tcross_loss: {:.6f}\tfisherloss: {:.6f}\tSw: {:.6f}\tSb: {:.6f}\tMMD: {:.6f}\tKLloss: {:.6f}\tAccuracy: {:.4f}'.format(
            epoch, Loss, cross_loss, fisherloss, Sw, Sb, KL_loss_nobeta, KL_loss, Accuracy))
    return max_threshold


# test the known data based on the max probabality
def test(model,data,label):
    correct = 0
    test_loss = 0
    known_data_pred = []
    known_data_label = []
    loss = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loader = get_source_loader(data, label)
    len_test_dataset = len(test_loader.dataset)
    for data_test,label_test in test_loader:
        data_test, label_test = Variable(data_test).float().to(device),Variable(label_test).type(torch.LongTensor).to(device)
        # test_av, test_pred, _, _ = model(data_test, data_test)
        # 20200826 cnnlstm
        test_pred = model(data_test)
        test_loss += loss(test_pred, label_test)
        pred = test_pred.max(1)[1]
        correct += pred.eq(label_test.data.view_as(pred)).cpu().sum()
        known_data_pred.append(pred.cpu().detach().data.tolist())
        known_data_label.append(label_test.cpu().detach().data.tolist())
    Accuracy = 100. * correct.type(torch.FloatTensor) / len_test_dataset
    test_loss /= len_test_dataset
    print('Test Loss: {:.6f}\tAccuracy: {:.4f}'.format(test_loss, Accuracy))
    list_known_data_label = list(flatten(known_data_label))
    list_known_data_pred = list(flatten(known_data_pred))
    print(classification_report(list_known_data_label, list_known_data_pred))
    return list_known_data_label, list_known_data_pred


# class incremental -> calculate the new class centroid
def cal_new_class_centroid(model):
    model.eval()
    # test_loader, N_class = get_source_loader(train_filename, dataset_number=1)
    test_loader, N_class = get_source_loader(test_filename, dataset_number=2)
    sum_new_class_centroid = torch.zeros(1,N_class1)
    txt_file = open('snmpguess.csv','ab')
    for data_test, label_test in test_loader:
        data_test, label_test = Variable(data_test).float().to(device), Variable(label_test).type(torch.LongTensor).to(
            device)
        # test_av, test_pred = model(data_test)
        test_av, test_pred, _, _ = model(data_test, data_test)
        np.savetxt(txt_file,test_av.cpu().detach().numpy(),fmt=['%.4f','%.4f','%.4f'],delimiter=',')
        # print('test_av',test_av)
        sum = (test_av.cpu().sum(0)/(test_av.cpu().size(0))).view(1,N_class1)
        sum_new_class_centroid += sum
    txt_file.close()
    new_class_centroid = sum_new_class_centroid/len(test_loader)
    sum_Sw = 0
    for i in range(test_av.size(0)):
        sum_Sw += ((test_av[i,].cpu()-new_class_centroid)**2).sum()
    Sw = sum_Sw/test_av.size(0)
    return new_class_centroid, Sw


# class incremental -> classify the base classes and the new class
def classify_based_distance(model,centroid):
    correct = 0
    new_class_correct = 0
    num_label_new_class = 0
    num_classes = centroid.size(0)-1
    print('centroid size-1',num_classes)
    model.eval()
    # test_loader, N_class = get_source_loader(train_filename, dataset_number=1)
    test_loader, N_class = get_source_loader(test_filename, dataset_number=2)
    len_test_dataset = len(test_loader.dataset)
    for data_test, label_test in test_loader:
        data_test, label_test = Variable(data_test).float().to(device), Variable(label_test).type(torch.LongTensor).to(
            device)
        test_av, test_pred, _, _ = model(data_test, data_test)
        # test_av, test_pred = model(data_test)
        _, pred = cal_min_dis_to_centroid(test_av.cpu(),centroid)
        # max_dist, pred = cal_max_cos_dist_to_centroid(test_av.cpu(),centroid)
        new_class_correct += torch.eq(pred,num_classes).sum(0)
        # print('num_newclass',new_class_correct)
        num_label_new_class += torch.eq(label_test.cpu(),num_classes).sum(0)
        # print('num_gt_newclass',num_label_new_class)
        correct += pred.eq(label_test.cpu().data.view_as(pred)).sum()
    print(test_av.cpu())
    print(label_test.cpu())
    Accuracy = 100. * correct.type(torch.FloatTensor) / len_test_dataset
    print('total_num_pred_newclass',new_class_correct)
    print('total_num_label_newclass', num_label_new_class)
    New_class_accuracy = 100. * new_class_correct.type(torch.FloatTensor) / num_label_new_class.type(torch.FloatTensor)
    print('Train Epoch:{}\tAccuracy: {:.4f}\tNew_class_Accuracy: {:.4f}'.format(epoch, Accuracy,New_class_accuracy))
    logging.info('Train Epoch:{}\tAccuracy: {:.4f}\tNew_class_Accuracy: {:.4f}'.format(epoch, Accuracy,New_class_accuracy))



# test the known data according the threshold
def test_knowndata(model,centroids,max_threshold,data,label,N_class):
    correct = 0
    test_loss = 0
    novelty = 0
    known_data_pred = []
    known_data_label = []
    loss = torch.nn.CrossEntropyLoss()
    # model.eval()
    test_loader = get_source_loader(data,label)
    len_test_dataset = len(test_loader.dataset)
    # txt_file0 = open(known_pred_filename,'ab')

    # txt_file1 = open(known_label_filename,'ab')
    # txt_file2 = open(known_data_filename, 'ab')
    # txt_file = open(known_score_filename, 'ab')

    last_centroids = torch.zeros(N_class,N_class)

    for data_test,label_test in test_loader:
        data_test, label_test = Variable(data_test).float().to(device),Variable(label_test).type(torch.LongTensor).to(device)
        test_av,test_pred, _, _ = model(data_test,data_test)

        centroid = cal_centroid(label_test.cpu(), test_av.cpu(), N_class, last_centroids)
        last_centroids = centroid

        test_loss += loss(test_pred, label_test)
        pred = test_pred.max(1)[1]
        dist_to_its_centriod, min_dist_class_index = cal_min_dis_to_centroid(pred=test_av.cpu(), centroid=centroids)
        # np.savetxt(txt_file1, label_test.cpu().detach().numpy(), fmt=['%d'], delimiter=',')
        # np.savetxt(txt_file2, test_av.cpu().detach().numpy(), fmt=['%.4f','%.4f','%.4f','%.4f','%.4f'], delimiter=',')
        # np.savetxt(txt_file, dist_to_its_centriod.detach().numpy(), fmt=['%.4f'], delimiter=',')
        for i in range(dist_to_its_centriod.size(0)):
            # pred[i]=min_dist_class_index[i]
            if dist_to_its_centriod[i] > max_threshold[min_dist_class_index[i]]:
                pred[i]=N_class
                novelty += 1
        # np.savetxt(txt_file0, pred.cpu().detach().numpy(), fmt=['%d'], delimiter=',')
        correct += pred.eq(label_test.data.view_as(pred)).cpu().sum()
        known_data_pred.append(pred.cpu().detach().data.tolist())
        known_data_label.append(label_test.cpu().detach().data.tolist())

    print('test known centroid',last_centroids)

    # txt_file0.close()
    # txt_file1.close()
    # txt_file2.close()
    # txt_file.close()
    Accuracy = 100. * correct.type(torch.FloatTensor) / len_test_dataset
    error_novelty = 100. * novelty / len_test_dataset
    test_loss /= len_test_dataset
    print('test_knowndata Train Epoch:{}\tLoss: {:.6f}\tAccuracy: {:.4f}\tFN_error of novelty: {:.4f}'.format(epoch, test_loss, Accuracy, error_novelty))
    logging.info('test_knowndata Train Epoch:{}\tLoss: {:.6f}\tAccuracy: {:.4f}\tFN_error of novelty: {:.4f}'.format(epoch, test_loss, Accuracy, error_novelty))
    list_known_data_label=list(flatten(known_data_label))
    list_known_data_pred = list(flatten(known_data_pred))
    return list_known_data_label,list_known_data_pred

def test_novelty(model,centroids,max_threshold,data,label,N_class):
    novelty = 0
    correct = 0
    model.eval()
    unknown_data_pred = []
    unknown_data_label = []
    test_loader = get_source_loader(data,label)
    len_test_dataset = len(test_loader.dataset)
    # txt_file0 = open(unknown_pred_filename, 'ab')
    # txt_file1 = open(unknown_label_filename, 'ab')

    # txt_file2 = open(unknown_data_filename, 'ab')
    # txt_file = open(unknown_score_filename, 'ab')
    sum_new_class_centroid = torch.zeros(1, N_class1)
    time_sum = 0
    for data_test,label_test in test_loader:
        data_test, label_test = Variable(data_test).float().to(device),Variable(label_test).type(torch.LongTensor).to(device)
        test_av, test_pred, _, _ = model(data_test, data_test)
        pred = test_pred.max(1)[1]
        dist_to_its_centriod,min_dist_class_index=cal_min_dis_to_centroid(pred=test_av.cpu(), centroid=centroids)
        # np.savetxt(txt_file1, label_test.cpu().detach().numpy(), fmt=['%d'], delimiter=',')
        # np.savetxt(txt_file2, test_av.cpu().detach().numpy(), fmt=['%.4f', '%.4f', '%.4f','%.4f','%.4f'], delimiter=',')
        # np.savetxt(txt_file, dist_to_its_centriod.detach().numpy(), fmt=['%.4f'], delimiter=',')
        start = time.clock()
        sum = (test_av.cpu().sum(0) / (test_av.cpu().size(0))).view(1, N_class1)
        sum_new_class_centroid += sum
        end = time.clock()
        time_sum+=(end-start)
        for i in range(dist_to_its_centriod.size(0)):
            if dist_to_its_centriod[i]>max_threshold[min_dist_class_index[i]]:
                pred[i] = N_class
                novelty += 1
            else:
                novelty += 0
        # np.savetxt(txt_file0, pred.cpu().detach().numpy(), fmt=['%d'], delimiter=',')
        unknown_data_pred.append(pred.cpu().detach().data.tolist())
        unknown_data_label.append(label_test.cpu().detach().data.tolist())
        # correct += pred.eq(label_test.data.view_as(pred)).cpu().sum()
    # txt_file0.close()
    # txt_file1.close()
    # txt_file2.close()
    # txt_file.close()
    start1=time.clock()
    new_class_centroid = sum_new_class_centroid / len(test_loader)
    end1=time.clock()
    time_sum+=(end1-start1)
    print('calculate new centroid',str(time_sum))
    print('new_class_centroid', new_class_centroid)
    # Novelty_accuracy = 100. * correct.type(torch.FloatTensor) / len_test_dataset
    TN_Accuracy = 100. * novelty / len_test_dataset
    print('test_novelty Train Epoch:{}\tTN_Accuracy: {:.4f}'.format(epoch, TN_Accuracy))
    logging.info('test_novelty Train Epoch:{}\tTN_Accuracy: {:.4f}'.format(epoch, TN_Accuracy))
    list_unknown_data_label = list(flatten(unknown_data_label))
    list_unknown_data_pred = list(flatten(unknown_data_pred))
    return list_unknown_data_label, list_unknown_data_pred

# test known and unknown data without novelty detection
def test_withoutnd(model,data,label):
    correct = 0
    model.eval()
    unknown_data_pred = []
    unknown_data_label = []
    test_loader = get_source_loader(data,label)
    len_test_dataset = len(test_loader.dataset)
    for data_test,label_test in test_loader:
        data_test, label_test = Variable(data_test).float().to(device),Variable(label_test).type(torch.LongTensor).to(device)
        test_av, test_pred, _, _ = model(data_test, data_test)
        pred = test_pred.max(1)[1]
        unknown_data_pred.append(pred.cpu().detach().data.tolist())
        unknown_data_label.append(label_test.cpu().detach().data.tolist())
        correct += pred.eq(label_test.data.view_as(pred)).cpu().sum()
    Accuracy = 100. * correct.type(torch.FloatTensor) / len_test_dataset
    print('test_novelty Train Epoch:{}\tAccuracy: {:.4f}'.format(epoch, Accuracy))
    logging.info('test_novelty Train Epoch:{}\t_Accuracy: {:.4f}'.format(epoch, Accuracy))
    list_unknown_data_label = list(flatten(unknown_data_label))
    list_unknown_data_pred = list(flatten(unknown_data_pred))
    return list_unknown_data_label, list_unknown_data_pred


# pretrain model using ours method
def train_pre(epoch,model,data,label):
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)
    correct = 0
    loss = torch.nn.CrossEntropyLoss()
    model.train()
    train_loader = get_source_loader(data, label)
    len_train_dataset = len(train_loader.dataset)
    len_train_loader = len(train_loader)
    iter_train = iter(train_loader)
    num_iter = len_train_loader
    for i in range(1, num_iter):
        data_train, label_train = iter_train.next()
        data_train, label_train = Variable(data_train).float().to(device),Variable(label_train).type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        # train_av, train_pred, _, _ = model(data_train,data_train)
        # CNN_LSTM
        train_pred = model(data_train)
        # print('train pred',train_pred.size())
        # print(train_pred)
        Loss = loss(train_pred, label_train)
        # print('loss',Loss)
        pred = train_pred.max(1)[1]
        # print('pred',pred.size)
        # print(pred)
        correct += pred.eq(label_train.data.view_as(pred)).cpu().sum()
        Loss.backward()
        optimizer.step()
    Accuracy = 100. * correct.type(torch.FloatTensor) / len_train_dataset
    print('Train Epoch:{}\tLoss: {:.6f}\tAccuracy: {:.4f}'.format(epoch, Loss, Accuracy))
    print(classification_report(label_train.data.view_as(pred), pred))


def load_model(model,model_path):
    model.load_state_dict(torch.load(model_path))
    return model



# N_class = 23
BATCH_SIZE = 512
LEARNING_RATE = 0.001
momentum = 0.9
l2_decay = 5e-4

# OURS METHODS
pre_train_model = 'pretrain_model_CICIDS_1105.pkl'
##odin_pre_train_model = './1123/odin_pretrain_model_CICIDS_1123.pkl'
epochs = 50


# train model with fisher loss + KL loss(introduce noise data)
if __name__ == '__main__':
    # OUR METHOD
    # logging.basicConfig(filename='./1123/20191124_CICIDS_CNNONLYFL.log', level=logging.DEBUG)
    logging.basicConfig(filename='./20200601_CICIDS_CNNONLYFL.log', level=logging.DEBUG)

    # define the save file
    parser = argparse.ArgumentParser()
    parser.add_argument('--unknow_attack', type=str, nargs='?', default='slowloris', help="attack name")#'infiltration','botnet','heartbleed','slowhttp','glodeneye','sshpatator','webattack','slowloris'
    # # BASELINE AND OIDN
    parser.add_argument('--magnitude', type=float, nargs='?', default=0.1, help="magnitude name")
    parser.add_argument('--temperature', type=float, nargs='?', default=10, help="temperature name")
    args = parser.parse_args()
    print(args.unknow_attack)
    logging.info(args.unknow_attack)
    # # OUR SAVE MODEL
    # # save_model = './1123/save_model_CICIDS_5_1122.pkl'
    # save_model = './save_model_CICIDS_5_20200601.pkl'

    # known_pred_filename = 'pred_known_CICIDS_1122.csv'
    known_data_filename = './1123/OUR/centroid_knowdata_CICIDS_1124.csv'
    known_label_filename = './1123/OUR/label_known_CICIDS_1124.csv'
    known_score_filename = './1123/OUR/score_knowdata_CICIDS_1123.csv' ### for cal_metrics
    unknown_score_filename = './1123/OUR/score_unknowdata_CICIDS_'+args.unknow_attack+'1123.csv' ### for cal_metrics
    unknown_data_filename = './1123/OUR/centroid_unknowdata_CICIDS_'+args.unknow_attack+'1123.csv' ### for cluster
    # unknown_pred_filename = 'pred_unknown_CICIDS_'+args.unknow_attack+'1122.csv'
    # unknown_label_filename = 'label_unknown_KDD99_CICIDS_'+args.unknow_attack+'1122.csv'


    max_tpr95 = 0
    max_auroc = 0
    max_auprin = 0
    max_auprout = 0
    max_detection_err = 0
    max_all_precision = 0
    max_all_recall = 0
    max_all_fscore = 0
    max_known_precision = 0
    max_known_recall = 0
    max_known_fscore = 0
    max_unknown_precision = 0
    max_unknown_recall = 0
    max_unknown_fscore = 0
    np_known_train,np_known_train_labels,np_known_test,np_known_test_labels = readdataset_known()
    np_unknown_test,np_unknown_test_labels = readdataset_unknown(args.unknow_attack)
    N_class1 = np.unique(np_known_train_labels).shape[0]
    np_all = np.concatenate((np_known_train,np_known_test,np_unknown_test), axis=0)
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(np_all)
    np_known_train_norm = min_max_scaler.transform(np_known_train)
    np_known_test_norm = min_max_scaler.transform(np_known_test)
    np_unknown_test_norm = min_max_scaler.transform(np_unknown_test)
    np_known_train_norm_3 = np_known_train_norm.reshape(-1, 1, 16, 16)
    np_known_test_norm_3 = np_known_test_norm.reshape(-1, 1, 16, 16)
    np_unknown_test_norm_3 = np_unknown_test_norm.reshape(-1, 1, 16, 16)
    # np_known_train_norm_3 = np_known_train_norm[:, np.newaxis, :, :]
    # np_known_test_norm_3 = np_known_test_norm[:, np.newaxis, :, :]
    # np_unknown_test_norm_3 = np_unknown_test_norm[:, np.newaxis, :, :]
    print('train known data size',np_known_train_norm_3.shape,'train label size',np_known_train_labels.shape,'N_class',N_class1)
    print('test known data size',np_known_test_norm_3.shape,'test label size',np_known_test_labels.shape)
    print('unkown data size',np_unknown_test_norm_3.shape,'unkown label size',np_unknown_test_labels.shape)

    # # OUR METHOD Test the model
    # epoch = 1
    # model = models.SharedCNN(N_class1).to(device)
    # # # # OUR METHOD
    # model = load_model(model, model_path=save_model)
    # centroids = torch.load('best_CICIDS_centroids_1121.pt')
    # max_threshold = torch.load('best_CICIDS_maxthreshold_1121.pt')
    # # Testing
    # t0 = time.clock()
    # list_known_data_label, list_known_data_pred = test_knowndata(model, centroids=centroids,max_threshold=max_threshold, data=np_known_test_norm_3,label=np_known_test_labels, N_class=N_class1)
    # t1 = time.clock()
    # print('test_knowndata',str(t1-t0))
    # list_unknown_data_label, list_unknown_data_pred = test_novelty(model, centroids=centroids,max_threshold=max_threshold,data=np_unknown_test_norm_3,label=np_unknown_test_labels, N_class=N_class1)
    # t2 = time.clock()
    # print('test_unknowndata',str(t2-t1))
    # list_all_data_label = list_known_data_label + list_unknown_data_label
    # list_all_data_pred = list_known_data_pred + list_unknown_data_pred
    # print(classification_report(list_all_data_label, list_all_data_pred))
    # print(confusion_matrix(list_all_data_label, list_all_data_pred))
    # # calculate the metrics
    # knowndata_distance = pd.read_csv(known_score_filename, header=None)
    # unkowndata_distance = pd.read_csv(unknown_score_filename, header=None)
    # known = -np.array(knowndata_distance)
    # novelty = -np.array(unkowndata_distance)
    # tpr95 = cal_metrics.tpr95(known, novelty)
    # auroc = cal_metrics.auroc(known, novelty)
    # auprin = cal_metrics.auprIn(known, novelty)
    # auprout = cal_metrics.auprOut(known, novelty)
    # detection_error = cal_metrics.detection(known, novelty)
    # print('fpr at tpr95',tpr95,'auroc', auroc,'auprin', auprin,'auprout', auprout,'detection error', detection_error)
    # logging.info("AUROC: {:.4f}\tAUPRIN: {:.4f}\tAUPROUT: {:.4f}\tDetection_Error: {:.4f}\tFPR_at_TPR95: {:.4f}".format(auroc,auprin,auprout,detection_error,tpr95))


    # # Train OUR model
    model = models.SharedCNN(N_class1).to(device)
    model = load_model(model,model_path=pre_train_model)
    correct = 0
    centroids_zeros = torch.zeros(N_class1,N_class1)
    # torch.save(centroids_zeros,'CICIDS_centroids.pt')
    torch.save(centroids_zeros, 'CICIDS_centroids20200601.pt')
    sum_time = 0
    for epoch in range(1, epochs+1):
        # # lamda = torch.tensor(0.01)
        # # alpha = torch.tensor(0.001*(math.exp(-5*(epoch/epochs))))
        # # OUR METHOD
        lamda = torch.tensor(0.05*(math.exp(-5*(epoch/epochs))))
        alpha = torch.tensor(0.0001*(math.exp(-5*(epoch/epochs))))
        beta = torch.tensor(0.01)
        print('lamda',lamda,'alpha',alpha)
        max_threshold = torch.zeros(N_class1)
        start_time = time.clock()
        max_threshold=train_sharedcnn(epoch,model,rank_rate=0.99,max_threshold=max_threshold,data=np_known_train_norm_3,label=np_known_train_labels,N_class=N_class1)
        end_time = time.clock()
        sum_time += (end_time-start_time)
        print('each training time',str(end_time-start_time),'sum time',str(sum_time))
        logging.info('each training time: {}\tsum time: {}'.format(str(end_time-start_time),str(sum_time)))
        # centroids = torch.load('CICIDS_centroids.pt')
        centroids = torch.load('CICIDS_centroids20200601.pt')
        # print('centroids',centroids)
        # print('max_threshold',max_threshold)
        list_known_data_label, list_known_data_pred = test_knowndata(model,centroids=centroids,max_threshold=max_threshold,data=np_known_test_norm_3,label=np_known_test_labels,N_class=N_class1)
        list_unknown_data_label, list_unknown_data_pred = test_novelty(model,centroids=centroids,max_threshold=max_threshold,data=np_unknown_test_norm_3,label=np_unknown_test_labels,N_class=N_class1)
        list_all_data_label = list_known_data_label +list_unknown_data_label
        list_all_data_pred = list_known_data_pred + list_unknown_data_pred
        # print(classification_report(list_all_data_label, list_all_data_pred))
        # print(confusion_matrix(list_all_data_label, list_all_data_pred))
        all_report = precision_recall_fscore_support(list_all_data_label, list_all_data_pred, average='weighted')
        all_precision = all_report[0]
        all_recall = all_report[1]
        all_fscore = all_report[2]
        # print('all_precision',all_precision,'all_recall',all_recall,'all_fscore',all_fscore)
        # # print(confusion_matrix(list_all_data_label, list_all_data_pred))
        # # print(classification_report(list_known_data_label, list_known_data_pred))
        known_report = precision_recall_fscore_support(list_known_data_label, list_known_data_pred, average='weighted')
        known_precision = known_report[0]
        known_recall = known_report[1]
        known_fscore = known_report[2]
        # print('known_precision',known_precision,'known_recall',known_recall,'known_fscore',known_fscore)
        # print(confusion_matrix(list_known_data_label, list_known_data_pred))
        # print(classification_report(list_unknown_data_label, list_unknown_data_pred))
        unknown_report = precision_recall_fscore_support(list_unknown_data_label, list_unknown_data_pred, average='weighted')
        unknown_precision = unknown_report[0]
        unknown_recall = unknown_report[1]
        unknown_fscore = unknown_report[2]
        # print('unknown_precision',unknown_precision,'unknown_recall',unknown_recall,'unknown_fscore',unknown_fscore)
        # logging.info('all_precision: {:.4f}\tall_recall: {:.4f}\tall_fscore: {:.4f}\tknown_precision: {:.4f}\tknown_recall: {:.4f}\tknown_fscore: {:.4f}\tunknown_precision: {:.4f}\tunknown_recall: {:.4f}\tunknown_fscore: {:.4f}'.format(all_precision,all_recall,all_fscore,unknown_precision,unknown_recall,unknown_fscore,known_precision,known_recall,known_fscore))
        if unknown_fscore>max_unknown_fscore:
            print('*********IMPROVED*********')
            max_all_precision = all_precision
            max_all_recall = all_recall
            max_all_fscore = all_fscore
            max_known_precision = known_precision
            max_known_recall = known_recall
            max_known_fscore = known_fscore
            max_unknown_precision = unknown_precision
            max_unknown_recall = unknown_recall
            max_unknown_fscore = unknown_fscore
            # OUR MODEL
            torch.save(model.state_dict(), save_model)
            #############
            # OUR MOMDEL
            # torch.save(centroids,'best_CICIDS_centroids_1121.pt')
            # torch.save(max_threshold,'best_CICIDS_maxthreshold_1121.pt')
            torch.save(centroids,'best_CICIDS_centroids_20200601.pt')
            torch.save(max_threshold,'best_CICIDS_maxthreshold_20200601.pt')
        print('max_all_fscore',max_all_fscore,'max_known_fscore',max_known_fscore,'max_unknown_fscore',max_unknown_fscore,'max_unknown_precision',max_unknown_precision,'max_unknown_recall',max_unknown_recall)
        logging.info('max_all_fscore: {:.4f}\tmax_all_precision: {:.4f}\tmax_all_recall: {:.4f}'.format(max_all_fscore,max_all_precision,max_all_recall))
        logging.info('max_known_fscore: {:.4f}\tmax_known_precision: {:.4f}\tmax_known_recall: {:.4f}'.format(max_known_fscore,max_known_precision,max_known_recall))
        logging.info('max_unknown_fscore: {:.4f}\tmax_unknown_precision: {:.4f}\tmax_unknown_recall: {:.4f}'.format(max_unknown_fscore,max_unknown_precision,max_unknown_recall))
    print('all training time',str(sum_time))

    ##Test without novelty detection (Closed-set classification)
    # epoch = 1
    # model = models.SharedCNN(N_class1).to(device)
    # model = load_model(model, model_path=pre_train_model)
    # list_unknown_data_label, list_unknown_data_pred = test_withoutnd(model,data=np_unknown_test_norm_3,label=np_unknown_test_labels)
    # list_known_data_label, list_known_data_pred = test(model, data=np_known_test_norm_3,label=np_known_test_labels)
    # list_all_data_label = list_known_data_label + list_unknown_data_label
    # list_all_data_pred = list_known_data_pred + list_unknown_data_pred
    # print(classification_report(list_all_data_label, list_all_data_pred))
    # print(confusion_matrix(list_all_data_label, list_all_data_pred))



# obtain pre-train model
# if __name__ == '__main__':
    # np_known_train,np_known_train_labels,np_known_test,np_known_test_labels = readdataset_known()
    # print('np_known_train',np_known_train.shape)
    # np_all = np.concatenate((np_known_train, np_known_test), axis=0)
    # min_max_scaler = MinMaxScaler()
    # min_max_scaler.fit(np_all)
    # np_known_train_norm = min_max_scaler.transform(np_known_train)
    # np_known_test_norm = min_max_scaler.transform(np_known_test)
    # # np_known_train_norm, np_known_train_labels, np_known_test_norm, np_known_test_labels = readdataset_known()
    # N_class1 = np.unique(np_known_train_labels).shape[0]
    # np_known_train_norm_3=np_known_train_norm.reshape(-1,1,16,16)
    # np_known_test_norm_3=np_known_test_norm.reshape(-1,1,16,16)
    # # np_known_train_norm_3 = np_known_train_norm[:, np.newaxis, :, :]
    # # np_known_test_norm_3 = np_known_test_norm[:, np.newaxis, :, :]
    # print('train data size', np_known_train_norm_3.shape, 'train label size', np_known_train_labels.shape, 'N_class',N_class1)
    # # OUR METHOD
    # # model = models.SharedCNN(N_class1).to(device)
    # correct = 0
    # time_sum = 0
    # for epoch in range(1, epochs + 1):
        # # OUR METHOD
        # # train_pre(epoch, model,data=np_known_train_norm_3,label=np_known_train_labels)
        # time0 = time.clock()
        # train_pre(epoch, model, data=np_known_train_norm_3, label=np_known_train_labels)
        # time1 = time.clock()
        # time_sum += (time1-time0)
        # print('epcoh',epoch,'time',time_sum)
    # test(model, data=np_known_test_norm_3, label=np_known_test_labels)
    # # OUR METHOD
    # # torch.save(model.state_dict(), pre_train_model)








