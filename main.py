import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import cnn as models
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
import logging
import math
from layers import SinkhornDistance
import mmd
import cal_metrics
import argparse
from collections import Iterable                            # < py38

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name == 0 :
            max_value = 58329
            min_value = 0
        elif feature_name == 4:
            # max_value = 1379963888
            max_value = 693375640
            min_value = 0
        elif feature_name == 5:
            # max_value = 1309937401
            max_value = 5203179
            min_value = 0
        elif feature_name == 9:
            max_value = 101
            min_value = 0
        elif feature_name == 10:
            max_value = 5
            min_value = 0
        elif feature_name == 12:
            max_value = 884
            min_value = 0
        elif feature_name == 15:
            max_value = 993
            min_value = 0
        elif feature_name == 16:
            max_value = 100
            min_value = 0
        elif feature_name == 17:
            max_value = 5
            min_value = 0
        elif feature_name == 18:
            max_value = 8
            min_value = 0
        elif feature_name == 20:
            max_value = 1
            min_value = 0
        else:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
        if max_value != min_value:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def readdataset(filename, dataset_number,unknown_map=None):
    raw_data_filename = filename
    raw_data = pd.read_csv(raw_data_filename, header=None)
    raw_data[1], protocols = pd.factorize(raw_data[1], sort=True)
    raw_data[3], flags = pd.factorize(raw_data[3], sort=True)
    if dataset_number == 0:
        # dataset_number = 0, train dataset known classes 5 NSL-KDD
        attacks_mapping = {'normal': 0, 'buffer_overflow': 70, 'loadmodule': 70, 'perl': 70, 'neptune': 1,
                           'smurf': 70, 'guess_passwd': 70, 'pod': 70, 'teardrop': 70, 'portsweep': 4,
                           'ipsweep': 3, 'land': 70, 'ftp_write': 70, 'back': 70, 'imap': 70, 'satan': 2, 'phf': 70,
                           'nmap': 70, 'multihop': 70, 'warezmaster': 70, 'warezclient': 70, 'spy': 70,
                           'rootkit': 70}
    elif dataset_number == 1:
        # dataset_number = 1, train dataset unknown classes（don't use it in our experiments）
        attacks_mapping = {'normal': 70, 'buffer_overflow': 70, 'loadmodule': 70, 'perl': 70, 'neptune': 70,
                           'smurf': 70, 'guess_passwd': 70, 'pod': 3, 'teardrop': 70, 'portsweep': 70,
                           'ipsweep': 70, 'land': 70, 'ftp_write': 70, 'back': 70, 'imap': 70, 'satan': 70, 'phf': 70,
                           'nmap': 70, 'multihop': 70, 'warezmaster': 70, 'warezclient': 70, 'spy': 70,
                           'rootkit': 70}
    elif dataset_number == 2:
        # # dataset_number = 2, teset dataset known dataset NSL-KDD
        attacks_mapping = {'normal': 0, 'snmpgetattack': 70, 'named': 70, 'xlock': 70, 'smurf': 70, 'ipsweep': 3,
                           'multihop': 70, 'xsnoop': 70, 'sendmail': 70, 'guess_passwd': 70, 'saint': 70,
                           'buffer_overflow': 70, 'portsweep': 4, 'pod': 70, 'apache2': 70, 'phf': 70,
                           'udpstorm': 70, 'warezmaster': 70, 'perl': 70, 'satan': 2, 'xterm': 70, 'mscan': 70,
                           'processtable': 70, 'ps': 70, 'nmap': 70, 'rootkit': 70, 'neptune': 1, 'loadmodule': 70,
                           'imap': 70, 'back': 70, 'httptunnel': 70, 'worm': 70, 'mailbomb': 70, 'ftp_write': 70,
                           'teardrop': 70, 'land': 70, 'sqlattack': 70, 'snmpguess': 70}
    else:
        # dataset_number = 3, test dataset unkown dataset
        attacks_mapping = unknown_map


    services_mapping = {'aol': 0, 'auth': 1, 'bgp': 2, 'courier': 3, 'csnet_ns': 4, 'ctf': 5, 'daytime': 6,'discard': 7,
                        'domain': 8, 'domain_u': 9, 'echo': 10, 'eco_i': 11, 'ecr_i': 12, 'efs': 13, 'exec': 14,'finger': 15, 'ftp': 16,
                        'ftp_data': 17, 'gopher': 18, 'harvest': 19, 'hostnames': 20, 'http': 21, 'http_2784': 22,'http_443': 23,
                        'http_8001': 24, 'imap4': 25, 'IRC': 26, 'iso_tsap': 27, 'klogin': 28, 'kshell': 29, 'ldap': 30,'link': 31,
                        'login': 32, 'mtp': 33, 'name': 34, 'netbios_dgm': 35, 'netbios_ns': 36, 'netbios_ssn': 37,'netstat': 38,
                        'nnsp': 39, 'nntp': 40, 'ntp_u': 41, 'other': 42, 'pm_dump': 43, 'pop_2': 44, 'pop_3': 45, 'printer': 46,
                        'private': 47, 'red_i': 48, 'remote_job': 49, 'rje': 50, 'shell': 51, 'smtp': 52, 'sql_net': 53, 'ssh': 54,
                        'sunrpc': 55, 'supdup': 56, 'systat': 57, 'telnet': 58, 'tftp_u': 59, 'tim_i': 60, 'time': 61,'urh_i': 62,
                        'urp_i': 63, 'uucp': 64, 'uucp_path': 65, 'vmnet': 66, 'whois': 67, 'X11': 68, 'Z39_50': 69}
    raw_data[2] = raw_data[2].map(services_mapping)
    raw_data[41] = raw_data[41].map(attacks_mapping)
    # NSL KDD
    features = raw_data.iloc[:, 0:raw_data.shape[1] - 2]
    labels = raw_data.iloc[:, raw_data.shape[1] - 2:raw_data.shape[1] - 1]
    # # KDD99
    # features = raw_data.iloc[:, 0:raw_data.shape[1] - 1]
    # labels = raw_data.iloc[:, raw_data.shape[1] - 1:]
    np_features = np.array(features)
    labels = labels.values.ravel()
    np_labels = np.array(labels)
    # print('np_labels',70 in np_labels)
    df = pd.DataFrame(np_features, index=np_labels)
    df_norm = normalize(df)
    df_drop = df_norm.drop(index=70)
    if dataset_number == 0:
        df_drop.sort_index(ascending=True, inplace=True)
        print('after sort', df_drop.index.values)
        # NSL-KDD 5 Classes
        a0 = df_drop.iloc[:6000]
        a1 = df_drop.iloc[67343:(67343 + 5000)]
        a2 = df_drop.iloc[(67343 + 41214):(67343 + 41214 + 3633)]
        a3 = df_drop.iloc[(67343 + 41214 + 3633):(67343 + 41214 + 3633 + 3599)]
        a4 = df_drop.iloc[(67343 + 41214 + 3633 + 3599):(67343 + 41214 + 3633 + 3599 + 2931)]
        df_drop = pd.concat([a0, a1, a2, a3, a4])
        # # KDD99 3 Classes
        # a0 = df_drop.iloc[0:97278]
        # a1 = df_drop.iloc[97278:(97278+107201)]
        # a2 = df_drop.iloc[(97278+107201):(97278+107201+100000)]
        # df_drop = pd.concat([a0, a1, a2])
    return df_drop


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
            kk = torch.tensor([col]).expand_as(label_correct)
            class_count[col,] = torch.eq(label_correct, kk.type(torch.LongTensor)).sum(0)
        positive_class_count = torch.max(class_count, torch.ones(class_count.size()))
        scatter_index = label_correct.expand(label_correct.size(0), N_class)
        centroid = torch.zeros(N_class, N_class).scatter_add(0,scatter_index.type(torch.LongTensor),pred_correct)
        mean_centroid = centroid / positive_class_count
        current_centroids = mean_centroid
        for i in range(0, mean_centroid.size(0)):
            if positive_class_count[i] == 1:
                current_centroids[i,] = last_centroids[i,]
            else:
                current_centroids[i,] = 0.5*last_centroids[i,]+0.5*current_centroids[i,]
    else:
        current_centroids = last_centroids
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
    return threshold

def inter_distance(A):
    s0 = int(A.size(0))
    s1 = int(A.size(1))
    # unsqueeze 指定维度上进行扩充1维
    aa = A.unsqueeze(0).expand(s0,s0,s1).contiguous().view(-1,s1)
    aat = A.unsqueeze(1).expand(s0,s0,s1).contiguous().view(-1,s1)
    dist = ((aa-aat)**2).sum(1).contiguous().view(s0,s0)
    # print('inter_centroid: {:.4f}',dist)
    return dist

def fisher_loss(pred,centroid):
    class_index = torch.argmax(pred,1).view(-1).type(torch.LongTensor)
    sum_Sw = 0
    for i in range(pred.size(0)):    
        sum_Sw += ((pred[i,]-centroid[class_index[i],])**2).sum()
    Sw = sum_Sw/pred.size(0)
    sum_Sb = inter_distance(centroid).sum()
    Sb = sum_Sb/(centroid.size(0)*(centroid.size(0)-1))
    return Sw,Sb


def train(epoch,model,rank_rate,max_threshold):
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)
    correct = 0
    loss = torch.nn.CrossEntropyLoss()
    model.train()
    train_loader,N_class = get_source_loader(train_filename,dataset_number=0)
    len_train_dataset = len(train_loader.dataset)
    len_train_loader = len(train_loader)
    iter_train = iter(train_loader)
    num_iter = len_train_loader
    last_centroids = torch.load('centroids.pt')
    for i in range(1, num_iter):
        data_train, label_train = iter_train.next()
        data_train, label_train = Variable(data_train).float().to(device),Variable(label_train).type(torch.LongTensor).to(device)
        with torch.no_grad():
            last_centroids = Variable(last_centroids)
        optimizer.zero_grad()
        train_av,train_pred = model(data_train)
        centroid = cal_centroid(label_train.cpu(),train_av.cpu(),N_class,last_centroids)
        last_centroids.data = centroid
        Sw,Sb = fisher_loss(pred=train_av.cpu(),centroid=centroid)
        threshold = cal_threshold(label=label_train.cpu(), pred=train_av.cpu(), centroid=centroid, rank_rate=rank_rate)
        cross_loss = loss(train_pred,label_train)
        fisherloss = (lamda.to(device))*((Sw.to(device))-(alpha.to(device))*(Sb.to(device)))
        Loss = cross_loss+ fisherloss
        pred = train_pred.max(1)[1]
        correct += pred.eq(label_train.data.view_as(pred)).cpu().sum()
        Loss.backward()
        optimizer.step()
    for m in range(threshold.size(0)):
        if threshold[m]>max_threshold[m]:
            max_threshold[m]=threshold[m]  
    torch.save(last_centroids.data,'centroids.pt')
    Accuracy = 100. * correct.type(torch.FloatTensor) / len_train_dataset
    print('Train Epoch:{}\tLoss: {:.6f}\tcross_loss: {:.6f}\tfisherloss: {:.6f}\tSw: {:.6f}\tSb: {:.6f}\tAccuracy: {:.4f}'.format(epoch,Loss,cross_loss,fisherloss,Sw,Sb,Accuracy))
    logging.info('Train Epoch:{}\tLoss: {:.6f}\tcross_loss: {:.6f}\tfisherloss: {:.6f}\tSw: {:.6f}\tSb: {:.6f}\tAccuracy: {:.4f}'.format(epoch,Loss,cross_loss,fisherloss,Sw,Sb,Accuracy))
    return max_threshold


# test the known data based on the max probabality
def test(model):
    correct = 0
    test_loss = 0
    loss = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loader,N_class = get_source_loader(test_filename,dataset_number=1)
    len_test_dataset = len(test_loader.dataset)
    for data_test,label_test in test_loader:
        data_test, label_test = Variable(data_test).float().to(device),Variable(label_test).type(torch.LongTensor).to(device)
        test_av,test_pred = model(data_test)
        test_loss += loss(test_pred, label_test)
        pred = test_pred.max(1)[1]
        correct += pred.eq(label_test.data.view_as(pred)).cpu().sum()
    Accuracy = 100. * correct.type(torch.FloatTensor) / len_test_dataset
    test_loss /= len_test_dataset
    print('Train Epoch:{}\tLoss: {:.6f}\tAccuracy: {:.4f}'.format(epoch, test_loss, Accuracy))


# class incremental -> calculate the new class centroid
def cal_new_class_centroid(model):
    model.eval()
    test_loader, N_class = get_source_loader(test_filename, dataset_number=2)
    sum_new_class_centroid = torch.zeros(1,N_class1)
    txt_file = open('snmpguess.csv','ab')
    for data_test, label_test in test_loader:
        data_test, label_test = Variable(data_test).float().to(device), Variable(label_test).type(torch.LongTensor).to(
            device)
        test_av, test_pred, _, _ = model(data_test, data_test)
        np.savetxt(txt_file,test_av.cpu().detach().numpy(),fmt=['%.4f','%.4f','%.4f'],delimiter=',')
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
    test_loader, N_class = get_source_loader(test_filename, dataset_number=2)
    len_test_dataset = len(test_loader.dataset)
    for data_test, label_test in test_loader:
        data_test, label_test = Variable(data_test).float().to(device), Variable(label_test).type(torch.LongTensor).to(
            device)
        test_av, test_pred, _, _ = model(data_test, data_test)
        _, pred = cal_min_dis_to_centroid(test_av.cpu(),centroid)
        new_class_correct += torch.eq(pred,num_classes).sum(0)
        num_label_new_class += torch.eq(label_test.cpu(),num_classes).sum(0)
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
    model.eval()
    test_loader = get_source_loader(data,label)
    len_test_dataset = len(test_loader.dataset)
    # txt_file0 = open(known_pred_filename, 'ab')
    # txt_file1 = open(known_label_filename, 'ab')
    # txt_file2 = open(known_data_filename, 'ab')
    # txt_file = open(known_score_filename, 'ab')
    for data_test,label_test in test_loader:
        data_test, label_test = Variable(data_test).float().to(device),Variable(label_test).type(torch.LongTensor).to(device)
        test_av,test_pred, _, _ = model(data_test,data_test)
        test_loss += loss(test_pred, label_test)
        pred = test_pred.max(1)[1]
        dist_to_its_centriod, min_dist_class_index = cal_min_dis_to_centroid(pred=test_av.cpu(), centroid=centroids)
        # np.savetxt(txt_file1, label_test.cpu().detach().numpy(), fmt=['%d'], delimiter=',')
        # np.savetxt(txt_file2, test_av.cpu().detach().numpy(), fmt=['%.4f', '%.4f', '%.4f', '%.4f', '%.4f'], delimiter=',')
        # np.savetxt(txt_file, dist_to_its_centriod.detach().numpy(), fmt=['%.4f'], delimiter=',')
        # # np.savetxt(known_data_filename, dist_to_its_centriod.detach().numpy(), fmt=['%.4f'], delimiter=',')
        for i in range(dist_to_its_centriod.size(0)):
            # pred[i]=min_dist_class_index[i]
            if dist_to_its_centriod[i] > max_threshold[min_dist_class_index[i]]:
                pred[i]=N_class
                novelty += 1
        # np.savetxt(txt_file0, pred.cpu().detach().numpy(), fmt=['%d'], delimiter=',')
        correct += pred.eq(label_test.data.view_as(pred)).cpu().sum()
        known_data_pred.append(pred.cpu().detach().data.tolist())
        known_data_label.append(label_test.cpu().detach().data.tolist())
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
    txt_file = open(unknown_score_filename, 'ab')
    sum_new_class_centroid = torch.zeros(1, N_class1)
    for data_test,label_test in test_loader:
        data_test, label_test = Variable(data_test).float().to(device),Variable(label_test).type(torch.LongTensor).to(device)
        test_av, test_pred, _, _ = model(data_test, data_test)
        pred = test_pred.max(1)[1]
        # np.savetxt(txt_file, test_av.cpu().detach().numpy(), fmt=['%.4f','%.4f','%.4f','%.4f','%.4f'], delimiter=',')
        dist_to_its_centriod,min_dist_class_index=cal_min_dis_to_centroid(pred=test_av.cpu(), centroid=centroids)
        # np.savetxt(txt_file1, label_test.cpu().detach().numpy(), fmt=['%d'], delimiter=',')
        # np.savetxt(txt_file2, test_av.cpu().detach().numpy(), fmt=['%.4f', '%.4f', '%.4f','%.4f','%.4f'], delimiter=',')
        np.savetxt(txt_file, dist_to_its_centriod.detach().numpy(), fmt=['%.4f'], delimiter=',')
        sum = (test_av.cpu().sum(0) / (test_av.cpu().size(0))).view(1, N_class1)
        sum_new_class_centroid += sum
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
    txt_file.close()
    new_class_centroid = sum_new_class_centroid / len(test_loader)
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



# test the known data using ODIN AND BASELINE
def test_odin_knowndata(model,magnitude,temperature,data,label):
    loss = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loader = get_source_loader(data, label)
    len_test_dataset = len(test_loader.dataset)
    # txt_file = open(odin_known_score_filename, 'ab')
    for data_test,label_test in test_loader:
        data_test, label_test = Variable(data_test, requires_grad=True).float().to(device),Variable(label_test).type(torch.LongTensor).to(device)
        test_av,test_pred = model(data_test)
        test_pred_t = test_pred/temperature
        labels = test_pred_t.data.max(1)[1]
        labels = Variable(labels)
        Loss = loss(test_pred_t,labels)
        C = torch.autograd.grad(Loss, data_test)
        gradient = torch.ge(C[0],0)
        gradient = (gradient.float()-0.5)*2
        tempInputs = torch.add(data_test.data, -magnitude, gradient)
        with torch.no_grad():
            tempInputs = Variable(tempInputs)
        outputs,_ = model(tempInputs)
        outputs = outputs/temperature
        soft_out = F.softmax(outputs,dim=1)
        soft_out, _ = torch.max(soft_out.data,dim=1)
        # np.savetxt(txt_file, soft_out.cpu().detach().numpy(), fmt=['%.4f'], delimiter=',')

# test the unknown data using ODIN AND BASELINE
def test_odin_novelty(model,magnitude,temperature,data,label):
    loss = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loader = get_source_loader(data, label)
    len_test_dataset = len(test_loader.dataset)
    txt_file = open(odin_unknown_score_filename, 'ab')
    for data_test,label_test in test_loader:
        data_test, label_test = Variable(data_test, requires_grad=True).float().to(device),Variable(label_test).type(torch.LongTensor)
        test_av, test_pred = model(data_test)
        test_pred_t = test_av / temperature
        labels = test_pred_t.data.max(1)[1]
        labels = Variable(labels)
        Loss = loss(test_pred_t, labels)
        C = torch.autograd.grad(Loss, data_test)
        gradient = torch.ge(C[0], 0)
        gradient = (gradient.float()-0.5)*2
        tempInputs = torch.add(data_test.data, -magnitude, gradient)
        with torch.no_grad():
            tempInputs = Variable(tempInputs)
        outputs,_ = model(tempInputs)
        outputs = outputs/temperature
        soft_out = F.softmax(outputs,dim=1)
        soft_out, _ = torch.max(soft_out.data,dim=1)
        np.savetxt(txt_file, soft_out.cpu().detach().numpy(), fmt=['%.4f'], delimiter=',')
    txt_file.close()

# pretrain model using ODIN AND BASELINE
def train_odin(epoch,model,data,label):
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)
    # optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    correct = 0
    loss = torch.nn.CrossEntropyLoss()
    model.train()
    train_loader = get_source_loader(data,label)
    len_train_dataset = len(train_loader.dataset)
    len_train_loader = len(train_loader)
    iter_train = iter(train_loader)
    num_iter = len_train_loader
    for i in range(1, num_iter):
        data_train, label_train = iter_train.next()
        data_train, label_train = Variable(data_train).float().to(device),Variable(label_train).type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        train_av,train_pred = model(data_train)
        Loss = loss(train_pred,label_train)
        pred = train_pred.max(1)[1]
        correct += pred.eq(label_train.data.view_as(pred)).cpu().sum()
        Loss.backward()
        optimizer.step()
    Accuracy = 100. * correct.type(torch.FloatTensor) / len_train_dataset
    print('Train Epoch:{}\tLoss: {:.6f}\tAccuracy: {:.4f}'.format(epoch, Loss, Accuracy))
    logging.info('Train Epoch:{}\tLoss: {:.6f}\tAccuracy: {:.4f}'.format(epoch, Loss, Accuracy))

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
        train_av, train_pred, _, _ = model(data_train,data_train)
        Loss = loss(train_pred, label_train)
        pred = train_pred.max(1)[1]
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
BATCH_SIZE = 256
LEARNING_RATE = 0.001
momentum = 0.9
l2_decay = 5e-4
# NSL-KDD Dataset
train_filename = "./NSL-KDD/KDDTrain+.txt"
test_filename = "./NSL-KDD/KDDTest+.txt"
pre_train_model = 'pretrain_model_NSL_5_0703.pkl'
# OUR METHOD
save_model = 'save_model_NSL_5_0703.pkl'


# NSL-KDD train 5 classes
pre_train_model = 'pretrain_model_NSL_5_0626.pkl'
save_model = 'save_model_NSL_5_0626.pkl'
# NSL-KDD train 5 calsses ODIN AND BASELINE
# pre_train_model = 'pretrain_model_odin_NSL_5_0626.pkl'
# save_model = 'save_model_odin_NSL_5_0626.pkl'
# # KDD99
# train_filename = "./kddcup_data_10_percent_corrected.csv"
# test_filename = "./corrected.csv"
# KDD99 ODIN AND BASELINE
# pre_train_model = 'pretrain_model_odin_KDD_3_0626.pkl'
# KDD99 OURS METHODS
# pre_train_model = 'pretrain_model_KDD_3_0702.pkl'
# save_model = 'save_model_KDD_satan_waraz_0628.pkl'
epochs = 200



# obtain pre-train model
# if __name__ == '__main__':
#     df_known_train = readdataset(train_filename, dataset_number=0)
#     np_known_train = df_known_train.values
#     # min_max_scaler = MinMaxScaler()
#     # np_known_train_norm = min_max_scaler.fit_transform(np_known_train)
#     np_known_train_norm_3 = np_known_train[:, np.newaxis, :]
#     np_known_train_labels = df_known_train.index.values
#     N_class1 = np.unique(np_known_train_labels).shape[0]
#     print('train data size', np_known_train_norm_3.shape, 'train label size', np_known_train_labels.shape, 'N_class',N_class1)
#     model = models.SharedCNN(N_class1).to(device)
#     # BASELINE AND OIDN
#     # model = models.CNN(N_class1).to(device)
#     correct = 0
#     for epoch in range(1, epochs + 1):
#         train_pre(epoch, model,data=np_known_train_norm_3,label=np_known_train_labels)
#         # BASELINE AND OIDN
#         # train_odin(epoch, model,data=np_known_train_norm_3,label=np_known_train_labels)
#     torch.save(model.state_dict(), pre_train_model)



# train model with fisher loss + MMD loss
if __name__ == '__main__':
    # OUR METHOD
    logging.basicConfig(filename='20190711_NSLKDD_results.log', level=logging.DEBUG)
    # define the save file
    score_data_dir = "./"
    # known_filename = 'score_knowdata_NSLKDD_5_0626.csv'
    # unknown_filename = 'score_unknowdata_NSLKDD_0626.csv'
    # known_data_filename = 'centroid_knowdata_NSLKDD_0705.csv'
    # known_label_filename = 'label_known_NSLKDD_0705.csv'
    # unknown_centroid_filename = 'centroid_unknowdata_NSLKDD_smurf_0703_our.csv'

    known_pred_filename = 'pred_known_NSLKDD_0710.csv'
    known_data_filename = 'centroid_knowdata_NSLKDD_0710.csv'
    known_label_filename = 'label_known_NSLKDD_0710.csv'
    unknown_pred_filename = 'pred_unknown_NSLKDD_processtable_0710.csv'
    unknown_data_filename = 'centroid_unknowdata_NSLKDD_processtable_0710.csv'
    unknown_label_filename = 'label_unknown_NSLKDD_processtable_0710.csv'
    parser = argparse.ArgumentParser()
    parser.add_argument('--unknow_attack', type=str, nargs='?', default='mulunkown', help="attack name")
    args = parser.parse_args()
    logging.info(args.unknow_attack)
    known_score_filename = 'score_knowdata_NSLKDD_0711.csv'
    unknown_score_filename = 'score_unknowdata_NSLKDD_' + args.unknow_attack + '0711.csv'
    if args.unknow_attack == 'guess_passwd':
        unknown_attacks_mapping = {'normal': 70, 'snmpgetattack': 70, 'named': 70, 'xlock': 70, 'smurf': 70, 'ipsweep': 70,
                       'multihop': 70, 'xsnoop': 70, 'sendmail': 70, 'guess_passwd': 5, 'saint': 70,
                       'buffer_overflow': 70, 'portsweep': 70, 'pod': 70, 'apache2': 70, 'phf': 70,
                       'udpstorm': 70, 'warezmaster': 70, 'perl': 70, 'satan': 70, 'xterm': 70, 'mscan': 70,
                       'processtable': 70, 'ps': 70, 'nmap': 70, 'rootkit': 70, 'neptune': 70,'loadmodule': 70,
                       'imap': 70, 'back': 70, 'httptunnel': 70, 'worm': 70, 'mailbomb': 70, 'ftp_write': 70,
                       'teardrop': 70, 'land': 70, 'sqlattack': 70, 'snmpguess': 70}
    elif args.unknow_attack == 'mscan':
        unknown_attacks_mapping = {'normal': 70, 'snmpgetattack': 70, 'named': 70, 'xlock': 70, 'smurf': 70, 'ipsweep': 70,
                   'multihop': 70, 'xsnoop': 70, 'sendmail': 70, 'guess_passwd': 70, 'saint': 70,
                   'buffer_overflow': 70, 'portsweep': 70, 'pod': 70, 'apache2': 70, 'phf': 70,
                   'udpstorm': 70, 'warezmaster': 70, 'perl': 70, 'satan': 70, 'xterm': 70, 'mscan': 5,
                   'processtable': 70, 'ps': 70, 'nmap': 70, 'rootkit': 70, 'neptune': 70,'loadmodule': 70,
                   'imap': 70, 'back': 70, 'httptunnel': 70, 'worm': 70, 'mailbomb': 70, 'ftp_write': 70,
                   'teardrop': 70, 'land': 70, 'sqlattack': 70, 'snmpguess': 70}
    elif args.unknow_attack == 'warezmaster':
        unknown_attacks_mapping = {'normal': 70, 'snmpgetattack': 70, 'named': 70, 'xlock': 70, 'smurf': 70, 'ipsweep': 70,
                   'multihop': 70, 'xsnoop': 70, 'sendmail': 70, 'guess_passwd': 70, 'saint': 70,
                   'buffer_overflow': 70, 'portsweep': 70, 'pod': 70, 'apache2': 70, 'phf': 70,
                   'udpstorm': 70, 'warezmaster': 5, 'perl': 70, 'satan': 70, 'xterm': 70, 'mscan': 70,
                   'processtable': 70, 'ps': 70, 'nmap': 70, 'rootkit': 70, 'neptune': 70,'loadmodule': 70,
                   'imap': 70, 'back': 70, 'httptunnel': 70, 'worm': 70, 'mailbomb': 70, 'ftp_write': 70,
                   'teardrop': 70, 'land': 70, 'sqlattack': 70, 'snmpguess': 70}
    elif args.unknow_attack == 'apache2':
        unknown_attacks_mapping = {'normal': 70, 'snmpgetattack': 70, 'named': 70, 'xlock': 70, 'smurf': 70, 'ipsweep': 70,
                   'multihop': 70, 'xsnoop': 70, 'sendmail': 70, 'guess_passwd': 70, 'saint': 70,
                   'buffer_overflow': 70, 'portsweep': 70, 'pod': 70, 'apache2': 5, 'phf': 70,
                   'udpstorm': 70, 'warezmaster': 70, 'perl': 70, 'satan': 70, 'xterm': 70, 'mscan': 70,
                   'processtable': 70, 'ps': 70, 'nmap': 70, 'rootkit': 70, 'neptune': 70,'loadmodule': 70,
                   'imap': 70, 'back': 70, 'httptunnel': 70, 'worm': 70, 'mailbomb': 70, 'ftp_write': 70,
                   'teardrop': 70, 'land': 70, 'sqlattack': 70, 'snmpguess': 70}
    elif args.unknow_attack == 'processtable':
        unknown_attacks_mapping = {'normal': 70, 'snmpgetattack': 70, 'named': 70, 'xlock': 70, 'smurf': 70, 'ipsweep': 70,
                   'multihop': 70, 'xsnoop': 70, 'sendmail': 70, 'guess_passwd': 70, 'saint': 70,
                   'buffer_overflow': 70, 'portsweep': 70, 'pod': 70, 'apache2': 70, 'phf': 70,
                   'udpstorm': 70, 'warezmaster': 70, 'perl': 70, 'satan': 70, 'xterm': 70, 'mscan': 70,
                   'processtable': 5, 'ps': 70, 'nmap': 70, 'rootkit': 70, 'neptune': 70,'loadmodule': 70,
                   'imap': 70, 'back': 70, 'httptunnel': 70, 'worm': 70, 'mailbomb': 70, 'ftp_write': 70,
                   'teardrop': 70, 'land': 70, 'sqlattack': 70, 'snmpguess': 70}
    elif args.unknow_attack == 'smurf':
        unknown_attacks_mapping = {'normal': 70, 'snmpgetattack': 70, 'named': 70, 'xlock': 70, 'smurf': 5, 'ipsweep': 70,
                   'multihop': 70, 'xsnoop': 70, 'sendmail': 70, 'guess_passwd': 70, 'saint': 70,
                   'buffer_overflow': 70, 'portsweep': 70, 'pod': 70, 'apache2': 70, 'phf': 70,
                   'udpstorm': 70, 'warezmaster': 70, 'perl': 70, 'satan': 70, 'xterm': 70, 'mscan': 70,
                   'processtable': 70, 'ps': 70, 'nmap': 70, 'rootkit': 70, 'neptune': 70,'loadmodule': 70,
                   'imap': 70, 'back': 70, 'httptunnel': 70, 'worm': 70, 'mailbomb': 70, 'ftp_write': 70,
                   'teardrop': 70, 'land': 70, 'sqlattack': 70, 'snmpguess': 70}
    elif args.unknow_attack == 'back':
        unknown_attacks_mapping = {'normal': 70, 'snmpgetattack': 70, 'named': 70, 'xlock': 70, 'smurf': 70, 'ipsweep': 70,
                   'multihop': 70, 'xsnoop': 70, 'sendmail': 70, 'guess_passwd': 70, 'saint': 70,
                   'buffer_overflow': 70, 'portsweep': 70, 'pod': 70, 'apache2': 70, 'phf': 70,
                   'udpstorm': 70, 'warezmaster': 70, 'perl': 70, 'satan': 70, 'xterm': 70, 'mscan': 70,
                   'processtable': 70, 'ps': 70, 'nmap': 70, 'rootkit': 70, 'neptune': 70,'loadmodule': 70,
                   'imap': 70, 'back': 5, 'httptunnel': 70, 'worm': 70, 'mailbomb': 70, 'ftp_write': 70,
                   'teardrop': 70, 'land': 70, 'sqlattack': 70, 'snmpguess': 70}
    elif args.unknow_attack == 'snmpguess':
        unknown_attacks_mapping = {'normal': 70, 'snmpgetattack': 70, 'named': 70, 'xlock': 70, 'smurf': 70, 'ipsweep': 70,
                   'multihop': 70, 'xsnoop': 70, 'sendmail': 70, 'guess_passwd': 70, 'saint': 70,
                   'buffer_overflow': 70, 'portsweep': 70, 'pod': 70, 'apache2': 70, 'phf': 70,
                   'udpstorm': 70, 'warezmaster': 70, 'perl': 70, 'satan': 70, 'xterm': 70, 'mscan': 70,
                   'processtable': 70, 'ps': 70, 'nmap': 70, 'rootkit': 70, 'neptune': 70,'loadmodule': 70,
                   'imap': 70, 'back': 70, 'httptunnel': 70, 'worm': 70, 'mailbomb': 70, 'ftp_write': 70,
                   'teardrop': 70, 'land': 70, 'sqlattack': 70, 'snmpguess': 5}
    elif args.unknow_attack == 'saint':
        unknown_attacks_mapping = {'normal': 70, 'snmpgetattack': 70, 'named': 70, 'xlock': 70, 'smurf': 70, 'ipsweep': 70,
                   'multihop': 70, 'xsnoop': 70, 'sendmail': 70, 'guess_passwd': 70, 'saint': 5,
                   'buffer_overflow': 70, 'portsweep': 70, 'pod': 70, 'apache2': 70, 'phf': 70,
                   'udpstorm': 70, 'warezmaster': 70, 'perl': 70, 'satan': 70, 'xterm': 70, 'mscan': 70,
                   'processtable': 70, 'ps': 70, 'nmap': 70, 'rootkit': 70, 'neptune': 70,'loadmodule': 70,
                   'imap': 70, 'back': 70, 'httptunnel': 70, 'worm': 70, 'mailbomb': 70, 'ftp_write': 70,
                   'teardrop': 70, 'land': 70, 'sqlattack': 70, 'snmpguess': 70}
    elif args.unknow_attack == 'mailbomb':
        unknown_attacks_mapping = {'normal': 70, 'snmpgetattack': 70, 'named': 70, 'xlock': 70, 'smurf': 70, 'ipsweep': 70,
                   'multihop': 70, 'xsnoop': 70, 'sendmail': 70, 'guess_passwd': 70, 'saint': 70,
                   'buffer_overflow': 70, 'portsweep': 70, 'pod': 70, 'apache2': 70, 'phf': 70,
                   'udpstorm': 70, 'warezmaster': 70, 'perl': 70, 'satan': 70, 'xterm': 70, 'mscan': 70,
                   'processtable': 70, 'ps': 70, 'nmap': 70, 'rootkit': 70, 'neptune': 70,'loadmodule': 70,
                   'imap': 70, 'back': 70, 'httptunnel': 70, 'worm': 70, 'mailbomb': 5, 'ftp_write': 70,
                   'teardrop': 70, 'land': 70, 'sqlattack': 70, 'snmpguess': 70}
    elif args.unknow_attack == 'snmpgetattack':
        unknown_attacks_mapping = {'normal': 70, 'snmpgetattack': 5, 'named': 70, 'xlock': 70, 'smurf': 70, 'ipsweep': 70,
                   'multihop': 70, 'xsnoop': 70, 'sendmail': 70, 'guess_passwd': 70, 'saint': 70,
                   'buffer_overflow': 70, 'portsweep': 70, 'pod': 70, 'apache2': 70, 'phf': 70,
                   'udpstorm': 70, 'warezmaster': 70, 'perl': 70, 'satan': 70, 'xterm': 70, 'mscan': 70,
                   'processtable': 70, 'ps': 70, 'nmap': 70, 'rootkit': 70, 'neptune': 70,'loadmodule': 70,
                   'imap': 70, 'back': 70, 'httptunnel': 70, 'worm': 70, 'mailbomb': 70, 'ftp_write': 70,
                   'teardrop': 70, 'land': 70, 'sqlattack': 70, 'snmpguess': 70}
    elif args.unknow_attack == 'httptunnel':
        unknown_attacks_mapping = {'normal': 70, 'snmpgetattack': 70, 'named': 70, 'xlock': 70, 'smurf': 70, 'ipsweep': 70,
                   'multihop': 70, 'xsnoop': 70, 'sendmail': 70, 'guess_passwd': 70, 'saint': 70,
                   'buffer_overflow': 70, 'portsweep': 70, 'pod': 70, 'apache2': 70, 'phf': 70,
                   'udpstorm': 70, 'warezmaster': 70, 'perl': 70, 'satan': 70, 'xterm': 70, 'mscan': 70,
                   'processtable': 70, 'ps': 70, 'nmap': 70, 'rootkit': 70, 'neptune': 70,'loadmodule': 70,
                   'imap': 70, 'back': 70, 'httptunnel': 5, 'worm': 70, 'mailbomb': 70, 'ftp_write': 70,
                   'teardrop': 70, 'land': 70, 'sqlattack': 70, 'snmpguess': 70}
    elif args.unknow_attack == 'mulunkown':
        unknown_attacks_mapping = {'normal': 0, 'snmpgetattack': 70, 'named': 70, 'xlock': 70, 'smurf': 70, 'ipsweep': 3,
                   'multihop': 70, 'xsnoop': 70, 'sendmail': 70, 'guess_passwd': 70, 'saint': 70,
                   'buffer_overflow': 70, 'portsweep': 4, 'pod': 70, 'apache2': 70, 'phf': 70,
                   'udpstorm': 70, 'warezmaster': 70, 'perl': 70, 'satan': 2, 'xterm': 70, 'mscan': 70,
                   'processtable': 70, 'ps': 70, 'nmap': 70, 'rootkit': 70, 'neptune': 1,'loadmodule': 70,
                   'imap': 70, 'back': 70, 'httptunnel': 70, 'worm': 70, 'mailbomb': 70, 'ftp_write': 70,
                   'teardrop': 70, 'land': 70, 'sqlattack': 70, 'snmpguess': 70}
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
    df_known_train = readdataset(train_filename, dataset_number=0)
    df_known_test = readdataset(test_filename, dataset_number=2)
    df_unknown_test = readdataset(test_filename, dataset_number=3,unknown_map=unknown_attacks_mapping)
    np_known_train = df_known_train.values
    np_known_test = df_known_test.values
    print('np_known_test',np_known_test.shape)
    np_unknown_test = df_unknown_test.values
    # labels
    np_known_train_labels = df_known_train.index.values
    print('np_known_train_labels',np_known_train_labels.shape)
    np_known_test_labels = (df_known_test.index.values).ravel()
    print('np_known_test_labels',np_known_test_labels.shape)
    np_unknown_test_labels = (df_unknown_test.index.values).ravel()
    N_class1 = np.unique(np_known_train_labels).shape[0]
    np_known_train_norm_3 = np_known_train[:, np.newaxis, :]
    np_known_test_norm_3 = np_known_test[:, np.newaxis, :]
    np_unknown_test_norm_3 = np_unknown_test[:, np.newaxis, :]
    print('train data size',np_known_train_norm_3.shape,'train label size',np_known_train_labels.shape,'N_class',N_class1)

    # # OUR METHOD Train the model
    model = models.SharedCNN(N_class1).to(device)
    model = load_model(model,model_path=pre_train_model)
    correct = 0
    centroids_zeros = torch.zeros(N_class1,N_class1)
    torch.save(centroids_zeros,'NSL_centroids.pt')
    # max_threshold = torch.zeros(N_class1)
    for epoch in range(1, epochs+1):
        lamda = torch.tensor(0.001*(math.exp(-5*(epoch/epochs))))
        alpha = torch.tensor(0.00001*(math.exp(-5*(epoch/epochs))))
        beta = torch.tensor(0.01)      
        print('lamda',lamda,'alpha',alpha)
        max_threshold = torch.zeros(N_class1)
        max_threshold=train_sharedcnn(epoch,model,rank_rate=0.97,max_threshold=max_threshold,data=np_known_train_norm_3,label=np_known_train_labels,N_class=N_class1)
        centroids = torch.load('NSL_centroids.pt')
        print('centroids',centroids)
        print('max_threshold',max_threshold)
    # torch.save(model.state_dict(), save_model)
    #  Test the model  
        list_known_data_label, list_known_data_pred = test_knowndata(model,centroids=centroids,max_threshold=max_threshold,data=np_known_test_norm_3,label=np_known_test_labels,N_class=N_class1)
        list_unknown_data_label, list_unknown_data_pred = test_novelty(model,centroids=centroids,max_threshold=max_threshold,data=np_unknown_test_norm_3,label=np_unknown_test_labels,N_class=N_class1)
        list_all_data_label = list_known_data_label +list_unknown_data_label
        list_all_data_pred = list_known_data_pred + list_unknown_data_pred
        # print(classification_report(list_all_data_label, list_all_data_pred))
        all_report = precision_recall_fscore_support(list_all_data_label, list_all_data_pred, average='weighted')
        all_precision = all_report[0]
        all_recall = all_report[1]
        all_fscore = all_report[2]
        print('all_precision',all_precision,'all_recall',all_recall,'all_fscore',all_fscore)
        # print(classification_report(list_known_data_label, list_known_data_pred))
        known_report = precision_recall_fscore_support(list_known_data_label, list_known_data_pred, average='weighted')
        known_precision = known_report[0]
        known_recall = known_report[1]
        known_fscore = known_report[2]
        print('known_precision',known_precision,'known_recall',known_recall,'known_fscore',known_fscore)
        # print(classification_report(list_unknown_data_label, list_unknown_data_pred))
        unknown_report = precision_recall_fscore_support(list_unknown_data_label, list_unknown_data_pred, average='weighted')
        unknown_precision = unknown_report[0]
        unknown_recall = unknown_report[1]
        unknown_fscore = unknown_report[2]
        print('unknown_precision',unknown_precision,'unknown_recall',unknown_recall,'unknown_fscore',unknown_fscore)
        logging.info('all_precision: {:.4f}\tall_recall: {:.4f}\tall_fscore: {:.4f}\tknown_precision: {:.4f}\tknown_recall: {:.4f}\tknown_fscore: {:.4f}\tunknown_precision: {:.4f}\tunknown_recall: {:.4f}\tunknown_fscore: {:.4f}'.format(all_precision,all_recall,all_fscore,unknown_precision,unknown_recall,unknown_fscore,known_precision,known_recall,known_fscore))
        if (unknown_fscore*all_fscore)>(max_unknown_fscore*max_all_fscore):
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
            torch.save(model.state_dict(), save_model)
            # OUR MEHTOD
            torch.save(centroids,'best_NSLKDD_centroids.pt')
            torch.save(max_threshold,'best_NSLKDD_maxthreshold.pt')
        print('max_all_fscore',max_all_fscore,'max_known_fscore',max_known_fscore,'max_unknown_fscore',max_unknown_fscore,'max_unknown_precision',max_unknown_precision,'max_unknown_recall',max_unknown_recall)
        logging.info('max_all_fscore: {:.4f}\tmax_all_precision: {:.4f}\tmax_all_recall: {:.4f}'.format(max_all_fscore,max_all_precision,max_all_recall))
        logging.info('max_known_fscore: {:.4f}\tmax_known_precision: {:.4f}\tmax_known_recall: {:.4f}'.format(max_known_fscore,max_known_precision,max_known_recall))
        logging.info('max_unknown_fscore: {:.4f}\tmax_unknown_precision: {:.4f}\tmax_unknown_recall: {:.4f}'.format(max_unknown_fscore,max_unknown_precision,max_unknown_recall))


    

# class incremental learning
# if __name__ == '__main__':
#     logging.basicConfig(filename='class_incremental_20190705.log', level=logging.DEBUG)
#     epoch = 1
#     # NSLKDD
#     base_class_centroids = torch.tensor([[  7.0226, -13.6010,  -6.7026,  -8.4695, -12.3190],
#         [ -5.6015,   5.9368,  -6.6947, -14.7424,  -4.9715],
#         [ -4.9779,  -6.1813,   4.6343, -11.1115,  -8.0881],
#         [ -2.9125, -17.5042,  -4.0337,   4.4102,  -3.4950],
#         [ -8.8935,  -3.9197,  -6.8907,  -3.3280,   4.8506]])
#     # cluster centroid
#  #    new_class_centroid = torch.tensor([[  3.41407417, -15.85182372,  -2.46781377,  -1.56851196,  -7.76121729],#]])
#  # [ -1.50232517,  -5.93046054, -14.73713415,  -9.82550612,  -5.1173166], #]])
#  # [ 7.31040878,  -2.83502136,  -8.30029621, -20.19710998, -16.70097625],#]])
#  # [ -4.98204205,   1.52213954,  -6.98788682, -11.72474916,  -4.04317197]])
#
#     # class mean
#     new_class_centroid = torch.tensor([[  3.3852, -15.8075,  -2.4407,  -1.5789,  -7.7599],
#                                        [-1.4863, -5.9415, -14.7030, -9.8278, -5.1399],
#                                        [4.4185, -7.6027, -5.1321, -10.6641, -11.7929],
#                                        [0.0325, -6.6269, -3.0853, -5.8959, -5.6192]])
#     centroids = torch.cat((base_class_centroids, new_class_centroid), 0)
#     data_dir = "./"
#     filename = 'centroid_unknowdata_NSLKDD_4_0703_our.csv'
#     raw_data_filename = data_dir + filename
#     raw_data = pd.read_csv(raw_data_filename, header=None)
#     features = raw_data.iloc[:, 0:raw_data.shape[1] - 1]
#     np_features = np.array(features)
#     labels = raw_data.iloc[:, raw_data.shape[1] - 1:]
#     labels = labels.values.ravel()
#     np_labels = np.array(labels)+5
#     # df = pd.DataFrame(np_features, index=np_labels)
#     # df_drop = df.drop(index=[8])
#     # np_features = df_drop.values
#     # np_labels = df_drop.index.values
#     print(np_labels.shape)
#     print(np_labels)
#     known_raw_data = pd.read_csv('./centroid_knowdata_NSLKDD_0705.csv', header=None)
#     known_features = known_raw_data.iloc[:, 0:known_raw_data.shape[1] - 1]
#     np_known_features = np.array(known_features)
#     known_labels = known_raw_data.iloc[:, known_raw_data.shape[1] - 1:]
#     known_labels = known_labels.values.ravel()
#     np_known_labels = np.array(known_labels)
#     np_all_features = np.concatenate((np_features,np_known_features),axis=0)
#     print('np_all_features',np_all_features.shape)
#     np_all_labels = np.concatenate((np_labels,np_known_labels),axis=0)
#     print('np_all_labels',np_all_labels.shape)
#     _, all_pred = cal_min_dis_to_centroid(torch.from_numpy(np_all_features).double(), centroids.double())
#     print(classification_report(np_all_labels.tolist(), all_pred.numpy().tolist()))
#     print(confusion_matrix(np_all_labels.tolist(), all_pred.numpy().tolist()))
#     all_report = precision_recall_fscore_support(np_all_labels.tolist(), all_pred.numpy().tolist(), average='weighted')
#     all_precision = all_report[0]
#     all_recall = all_report[1]
#     all_fscore = all_report[2]
#     print('all_precision', all_precision, 'all_recall', all_recall, 'all_fscore', all_fscore)
#     ## only known data
#     _, known_pred = cal_min_dis_to_centroid(torch.from_numpy(np_known_features).double(), centroids.double())
#     print(classification_report(np_known_labels.tolist(), known_pred.numpy().tolist()))
#     print(confusion_matrix(np_known_labels.tolist(), known_pred.numpy().tolist()))
#     known_report = precision_recall_fscore_support(np_known_labels.tolist(), known_pred.numpy().tolist(), average='weighted')
#     known_precision = known_report[0]
#     known_recall = known_report[1]
#     known_fscore = known_report[2]
#     print('known_precision', known_precision, 'known_recall', known_recall, 'known_fscore', known_fscore)
#     ## only new unkown data
#     _, pred = cal_min_dis_to_centroid(torch.from_numpy(np_features).double(), centroids.double())
#     print(classification_report(np_labels.tolist(), pred.numpy().tolist()))
#     print(confusion_matrix(np_labels.tolist(), pred.numpy().tolist()))
#     unknown_report = precision_recall_fscore_support(np_labels.tolist(), pred.numpy().tolist(), average='weighted')
#     unknown_precision = unknown_report[0]
#     unknown_recall = unknown_report[1]
#     unknown_fscore = unknown_report[2]
#     print('unknown_precision',unknown_precision,'unknown_recall',unknown_recall,'unknown_fscore',unknown_fscore)








