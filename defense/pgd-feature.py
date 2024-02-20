from ast import arg
import logging
import time
import torch.nn.functional as F
from calendar import c
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Resize
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10, GTSRB, ImageFolder, CIFAR100
import torch
import logging
import argparse
import sys
from tqdm import tqdm
import torch.nn as nn
import os
from torch.utils.data import random_split
sys.path.append('../')
sys.path.append(os.getcwd())

import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform, get_dataset_mean_std
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.nCHW_nHWC import *
from utils.save_load_attack import load_attack_result
sys.path.append(os.getcwd())
import yaml
from pprint import pprint, pformat


import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from utils.choose_index import choose_index_balanced, choose_index

import matplotlib.pyplot as plt
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, x):
        return (x - self.mean) / self.std

def dynamiccluster(arrays):
    score_list = list()
    arrays = np.array(arrays)
    silhouette_int = float("-inf")
    arrays = arrays.reshape(-1,1)
    print(arrays)
    for n_clusters in range(2, 3):
        model_kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels_tmp =model_kmeans.fit_predict(arrays)
        silhouette_tmp =metrics.silhouette_score(arrays, cluster_labels_tmp)
        if silhouette_tmp >silhouette_int:
            best_k =n_clusters
            silhouette_int =silhouette_tmp
            best_kmeans =model_kmeans
            cluster_labels_k =cluster_labels_tmp
    score_list.append([n_clusters, silhouette_tmp])
    minor =  np.argmin(np.bincount(cluster_labels_k))
    # print(len([idx for idx in range(len(cluster_labels_k)) if cluster_labels_k[idx] == minor]))
    return [idx for idx in range(len(cluster_labels_k)) if cluster_labels_k[idx] == minor]

def anomaly_det(arrays):
    score_list = list()
    arrays = np.array(arrays)
    # plt.scatter(arrays[:,0], arrays[:,2], s=10)
    # acc_diff_1 = arrays[:,1]
    # acc_diff_2 = arrays[:,0]
    # diff_arrays = np.concatenate((acc_diff_1[:,None], acc_diff_2[:,None]), axis=1)
    # diff_arrays = acc_diff_1-acc_diff_2
    arrays = arrays.reshape(-1,1)            # using acc change to cluster
    print(arrays)
    
    # print('clustering...')
    # print(arrays.shape)
    # print(arrays)
    # plt.scatter(diff_arrays[:,0], diff_arrays[:,1], s=10)
    # helper = dynamiccluster(arrays)
    # anomaly_ratio = len(helper)/len(arrays)
    # print(anomaly_ratio)
    # print(anomaly_ratio)
    # algo = LocalOutlierFactor()
    # Robust Convariance
    # algo = EllipticEnvelope(contamination = 0.2, random_state=0)
    # Empirical Covariance
    algo = EllipticEnvelope(contamination = 0.1, support_fraction=1., random_state=0)
    # algo = svm.OneClassSVM(nu = 0.1, kernel='rbf', gamma=0.1)
    
    y_pred = algo.fit_predict(arrays)
    # y_pred = algo.fit(arrays).predict(arrays)
    # print(y_pred)
    idx = np.where(y_pred == -1)[0]
    
    # plt.savefig('./diff_array.png')
    # print('saved')
    return idx


class Norm_layer(nn.Module):
    def __init__(self,mean,std) -> None:
        super(Norm_layer,self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1),requires_grad = False)

        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1),requires_grad = False)

    def forward(self,x):
        return x.sub(self.mean).div(self.std)


def get_args():
    #set the basic parameter
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, help='cuda, cpu')
    parser.add_argument('--checkpoint_load', type=str)
    parser.add_argument('--checkpoint_save', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument("--data_root", type=str)

    parser.add_argument('--dataset', type=str, help='mnist, cifar10, gtsrb, celeba, tiny') 
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--input_height", type=int)
    parser.add_argument("--input_width", type=int)
    parser.add_argument("--input_channel", type=int)
    parser.add_argument("--inference", type=bool, default = False)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--num_workers", type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')

    parser.add_argument('--poison_rate', type=float)
    parser.add_argument('--target_type', type=str, help='all2one, all2all, cleanLabel') 
    parser.add_argument('--target_label', type=int)
  
    parser.add_argument('--model', type=str, help='resnet18')
    parser.add_argument('--random_seed', type=int, help='random seed')
    parser.add_argument('--index', type=str, help='index of clean data')
    parser.add_argument('--result_file', type=str, help='the location of result')

    parser.add_argument('--yaml_path', type=str, default="./config/defense/ac/config.yaml", help='the path of yaml')

    #set the parameter for the ac defense
    parser.add_argument('--nb_dims', type=int, help='umber of dimensions to reduce activation to')
    parser.add_argument('--nb_clusters', type=int, help='number of clusters (defaults to 2 for poison/clean).')
    parser.add_argument('--cluster_analysis', type=str, help='the method of cluster analysis')
    
    arg = parser.parse_args()

    print(arg)
    return arg

def ranking(model,adv_list,label):
    rank_list = []
    for data in adv_list:
        # print(len(data))  s
        result_list = []
        for image in data:
            pred = model(image.cuda())
            pred = np.argmax(pred.cpu().detach(), axis=-1)
            correct = pred == label
            correct = np.sum(correct.numpy(), axis=-1)
            result_list.append(correct/image.shape[0])

        rank_list.append(result_list)
    return rank_list

def obtain_adv_dataset(model, dataset):
    mean, std = get_dataset_mean_std(args.dataset)

    if args.dataset == 'tiny':
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        model = nn.Sequential(Norm_layer(mean,std),model)
        model = model.cuda()
    elif args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
        model = nn.Sequential(Norm_layer(mean,std),model)
        model = model.cuda()
    elif args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        model = nn.Sequential(Norm_layer(mean,std),model)
        model = model.cuda()
    elif args.dataset == 'gtsrb':
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        model = nn.Sequential(Norm_layer(mean,std),model)
        model = model.cuda()

    for idx, (data, label) in enumerate(tqdm(train_loader)):
        x = data
        y = label
        adv_images = x + args.alpha*torch.empty_like(x).uniform_(-args.alpha, args.alpha).sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach().cuda()
        break
    conv_outputs = []
    def get_conv_output_hook(module, input, output):
        if isinstance(module, nn.Conv2d):
            conv_outputs.append(output)

    # Register the hook on all convolutional layers
    def model_hook(model):
        handles = []
        for module in model.modules():
            handle = module.register_forward_hook(get_conv_output_hook)
            handles.append(handle)
        return handles

    def remove_hook(handles):
        for handle in handles:
            handle.remove
    handles = model_hook(model)
    x = x.cuda()
    output = model(x)
    num_conv_layer = len(conv_outputs)
    total_feature = 0
    inter_node = []
    for layer in conv_outputs:
        total_feature+=layer.shape[1]
        inter_node.append(total_feature)
    for handle in handles:
        handle.remove()
    
    def adv_sample_generation(idx,x, adv_images):
        image_list = []
        conv_outputs = []
        def get_conv_output_hook(module, input, output):
            if isinstance(module, nn.Conv2d):
                conv_outputs.append(output)

        # Register the hook on all convolutional layers
        def model_hook(model):
            handles = []
            for module in model.modules():
                handle = module.register_forward_hook(get_conv_output_hook)
                handles.append(handle)
            return handles

        def remove_hook(handles):
            for handle in handles:
                handle.remove
        handles = model_hook(model)
        x = x.cuda()
        adv_images = adv_images.requires_grad_(True).cuda()
        output = model(x)
        adv_output = model(adv_images)
        clean_feature_maps = []
        adv_feature_maps = []
        start = 0
        for i in range(conv_outputs[idx].shape[1]):
            multi_step_images = []
            clean_feature_map = conv_outputs[idx][:,i, :,:]
            adv_feature_map = conv_outputs[idx+len(conv_outputs)//2][:,i, :,:]
            loss = F.mse_loss(clean_feature_map,adv_feature_map)
            grad = torch.autograd.grad(loss, adv_images,
                                        retain_graph=True, create_graph=False)[0]
            adv_images_step1 = adv_images.detach() + args.alpha*grad.sign()
            dv_images_step2 = adv_images.detach() + 2*(args.alpha*grad.sign())
            delta1 = torch.clamp(adv_images_step1 - x, min=-args.eps, max=args.eps)
            delta2 = torch.clamp(dv_images_step2 - x, min=-args.eps, max=args.eps)
            adv_images1 = torch.clamp(x + delta1, min=0, max=1).detach()
            adv_images2 = torch.clamp(x + delta2, min=0, max=1).detach()
            multi_step_images.append(adv_images1.cpu())
            multi_step_images.append(adv_images2.cpu())
            for handle in handles:
                handle.remove()
            # image_list.append([adv_images1.cpu(),adv_images2.cpu()])
            image_list.append(multi_step_images)
        return image_list

    adv_list = []

    total_rank_list = []
    rank_acc_list = []
    channel_idx = 0
    feature_idx = 0
    for i in tqdm(range(len(conv_outputs))):
        adv_list = adv_sample_generation(i, x, adv_images)
        # adv_list.append(adv_sample_generation(i, x, adv_images))
        rank_list = ranking(model,adv_list,y)
        rank_acc_list.append(rank_list)
        total_rank_list.append(dynamiccluster(rank_list))
    
    # for i in range(len(conv_outputs)//2, len(conv_outputs)):    
    #     total_rank_list.append([])
    #     rank_acc_list.append([])

    return total_rank_list, rank_acc_list

def initialize(model, rank_list):
    start_idx = 0
    for name, parameter in model.named_modules():
        if isinstance(parameter, nn.BatchNorm2d):
            helper = rank_list[start_idx-1]
            for fmidx in helper:
                parameter.weight.data[fmidx] = torch.zeros_like(parameter.weight.data[fmidx])
                # parameter.bias.data[fmidx] = torch.zeros_like(parameter.bias.data[fmidx])
        elif isinstance(parameter, nn.Conv2d):
            start_idx+=1
            helper = rank_list[start_idx-1]
            for fmidx in helper:
                parameter.weight.data[fmidx] = torch.zeros_like(parameter.weight.data[fmidx])
                # parameter.bias.data[fmidx] = torch.zeros_like(parameter.bias.data[fmidx])
    return model

def train(model,train_loader,test_loader,rank_list):
    model = initialize(model, rank_list)
            

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        for i, (inputs,labels) in enumerate(tqdm(train_loader)):
            model.train()
            model.to(args.device)
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        # total = 0
        # clean_acc = 0
        # for i, (inputs,labels) in enumerate(tqdm(test_loader)):
        #     inputs = inputs.to(args.device)
        #     outputs = model(inputs)
        #     pred = np.argmax(outputs.cpu().detach(), axis=-1)
        #     curr_correct = pred == labels
        #     clean_acc += np.sum(curr_correct.numpy(), axis=-1)
        #     total+=len(labels)
        # print('epoch: {} test acc: {}'.format(epoch, clean_acc/total))

    return model

if __name__ == "__main__":
    
    ### 1. basic setting: args
    start_time = time.time()
    args = get_args()
    with open(args.yaml_path, 'r') as stream: 
        config = yaml.safe_load(stream) 
    config.update({k:v for k,v in args.__dict__.items() if v is not None})
    args.__dict__ = config
    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    
    fix_random(args.random_seed)
    
    save_path = '/record/' + args.result_file
    if args.checkpoint_save is None:
        args.checkpoint_save = save_path + '/record/defence/feature/'
        if not (os.path.exists(os.getcwd() + args.checkpoint_save)):
            os.makedirs(os.getcwd() + args.checkpoint_save) 
    if args.log is None:
        args.log = save_path + '/saved/feature/'
        if not (os.path.exists(os.getcwd() + args.log)):
            os.makedirs(os.getcwd() + args.log) 
    args.save_path = save_path

    ### 2. attack result(model, train data, test data)
    result = load_attack_result(os.getcwd() + save_path + '/attack_result.pt')
    model = generate_cls_model(args.model,args.num_classes)
    model.load_state_dict(result['model'])
    model = model.to(args.device)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_test']['x']
    y = result['bd_test']['y']
    data_bd_test = list(zip(x,y))
    data_bd_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_bd_test,
        poison_idx=np.zeros(len(data_bd_test)),
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['clean_test']['x']
    y = result['clean_test']['y']
    data_clean_test = list(zip(x,y))
    data_clean_testset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_test,
        poison_idx=np.zeros(len(data_clean_test)),
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)

    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = True)
    x = result['clean_train']['x']
    y = result['clean_train']['y']
    data_all_length = len(y)
    # ran_idx = choose_index(args, data_all_length) 
    ran_idx = choose_index_balanced(args, data_all_length, y, num_classes=args.num_classes)
    log_index = os.getcwd() + args.log + 'index.txt'
    np.savetxt(log_index,ran_idx, fmt='%d')
    data_clean_train = list(zip([x[ii] for ii in ran_idx],[y[ii] for ii in ran_idx]))
    data_clean_trainset = prepro_cls_DatasetBD(
        full_dataset_without_transform=data_clean_train,
        poison_idx=np.zeros(len(data_clean_train)),
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=tran,
        ori_label_transform_in_loading=None,
        add_details_in_preprocess=False,
    )
    data_train_loader = torch.utils.data.DataLoader(data_clean_trainset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)


    if args.inference:
        helper = torch.load(os.getcwd() + f'{save_path}/feature/defense_result.pt')
        model.load_state_dict(helper['model'])
        print('clean acc: {}'.format(helper['clean_acc']))
        print('ASR acc: {}'.format(helper['asr_acc']))
    else:
        total_bd = 0
        total_clean = 0
        total_train = 0
        rank_list,rank_acc_list = obtain_adv_dataset(model,data_clean_trainset)
        np.save('acc_list.npy', np.array(rank_acc_list))
        # rank_acc_list = torch.load(os.getcwd() + f'{save_path}/feature/defense_result.pt')['rank_acc_list']
        # rank_list = []
        # for i in range(len(rank_acc_list)):
        #     print(rank_acc_list[i])
        #     rank_list.append(anomaly_det(rank_acc_list[i]))
        # rank_list = torch.load(os.getcwd() + f'{save_path}/feature/defense_result.pt')['rank_list']

        model = train(model,data_train_loader,data_clean_loader,rank_list)
        with torch.no_grad():
            model.eval()
            asr_acc = 0
            
            for i, (inputs,labels) in enumerate(data_bd_loader):
                inputs = inputs.to(args.device)
                outputs = model(inputs)
                pred = np.argmax(outputs.cpu().detach(), axis=-1)
                curr_correct = pred == labels
                asr_acc += np.sum(curr_correct.numpy(), axis=-1)
                total_bd += len(labels)
            clean_correct = 0
            for i, (inputs,labels) in enumerate(data_clean_loader):

                inputs = inputs.to(args.device)
                outputs = model(inputs)
                pred = np.argmax(outputs.cpu().detach(), axis=-1)
                curr_correct = pred == labels
                clean_correct += np.sum(curr_correct.numpy(), axis=-1)
                total_clean += len(labels)
    tran = get_transform(args.dataset, *([args.input_height,args.input_width]) , train = False)
    x = result['bd_test']['x']
    robust_acc = -1
    if 'original_targets' in result['bd_test']:
        y_ori = result['bd_test']['original_targets']
        if y_ori is not None:
            if len(y_ori) != len(x):
                y_idx = result['bd_test']['original_index']
                y = y_ori[y_idx]
            else :
                y = y_ori
            data_bd_test = list(zip(x,y))
            data_bd_testset = prepro_cls_DatasetBD(
                full_dataset_without_transform=data_bd_test,
                poison_idx=np.zeros(len(data_bd_test)),
                bd_image_pre_transform=None,
                bd_label_pre_transform=None,
                ori_image_transform_in_loading=tran,
                ori_label_transform_in_loading=None,
                add_details_in_preprocess=False,
            )
            data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,drop_last=False, shuffle=True,pin_memory=True)
        
            robust_acc = 0
            for i, (inputs,labels) in enumerate(data_bd_loader):  # type: ignore
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs)
                pre_label = torch.max(outputs,dim=1)[1]
                robust_acc += torch.sum(pre_label == labels)
            robust_acc = robust_acc/len(data_bd_test)
    print(asr_acc/total_bd, clean_correct/total_clean, robust_acc)
    if not (os.path.exists(os.getcwd() + f'{save_path}/feature/')):
        os.makedirs(os.getcwd() + f'{save_path}/feature/')
    torch.save(
    {
        'model_name':args.model,
        'model':model.state_dict(),
        'clean_acc':clean_correct/total_clean,
        'asr_acc':asr_acc/total_bd,
        'rank_list':rank_list,
        'rank_acc_list':rank_acc_list,
        'ra':robust_acc,
    }, os.getcwd() + f'{save_path}/feature/defense_result.pt')

    with open(os.getcwd() + f'{save_path}/feature/shuffledExtimationBS256_empirical_conv_defense_result_outlier0.3_lr0.01_data0.05-bs256.txt', 'w') as f:
        f.write('asr: '+str(asr_acc/total_bd)+' acc: '+str(clean_correct/total_clean)+' overhead: '+str(time.time()-start_time))
        f.close()