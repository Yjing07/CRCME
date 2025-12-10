# from lifelines.utils import concordance_index
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from random import shuffle
import torch
import random
import logging
import utils as utils
import numpy as np
import time,datetime
import json
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
import argparse
from utils import read_json
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math
from loss.loss import cox_loss_torch,nll_loss,CoxLoss,NegativeLogLikelihood,ClipLoss,FocalLoss
from metrics import concordance_index_torch
from model.vit_fusion_image import FusionModel, ViT_fu, ViT_ct,FusionPipeline
import matplotlib.pyplot as plt
# import sns
from dataset.ct_dataset import MAE_class
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, roc_auc_score, precision_recall_fscore_support,confusion_matrix
# from model.clip_tqn import TQN_Model
# from model.CTEncoder_kad import CT_fusion
# from model.WsiEncoder_kad import Wsi_fusion
# from torch_geometric.loader import DataLoader
# from model.ct_wsi_fusion import Fusion_Model

def bucketize(a: torch.Tensor, ids: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mapping = {k.item(): v.item() for k, v in zip(a, ids)}

    # From `https://stackoverflow.com/questions/13572448`.
    palette, key = zip(*mapping.items())
    key = torch.tensor(key)
    palette = torch.tensor(palette)

    index = torch.bucketize(b.ravel(), palette)
    remapped = key[index].reshape(b.shape)

    return remapped

def get_training_class_count(num_classes,label_dict,image_list):
    label_cal = {}
    for i in range(num_classes):
        label_cal[i] = []
    for key in image_list:
        if label_dict[key] in label_cal:
            label_cal[label_dict[key]].append(key)
    class_count = torch.zeros(len(label_cal.keys()))
    for key,values in label_cal.items():
        class_count[key] = len(values)
        # print(key,len(values))
    return class_count

def get_weight(args, lab_dict):
    image_list = lab_dict.keys()
    class_count = get_training_class_count(args.num_classes,lab_dict,image_list)
    weight = torch.tensor([sum(class_count)/i for i in class_count])
    weight = torch.nn.functional.softmax(torch.log(weight),dim=0)
    # weight = torch.nn.functional.softmax(weight,dim=0)
    return weight


def precision_recall_f1(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(num_classes):
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP

        precision = (TP+1e-15) / (TP + FP+1e-15)
        recall = (TP+1e-15) / (TP + FN+1e-15)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(2 * (precision * recall) / (precision + recall))
    return precision_list,recall_list,f1_list

def save_confusion_matrix(num_classes,confusion_matrix,save_dir):
    # sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(num_classes)), yticklabels=list(range(num_classes)))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(save_dir) 
    plt.clf()
    return 

def Acc(confusion_matrix):
    return np.trace(confusion_matrix)/np.sum(confusion_matrix)

@torch.no_grad()
def return_auc(target_array,possibility_array,num_classes):
    enc = OneHotEncoder()
    target_onehot = enc.fit_transform(target_array.unsqueeze(1))
    target_onehot = target_onehot.toarray()
    class_auc_list = []
    for i in range(num_classes):
        class_i_auc = roc_auc_score(target_onehot[:,i], possibility_array[:,i])
        class_auc_list.append(class_i_auc)
    macro_auc = roc_auc_score(np.round(target_onehot,0), possibility_array, average="macro", multi_class="ovo")
    return macro_auc, class_auc_list

@torch.no_grad()
def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]
def weighted_voting(model1_preds, model2_preds, weights=(0.5, 0.5)):
    # 计算加权投票，返回类别标签
    weighted_preds = []
    for pred1, pred2 in zip(model1_preds, model2_preds):
        weighted_score = [0, 0, 0, 0]  # 假设有3个类别
        weighted_score[pred1] += weights[0]
        weighted_score[pred2] += weights[1]
        weighted_preds.append(np.argmax(weighted_score))
    return np.array(weighted_preds)
def calculate_auc_acc(labels, predictions, num_classes):
    # Convert predictions to class labels
    # this_class_label1 = np.argmax(predictions1, axis=-1)
    # this_class_label2 = np.argmax(predictions2, axis=-1)
    # Calculate accuracy
    # this_class_label = weighted_voting(this_class_label1, this_class_label2, weights=(0.9, 0.1))
    this_class_label = np.argmax(predictions, axis=-1)
    acc = accuracy_score(labels, this_class_label)

    # Calculate AUC based on number of classes
    if num_classes > 2:
        auc_value, class_auc_list = return_auc(torch.LongTensor(np.array(labels)), 
                                               torch.Tensor(np.array(predictions)), 
                                               num_classes)
    else:
        # Binary case: Use the probability for class 1 for AUC calculation
        auc_value = roc_auc_score(labels, np.array([i[1] for i in predictions]))
        class_auc_list = []

    return auc_value, acc
@torch.no_grad()
def five_scores(labels, predictions, num_classes, all_patient_ids,flag):
    
    # predictions = this_class_label
    this_class_label = list(np.argmax(predictions,axis=-1))
    # this_class_label =list(predictions)
    if flag:
        val_dict = {}
        for i in range (len(all_patient_ids)):
            _dic = {}
            name = all_patient_ids[i]
            _dic['label'] = int(labels[i])
            _dic['pred'] = int(this_class_label[i])
            val_dict[name] = _dic
        # logging.info(f'The patient_ids of val: {all_patient_ids}')
        # logging.info(f'The labels of val: {labels}')
        # logging.info(f'The predtions label of val: {this_class_label}')
        # logging.info(f'The predtions of val: {predictions}')
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, this_class_label, average='macro')
    acc=accuracy_score(labels, this_class_label)
    if num_classes>2:
        auc_value, class_auc_list = return_auc(torch.LongTensor(np.array(labels)), torch.Tensor(np.array(predictions)), num_classes)
        # c_m = confusion_matrix(labels, this_class_label)
    else:
        # fpr, tpr, threshold = roc_curve(labels, [i[1] for i in predictions])
        # fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        # this_class_label = np.array(predictions)
        # this_class_label[this_class_label>=threshold_optimal] = 1
        # this_class_label[this_class_label<threshold_optimal] = 0
        auc_value = roc_auc_score(labels, [i[1] for i in predictions])
        class_auc_list = []
    c_m = confusion_matrix(labels,  this_class_label)
    return c_m, auc_value, class_auc_list, acc, precision, recall, fscore

@torch.no_grad()
def evaluate(model_a, model_b, fusion_model, data_loader, args,flag=0):
    model_a.eval()
    model_b.eval()
    fusion_model.eval()
    pred_all = []
    # pred_all2 = []
    label_all = []
    all_patient_id = []
    weights = (0.6, 0.4)
    for step, data in enumerate(data_loader):
        images, label, patient_id = data
        for i in patient_id:
            all_patient_id.append(i)
        # Move tensors to GPU and add extra dimension
        images = torch.unsqueeze(images, 1).to(torch.float32).cuda()
        # rois = torch.unsqueeze(rois, 1).to(torch.float32).cuda()

        # Model prediction based on args.model_name
        if args.model_name == 'resnet18_fe':
            images = torch.cat([images, rois], 1)
            pred = model(images)
        else:
            pred = fusion_model(images)
            # output_a = model_a(images)
            # output_b = model_b(images) 
            # pred1,pred2 = fusion_model(output_a, output_b)
            # pred = fusion_model(output_a, output_b)

        # Apply softmax and move to CPU only once per batch
        # pred_all.extend(F.softmax(pred1*weights[0]+pred2*weights[1], dim=-1).cpu().numpy())
        pred_all.extend(F.softmax(pred, dim=-1).cpu().numpy())
        label_all.extend(label.cpu().numpy())

    # Calculate AUC and Accuracy
    # auc_value, accuracy = calculate_auc_acc(label_all, pred_all, args.num_classes)
    c_m, auc_value, class_auc_list, accuracy, precision, recall, fscore = five_scores(label_all, pred_all, args.num_classes,all_patient_id,flag)
    # return auc_value, accuracy
    return c_m, auc_value, class_auc_list, accuracy, precision, recall, fscore, label_all, pred_all, all_patient_id

def train_one_epoch(model_a, model_b, fusion_model, optimizer, data_loader, loss_fn, epoch, writer,args):
    
    # model_a.eval()
    # model_b.eval()
    fusion_model.train()
    for step, data in enumerate(data_loader):
        images, label,_ = data
        images = torch.unsqueeze(images,1).to(torch.float32).cuda()
        # rois = torch.unsqueeze(rois,1).to(torch.float32).cuda()
        label = label.cuda()
        # output_a = model_a(images)
        # output_b = model_b(images)
        pred = fusion_model(images)
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), epoch * len(data_loader) + step)
        writer.add_scalar('Loss/loss_class', loss.item(), epoch * len(data_loader) + step)

        if step%10==0:
            print("[epoch {},step {}/ {}] loss {} ".format(epoch,step, len(data_loader), round(loss.detach().cpu().item(), 4)))
            logging.info("[epoch {},step {}/ {}] loss {} ".format(epoch,step, len(data_loader), round(loss.detach().cpu().item(), 4)))

    return 

def main(args):
    # device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    batch_size = args.batch_size
    summary_writer = SummaryWriter(args.log_dir)

    model_a = ViT_ct(
    image_size = 256,          # image size
    frames = 32,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 16,      # frame patch size
    dim = 1024,
    depth = 24,
    heads = 16,
    emb_dropout = 0.1,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)) ### 196
 
    ct_weights_path = args.pretrained
    model_a_dict = model_a.state_dict()
    if os.path.exists(ct_weights_path):
        updata_dict = OrderedDict()
        weights_dict = torch.load(ct_weights_path, map_location='cpu')
        pretrained_dict = weights_dict['model']
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items(): # k为module.xxx.weight, v为权重
            name = k[7:] # 截取`module.`后面的xxx.weight
            new_state_dict[name] = v
        for k, v in new_state_dict.items():
            if k in model_a_dict:
                updata_dict[k]=v
        model_a_dict.update(updata_dict)
        model_a.load_state_dict(model_a_dict)
        print("model a load successed!!!!")
        for param in model_a.parameters():
            param.requires_grad = False

        for name, param in model_a.named_parameters():
            if 'blocks' not in name and 'mlp_head' in name:
                param.requires_grad = True
            elif 'adapter' in name:
                param.requires_grad = True
            elif 'prompt' in name:
                param.requires_grad = True
            elif 'c_attn' in name:
                param.requires_grad = True
            elif 'pool' in name or 'head' in name:
                param.requires_grad = True
        # 打印没有被冻结的参数名称
        for name, param in model_a.named_parameters():
            if param.requires_grad:
                print(name)

    model_a = model_a.cuda()
    # for param in model_a.parameters():
    #     param.requires_grad = False

    model_b = ViT_fu(
    image_size = 256,          # image size
    frames = 32,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 16,      # frame patch size
    dim = 1024,
    depth = 24,
    heads = 16,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)) ### 196
 
    fusion_weights_path = args.pretrained_fu
    model_b_dict = model_b.state_dict()
    if os.path.exists(fusion_weights_path):
        updata_dict = OrderedDict()
        weights_dict = torch.load(fusion_weights_path, map_location='cpu')
        pretrained_dict = weights_dict['model']
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items(): # k为module.xxx.weight, v为权重
            name = k[7:] # 截取`module.`后面的xxx.weight
            new_state_dict[name] = v
        for k, v in new_state_dict.items():
            if k in model_b_dict:
                updata_dict[k]=v
        model_b_dict.update(updata_dict)
        model_b.load_state_dict(model_b_dict)
        print("model b load successed!!!!")
        for param in model_b.parameters():
            param.requires_grad = False

        for name, param in model_b.named_parameters():
            if 'blocks' not in name and 'mlp_head' in name:
                param.requires_grad = True
            # elif 'adapter' in name:
            #     param.requires_grad = True
            elif 'prompt' in name:
                param.requires_grad = True
            elif 'c_attn' in name:
                param.requires_grad = True
            elif 'pool' in name or 'head' in name:
                param.requires_grad = True
        # 打印没有被冻结的参数名称
        for name, param in model_b.named_parameters():
            if param.requires_grad:
                print(name)
    model_b = model_b.cuda()
    # for param in model_b.parameters():
    #     param.requires_grad = False

    fusion_model = FusionModel(input_dim_a=1024, input_dim_b=1024, classes=args.num_classes)
    pipeline = FusionPipeline(model_a, model_b, fusion_model,args.num_classes)
    # 打印没有被冻结的参数名称
    for name, param in pipeline.named_parameters():
        if param.requires_grad:
            print(name)
    pipeline = pipeline.cuda()
    # 数据划分，并建立训练集，测试集
    args.Fold = 'Fold_' + str(args.Fold)
    img_idx_list = read_json(args.img_idx)
    label = read_json(args.label_path)
    train_ind = img_idx_list[args.Fold]['train']
    val_ind = img_idx_list[args.Fold]['val']

    trainset = MAE_class(train_ind, label,args.data_path, args.shape, args.task)
    valset = MAE_class(val_ind, label,args.data_path, args.shape, args.task)

    train_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size,
                                            pin_memory=True,
                                            num_workers=opt.num_workers,
                                            shuffle=True,
                                            drop_last=True)

    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=batch_size,
                                             pin_memory=True,
                                             num_workers=opt.num_workers,
                                             shuffle=True,
                                             drop_last=True
                                             )
    
    print("Creating model")
    # params = list(model_a.parameters()) + list(model_b.parameters()) + list(fusion_model.parameters())
    # params = list(model_a.parameters()) + list(model_b.parameters()) + list(fusion_model.parameters()) + list(pipeline.parameters())
    optimizer = optim.AdamW(pipeline.parameters(), lr=args.lr, weight_decay=1e-4) #, momentum=0.9
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / (args.epochs))) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # if args.warmup_epochs > 0:
    #     scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1e-7, total_iters=args.warmup_epochs)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    # loss_fn = FocalLoss(alpha=get_weight(args, train_ind)).cuda()
    # loss_fn = torch.nn.CrossEntropyLoss().cuda()
    val_best = 0
    for epoch in range(args.epochs):
        train_one_epoch(model_a,
                        model_b,
                        pipeline,
                        optimizer=optimizer,
                        data_loader=train_loader,
                        loss_fn =loss_fn,
                        epoch=epoch,
                        writer=summary_writer,
                        args=args)
        
        scheduler.step()
        # val_auc, val_accuracy = evaluate(model_a,model_b,pipeline, data_loader=val_loader, args=args)
        train_c_m, train_auc, train_auc_list, train_accuracy, train_precision, train_recall, train_fscore, labels_train, pres_train, ids_train = evaluate(model_a,model_b,pipeline, data_loader=train_loader, args=args)
        val_c_m, val_auc, val_auc_list, val_accuracy, val_precision, val_recall, val_fscore, labels_val, pres_val, ids_val = evaluate(model_a,model_b,pipeline, data_loader=val_loader, args=args,flag=1)
        if val_auc > val_best:
            _all = {}
            all_labels = labels_train + labels_val
            all_preds = pres_train + pres_val
            all_ids = ids_train + ids_val
            assert len(all_labels) == len(all_preds) == len(all_ids)
            _all['ids'] = all_ids
            _all['preds'] = [list(map(float, list(part))) for part in all_preds]
            _all['labels'] = [int(part) for part in all_labels]
            json.dump(_all, open(args.log_dir + f'/{args.Fold}_pred.json','w'))
            val_best = val_auc
            torch.save(pipeline.state_dict(), os.path.join(args.model_dir, f'the best auc model.pth'))
            logging.info('***********************************')

        print("[epoch {}] train auc: {}, train auc list: {}, train_acc: {}, train_precision: {}, train_recall: {}, train_fscore: {}".format(epoch, train_auc, train_auc_list, train_accuracy, train_precision, train_recall, train_fscore))
        print("[epoch {}] train confusion matrix:{}".format(epoch, train_c_m))
        logging.info("[epoch {}] train auc: {}, train auc list: {}, train_acc: {}, train_precision: {}, train_recall: {}, train_fscore: {}".format(epoch, train_auc, train_auc_list, train_accuracy, train_precision, train_recall, train_fscore))
        logging.info("[epoch {}] train confusion matrix:{}".format(epoch, train_c_m))
        print("[epoch {}] val auc: {}, val auc list: {}, val_acc: {}, val_precision: {}, val_recall: {}, val_fscore: {}".format(epoch, val_auc, val_auc_list, val_accuracy, val_precision, val_recall, val_fscore))
        print("[epoch {}] val confusion matrix:{}".format(epoch, val_c_m))
        logging.info("[epoch {}] val auc: {}, val auc list: {}, val_acc: {}, val_precision: {}, val_recall: {}, val_fscore: {}".format(epoch, val_auc, val_auc_list, val_accuracy, val_precision, val_recall, val_fscore))
        logging.info("[epoch {}] val confusion matrix:{}".format(epoch, val_c_m))
        summary_writer.add_scalar('train auc', train_auc, epoch)
        summary_writer.add_scalar('train acc', train_accuracy, epoch)
        summary_writer.add_scalar('train prec', train_precision, epoch)
        summary_writer.add_scalar('train rec', train_recall, epoch)
        summary_writer.add_scalar('train f1', train_fscore, epoch)

        summary_writer.add_scalar('val auc', val_auc, epoch)
        summary_writer.add_scalar('val acc', val_accuracy, epoch)
        summary_writer.add_scalar('val prec', val_precision, epoch)
        summary_writer.add_scalar('val rec', val_recall, epoch)
        summary_writer.add_scalar('val f1', val_fscore, epoch)

    print('The best val auc: {}'.format(val_best))
    logging.info('The best val auc: {}'.format(val_best))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='mae',
                        help='model name:[resnet18_fe,resnet34,resnet50_fe,resnet101,resnext50,resnext50_fe,resnext152_fe,resnet18_fe,resnet34_fe]')
    parser.add_argument('--vit', default=True, type=bool)
    parser.add_argument('--pretrained', default='/cache/yangjing/main_files/CRCFound2/argo2/mymodel/checkpoint-999.pth', type=str)
    parser.add_argument('--pretrained_fu', default='/cache/yangjing/main_files/CRCFound2/CRCFound2_98/test/patch40_1_3/mix_1880/patch16_frame16_large_256-None/logs/checkpoint-70.pth', type=str)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--scale', type=list, default=[1])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--shape', type=tuple, default=(32,256,256))
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--mask_ratio', default=0., type=float, help='mask ratio of pretrain')
    parser.add_argument('--Fold', type=int, default=4)
    # parser.add_argument('--data_path', type=str, default="/cache/yangjing/CRCFound2/datas/all_CT/data/crop_256_32")
    parser.add_argument('--data_path', type=str, default="/cache/yangjing/main_files/CRCFound2/argo2/liuyuan_CT/crop_256_32/image")
    parser.add_argument('--label_path', type=str, default="/cache/yangjing/main_files/CRCFound2/datas/all_CT/class_data_split/new_split0321/label_1011.json")
    parser.add_argument('--img_idx', type=str, default="/cache/yangjing/main_files/CRCFound2/datas/all_CT/class_data_split/new_split0321/cms_320.json")
    parser.add_argument('--task', type=str, default="cms")
    # parser.add_argument('--img_idx', type=str, default="/cache/yangjing/CRCFound2/argo2/NB_five_fold_n_361.json")
    parser.add_argument('--log_path', type=str,
                        default='/cache/yangjing/main_files/CRCFound2/CRCFound2_98/pretrain_13308/logs/downtask/debug',
                        help='path to log')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--ver', type=str,
                        default='/Fold4',
                        help='version of training')
    parser.add_argument('--device', default='cuda:4', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    exp_name = opt.model_name+opt.ver+'-'+str(opt.lr)
    opt.log_dir = opt.log_path+'/' + opt.task.split('.')[-1] +'/'+ exp_name + '/logs'
    opt.model_dir = opt.log_path+'/'+ opt.task.split('.')[-1] +'/'+ exp_name + '/models'
    os.makedirs(opt.log_dir, exist_ok=True)
    os.makedirs(opt.model_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(opt.log_dir, 'train_log.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    logging.info('Hyperparameter setting{}'.format(opt))
    main(opt)