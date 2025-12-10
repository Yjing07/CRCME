import os
import SimpleITK as sitk
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
# from utils.util import resize_image_itk,randomcrop,crop,read_json #utils.
import random
random.seed(42)
import joblib
from torch.utils.data import BatchSampler
from torchvision import transforms
from PIL import Image
import cv2
import torch.nn.functional as F
from scipy import ndimage
from timm.models.layers import to_3tuple
# from img_transform import *
def randomflip_z(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[:, ::-1, ...]

def randomflip_x(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[..., ::-1]

def randomflip_y(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[:, :, ::-1, ...]

def random_flip(image, mode='x', p=0.5):
    if mode == 'x':
        image = randomflip_x(image, p=p)
    elif mode == 'y':
        image = randomflip_y(image, p=p)
    elif mode == 'z':
        image = randomflip_z(image, p=p)
    else:
        raise NotImplementedError(f'Unknown flip mode ({mode})')
    return image

def rotate(image, angle=10):
    angle = random.randint(-10, 10)
    r_image = ndimage.rotate(image, angle=angle, axes=(-2, -1), reshape=True)
    if r_image.shape != image.shape:
        r_image = center_crop(r_image, target_shape=image.shape[1:])
    return r_image

def window_level_normalization(img, level=35, window=300):
    min_HU = level - window / 2
    max_HU = level + window / 2
    img[img > max_HU] = max_HU
    img[img < min_HU] = min_HU
    img = 1. * (img - min_HU) / (max_HU - min_HU)
    return img

def process(img, level=35, window=300):
    img = window_level_normalization(img, level=35, window=300)
    return img[:,50:450]*255

class colon_cancer(Dataset):
    """自定义数据集"""

    def __init__(self, train_ind, data_path, lab_dict, shape):
        super(colon_cancer,self).__init__()

        self.images_ind = train_ind
        self.lab_dict = lab_dict
        self.ct_path = data_path
        self.size = shape
        self.tuominID = [self.lab_dict[i]['tuominID'] for i in self.images_ind]
        self.label = [self.lab_dict[i]['dfs.event'] for i in self.images_ind]
        self.delay = [self.lab_dict[i]['dfs.delay'] for i in self.images_ind]
        self.ct_texts = [self.lab_dict[i]['ct_report'] for i in self.images_ind]
        self.wsi_texts = [self.lab_dict[i]['patho_report'] for i in self.images_ind]

    def __len__(self):
        return len(self.tuominID)

    def __getitem__(self, i):
        img_root = os.path.join(self.ct_path,'image',self.tuominID[i])
        roi_root = os.path.join(self.ct_path,'roi',self.tuominID[i])
        img = sitk.ReadImage(img_root)
        roi = sitk.ReadImage(roi_root)

        img = sitk.GetArrayFromImage(img).astype(np.float32)
        roi = sitk.GetArrayFromImage(roi).astype(np.float32)
        # img,roi = randomcrop(img,roi,self.size)
        img = torch.from_numpy(img)
        roi = torch.from_numpy(roi)
        
        event =  torch.tensor(self.label[i])
        delay =  torch.tensor(self.delay[i])
        text_caption = self.ct_texts[i][:500]
        wsi_caption = self.wsi_texts[i][:500]
        
        return img,roi, event, delay, text_caption, wsi_caption, self.images_ind[i]   #,os.path.basename(self.images_ind[i])
def get_start_idx(orig_dim, target_dim):
        if orig_dim > target_dim:
            return np.random.randint(0, orig_dim - target_dim + 1)
        else:
            return 0  # 如果原始尺寸小于目标尺寸，从 0 开始裁剪

def randomcrop(ct_array, target_size):
    orig_d, orig_h, orig_w = ct_array.shape
    target_d, target_h, target_w = target_size

    d_start = get_start_idx(orig_d, target_d)
    h_start = get_start_idx(orig_h, target_h)
    w_start = get_start_idx(orig_w, target_w)
    cropped_ct = ct_array[d_start:d_start + target_d, h_start:h_start + target_h, w_start:w_start + target_w]
    pad_d = max(target_d - cropped_ct.shape[0], 0)
    pad_h = max(target_h - cropped_ct.shape[1], 0)
    pad_w = max(target_w - cropped_ct.shape[2], 0)

    cropped_ct = np.pad(
        cropped_ct,
        pad_width=((0, pad_d), (0, pad_h), (0, pad_w)),  # (前面填充 0，后面填充差值)
        mode='constant',
        constant_values=0
    )
    assert cropped_ct.shape == target_size
    return cropped_ct

class pretrain_dataset_uni(Dataset):
    """自定义数据集"""

    def __init__(self, lab_dict, size,batch_size,transform_train):
        super(pretrain_dataset_uni,self).__init__()
        self.batch_size = batch_size
        self.filenames = lab_dict
        self.size = size
        self.transform = transform_train

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        img_path = self.filenames[i]
        img_name = os.path.basename(img_path)
        # print(img_name)
        if img_name[-3:] != 'png':
            img = sitk.ReadImage(img_path)
            img = sitk.GetArrayFromImage(img).astype(np.float32)
            img = torch.from_numpy(randomcrop(img, self.size))
        else:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            # print(img.shape)
        return img, img_name
    
class pretrain_dataset(Dataset):
    """自定义数据集"""

    def __init__(self, ct_data_path,wsi_data_path, lab_dict, size,batch_size,transform_train):
        super(pretrain_dataset,self).__init__()
        self.ct_path = ct_data_path
        self.wsi_data_path = wsi_data_path
        # self.images_ind = list(lab_dict.keys())
        self.batch_size = batch_size
        self.ct_filenames = lab_dict['ct']
        self.wsi_filenames = lab_dict['wsi']
        self.files = []
        self.files.extend([0] * (len(self.ct_filenames) // (batch_size)))
        self.files.extend([1] * (len(self.wsi_filenames) // (batch_size)))
        self.size = size
        self.transform = transform_train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        data_loader_index = self.files[i]
        if data_loader_index == 0:
            selected_keys = np.random.choice(self.ct_filenames, self.batch_size, True, None)
            image3Ds = torch.zeros(size=(self.batch_size, 1, self.size[0], self.size[1], self.size[-1]))
            for j, img_path in enumerate(selected_keys):
                img_root = img_path
                img = sitk.ReadImage(img_root)
                img = sitk.GetArrayFromImage(img).astype(np.float32)
                img = torch.from_numpy(img)
                image3Ds[j]=img
            return image3Ds, img_path
        
        elif data_loader_index == 1:

            selected_keys = np.random.choice(self.wsi_filenames, self.batch_size, True, None)
            image2Ds = torch.zeros(size=(self.batch_size, 3, 256, 256))
            for j, img_path in enumerate(selected_keys):
                image2D = Image.open(os.path.join(self.wsi_data_path, img_path)).convert("RGB")
                image2D_trans = self.transform(image2D)
                image2Ds[j] = image2D_trans

            return image2Ds, img_path
        
        
        #     image_path = os.path.join(
        #     self.root, 'image', self.images[index])
        
        #     img = Image.open(image_path).convert("RGB")

            
        # # texts = "一张位于{}的{}ct图像。".format(self.tumor_location[i], self.histopathology[i])
        
class kmean_dataset(Dataset):
    """自定义数据集"""

    def __init__(self,wsi_data_path, size,transform_train):
        super(kmean_dataset,self).__init__()
        self.wsi_data_path = wsi_data_path
        self.transform = transform_train
        self.size = size

    def __len__(self):
        return len(self.wsi_data_path)

    def __getitem__(self, i):
        img_path = self.wsi_data_path[i]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image2D = Image.fromarray(img).convert("RGB")
        # image2D = Image.open(img_path).convert("RGB")
        image2D_trans = self.transform(image2D)


        return image2D_trans, img_path


class MAE_class(Dataset):
    """自定义数据集"""

    def __init__(self, img_ind, label, ct_data_path, size, task, transform=True):
        super(MAE_class,self).__init__()

        # self.images_idx = list(lab_dict.keys())
        self.img_label = label
        self.images_idx = img_ind
        self.task = task
        self.ct_path = ct_data_path
        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.images_idx)
    
    def __getitem__(self, i):

        tumormin_id = self.images_idx[i]
        img_root = os.path.join(self.ct_path, tumormin_id)
        # if '_' in tumormin_id:
        #     img_root = os.path.join(self.ct_path, tumormin_id.split('_')[0]+'.nii.gz')
        # else:
        #     img_root = os.path.join(self.ct_path, tumormin_id)
        img = sitk.ReadImage(img_root)

        img = sitk.GetArrayFromImage(img).astype(np.float32)
        img = torch.from_numpy(img)

        label = self.img_label[tumormin_id][self.task]
        
        return img, label,tumormin_id  #,os.path.basename(self.images_ind[i])


class test_class(Dataset):
    """自定义数据集"""

    def __init__(self, img_ind, ct_data_path, size, task_name, transform=True):
        super(test_class,self).__init__()

        self.img_label = img_ind
        self.images_idx = list(self.img_label.keys())
        self.ct_path = ct_data_path
        self.size = size
        self.transform = transform
        self.task_name = task_name

    def __len__(self):
        return len(self.images_idx)
    
    def __getitem__(self, i):
        # patient_id = self.images_idx[i]
        # tumormin_id = self.all_label[patient_id]['tuominID']
        tumormin_id = self.images_idx[i]
        img_root = os.path.join(self.ct_path,'image', tumormin_id)
        roi_root = os.path.join(self.ct_path,'roi', tumormin_id)
        img = sitk.ReadImage(img_root)
        roi = sitk.ReadImage(roi_root)

        img = sitk.GetArrayFromImage(img).astype(np.float32)
        roi = sitk.GetArrayFromImage(roi).astype(np.float32)
        img = torch.from_numpy(img)
        roi = torch.from_numpy(roi)

        label = self.img_label[tumormin_id][self.task_name]
        
        return img,roi, label,tumormin_id  #,os.path.basename(self.images_ind[i])

class MAE_fusion_class(Dataset):
    """自定义数据集"""

    def __init__(self, img_ind, label, ct_data_path, size, task, transform=True):
        super(MAE_fusion_class,self).__init__()

        self.images_idx = img_ind
        self.ct_path = ct_data_path
        self.task = task
        self.img_label = label
        self.size = size
        self.text_embeddings = joblib.load('/cache/yangjing/main_files/CRCFound2/datas/all_CT/class_data_split/new_text.pkl')

    def __len__(self):
        return len(self.images_idx)
    
    def __getitem__(self, i):
        # patient_id = self.images_idx[i]
        # tumormin_id = self.all_label[patient_id]['tuominID']
        tumormin_id = self.images_idx[i]
        img_root = os.path.join(self.ct_path,'image', tumormin_id)
        # roi_root = os.path.join(self.ct_path,'roi', tumormin_id)
        img = sitk.ReadImage(img_root)
        # roi = sitk.ReadImage(roi_root)
        img = sitk.GetArrayFromImage(img).astype(np.float32)
        # roi = sitk.GetArrayFromImage(roi).astype(np.float32)
        # img,roi = randomcrop(img,roi,self.size)
        img = torch.from_numpy(img)
        # roi = torch.from_numpy(roi)
        text= self.text_embeddings[tumormin_id]['embeddings']
        if text.shape[0] > 1000:
            text = torch.mean(text[:1000,::], dim=0)
        else:
            text = torch.mean(text, dim=0)

        # text = self.all_label[tumormin_id]['image_findings'][:510]

        label = self.img_label[tumormin_id][self.task]
        
        return img, text, label, tumormin_id 
        
class KAD_Survival(Dataset):
    """自定义数据集"""

    def __init__(self, img_ind, ct_data_path, lab_dict, size, task_name):
        super(KAD_Survival,self).__init__()

        self.images_idx = img_ind
        self.ct_path = ct_data_path
        self.patient_slide = {}
        self.all_label = lab_dict
        self.size = size
        self.labels = np.array([self.all_label[i][f'{task_name}.event'] for i in self.images_idx])
        self.task_name = task_name
    def __len__(self):
        return len(self.images_idx)
    
    def __getitem__(self, i):
        # patient_id = self.images_idx[i]
        # tumormin_id = self.all_label[patient_id]['tuominID']
        tumormin_id = self.images_idx[i]
        img_root = os.path.join(self.ct_path, tumormin_id)
        # roi_root = os.path.join(self.ct_path,'roi', tumormin_id)
        img = sitk.ReadImage(img_root)
        # roi = sitk.ReadImage(roi_root)

        img = sitk.GetArrayFromImage(img).astype(np.float32)
        # roi = sitk.GetArrayFromImage(roi).astype(np.float32)
        # img,roi = randomcrop(img,roi,self.size)
        img = torch.from_numpy(img)
        # roi = torch.from_numpy(roi)
        
        event =  torch.tensor(self.all_label[tumormin_id][f'{self.task_name}.event'])
        delay =  torch.tensor(self.all_label[tumormin_id][f'{self.task_name}.delay'])

        # ct_caption = "一张位于{}的{}ct图像。".format(self.all_label[patient_id]['tumor_location'], self.all_label[patient_id]['histopathology'])
 
        return img, event, delay, tumormin_id  #,os.path.basename(self.images_ind[i])

class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samplers):
        # super(BalancedBatchSampler, self).__init__()
        self.labels = np.array(labels)
        self.labels_set = list(set(self.labels))
        self.labels_to_indices = {
            label: np.where(self.labels == label)[0] for label in self.labels_set
        }
        for i in self.labels_set:
            np.random.shuffle(self.labels_to_indices[i])
 
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samplers
        self.batch_size = self.n_classes * self.n_samples
        self.n_dataset = len(self.labels)
 
    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                if self.used_label_indices_count[class_] + self.n_samples < len(self.labels_to_indices[class_]):
                    temp = self.labels_to_indices[class_][self.used_label_indices_count[class_]: self.used_label_indices_count[class_] + self.n_samples]
                    self.used_label_indices_count[class_] += self.n_samples
                else:
                    temp = self.labels_to_indices[class_][self.used_label_indices_count[class_]: len(self.labels_to_indices[class_])-1]
                    np.random.shuffle(self.labels_to_indices[class_])
                    temp =np.concatenate([temp,self.labels_to_indices[class_][:self.n_samples-len(temp)]],0)
                    self.used_label_indices_count[class_] = self.n_samples-len(temp)
                indices.extend(temp)
            # print(indices)
            yield indices
            self.count += self.n_classes * self.n_samples
 
    def __len__(self):
        return self.n_dataset // self.batch_size
