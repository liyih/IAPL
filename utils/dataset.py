import os, io
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
import sys
sys.path.append('..')
from augmix import AugMixAugmenter
import math
from PIL import Image
import cv2, random
from PIL import Image, ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
    
class Dataset_Creator:
    def __init__(self, dataset_path, batch_size, num_workers=0, img_resolution=256, crop_resolution=224):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.all_dataset = {
            "train": ['car' 'cat' 'chair' 'horse'] + ['SDv14'],
            "val": ['car' 'cat' 'chair' 'horse'] + ['SDv14'],
            "test": ['crn', 'cyclegan', 'dalle', 'biggan', 'deepfake', 'gaugan', 'glide_50_27', 'glide_100_10', 'glide_100_27', 'guided', 'imle', 'ldm_100', 'ldm_200', 'ldm_200_cfg', 'progan', 'san', 'seeingdark', 'stargan', 'stylegan'] + ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong'] + ['Chameleon'],
            "tta": ['crn', 'cyclegan', 'dalle', 'biggan', 'deepfake', 'gaugan', 'glide_50_27', 'glide_100_10', 'glide_100_27', 'guided', 'imle', 'ldm_100', 'ldm_200', 'ldm_200_cfg', 'progan', 'san', 'seeingdark', 'stargan', 'stylegan'] + ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong'] + ['Chameleon'],
        }

        base_transform = transforms.Compose([
        transforms.Resize((img_resolution, img_resolution)),
        transforms.CenterCrop(crop_resolution)])

        preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
        tta_data_transform = AugMixAugmenter(base_transform, img_resolution, crop_resolution, preprocess, n_views=batch_size-1, augmix=False, dataset='UniversalFakeDetect')

        self.transforms = {
            "train": transforms.Compose([
                transforms.Resize((img_resolution, img_resolution)),
                transforms.RandomCrop(crop_resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            "val": transforms.Compose([
                transforms.Resize((img_resolution, img_resolution)),
                transforms.CenterCrop(crop_resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            "test": transforms.Compose([
                transforms.Resize((img_resolution, img_resolution)),
                transforms.CenterCrop(crop_resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            "tta": tta_data_transform
        }
        
    
    def build_dataset(self, spilt_dataset, selected_subsets="all"):
        assert spilt_dataset in self.all_dataset.keys()

        if selected_subsets == "all":
            selected_subsets = self.all_dataset[spilt_dataset]
        else:
            assert isinstance(selected_subsets, list)
            for subset in selected_subsets:
                # assert subset in self.all_dataset[spilt_dataset]
                pass

        
        sub_datasets = []
        for subset in selected_subsets:
            if spilt_dataset == 'tta':
                subset_path = os.path.join(self.dataset_path, 'test', subset)
            else:
                subset_path = os.path.join(self.dataset_path, spilt_dataset, subset)
            # identify multi-classes subset
            
            if "0_real" in os.listdir(subset_path) and "1_fake" in os.listdir(subset_path):
                sub_datasets.append(ImageFolder(
                    subset_path,
                    self.transforms[spilt_dataset]
                ))
            elif (spilt_dataset == "test") or (spilt_dataset == "tta"):
                tmp_datasets = []
                for sub_class in os.listdir(subset_path):
                    tmp_datasets.append(ImageFolder(
                        os.path.join(subset_path, sub_class),
                        self.transforms[spilt_dataset]
                    ))
                sub_datasets.append(ConcatDataset(tmp_datasets))
            else:
                for sub_class in os.listdir(subset_path):
                    sub_datasets.append(ImageFolder(
                        os.path.join(subset_path, sub_class),
                        self.transforms[spilt_dataset]
                    ))
        
        if spilt_dataset == "test":
            return sub_datasets, selected_subsets
        
        if spilt_dataset == "tta":
            return sub_datasets, selected_subsets
        
        return ConcatDataset(sub_datasets)

def translate_duplicate(img, cropSize):
    if min(img.size) < cropSize:
        width, height = img.size
        
        new_width = width * math.ceil(cropSize/width)
        new_height = height * math.ceil(cropSize/height)
        
        new_img = Image.new('RGB', (new_width, new_height))
        for i in range(0, new_width, width):
            for j in range(0, new_height, height):
                new_img.paste(img, (i, j))
        return new_img
    else:
        return img
    
class Dataset_Creator_GenImage:
    def __init__(self, dataset_path, batch_size, num_workers=0, img_resolution=256, crop_resolution=224):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.all_dataset = {
            "train": ['car' 'cat' 'chair' 'horse'] + ['SDv14'],
            "val": ['car' 'cat' 'chair' 'horse'] + ['SDv14'],
            "test": ['crn', 'cyclegan', 'dalle', 'biggan', 'deepfake', 'gaugan', 'glide_50_27', 'glide_100_10', 'glide_100_27', 'guided', 'imle', 'ldm_100', 'ldm_200', 'ldm_200_cfg', 'progan', 'san', 'seeingdark', 'stargan', 'stylegan'] + ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong'] + ['Chameleon'],
            "tta": ['crn', 'cyclegan', 'dalle', 'biggan', 'deepfake', 'gaugan', 'glide_50_27', 'glide_100_10', 'glide_100_27', 'guided', 'imle', 'ldm_100', 'ldm_200', 'ldm_200_cfg', 'progan', 'san', 'seeingdark', 'stargan', 'stylegan'] + ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong'] + ['Chameleon'],
        }

        base_transform = transforms.Compose([
        transforms.Resize((img_resolution, img_resolution)),
        transforms.CenterCrop(crop_resolution)])

        preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276])
        ])
    
        tta_data_transform = AugMixAugmenter(base_transform, img_resolution, crop_resolution, preprocess, n_views=batch_size-1, augmix=False , dataset='GenImage')

        self.transforms = {
            "train": transforms.Compose([
                transforms.Lambda(lambda img: translate_duplicate(img, crop_resolution)),
                transforms.RandomCrop(crop_resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            "val": transforms.Compose([
                transforms.Lambda(lambda img: translate_duplicate(img, crop_resolution)),
                transforms.CenterCrop(crop_resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276]),
            ]),
            "test": transforms.Compose([
                transforms.Lambda(lambda img: translate_duplicate(img, crop_resolution)),
                transforms.CenterCrop(crop_resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276]),
            ]),
            "tta": tta_data_transform
        }
        
    
    def build_dataset(self, spilt_dataset, selected_subsets="all"):
        assert spilt_dataset in self.all_dataset.keys()

        if selected_subsets == "all":
            selected_subsets = self.all_dataset[spilt_dataset]
        else:
            assert isinstance(selected_subsets, list)
            for subset in selected_subsets:
                # assert subset in self.all_dataset[spilt_dataset]
                pass

        
        sub_datasets = []
        for subset in selected_subsets:
            if spilt_dataset == 'tta':
                subset_path = os.path.join(self.dataset_path, 'test', subset)
            else:
                subset_path = os.path.join(self.dataset_path, spilt_dataset, subset)
            # identify multi-classes subset
            
            if "0_real" in os.listdir(subset_path) and "1_fake" in os.listdir(subset_path):
                sub_datasets.append(ImageFolder(
                    subset_path,
                    self.transforms[spilt_dataset]
                ))
            elif (spilt_dataset == "test") or (spilt_dataset == "tta"):
                tmp_datasets = []
                for sub_class in os.listdir(subset_path):
                    tmp_datasets.append(ImageFolder(
                        os.path.join(subset_path, sub_class),
                        self.transforms[spilt_dataset]
                    ))
                sub_datasets.append(ConcatDataset(tmp_datasets))
            else:
                for sub_class in os.listdir(subset_path):
                    sub_datasets.append(ImageFolder(
                        os.path.join(subset_path, sub_class),
                        self.transforms[spilt_dataset]
                    ))
        
        if spilt_dataset == "test":
            return sub_datasets, selected_subsets
        
        if spilt_dataset == "tta":
            return sub_datasets, selected_subsets
        
        return ConcatDataset(sub_datasets)
    
class Dataset_Creator_Chameleon:
    def __init__(self, dataset_path, batch_size, num_workers=0, img_resolution=256, crop_resolution=224):

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.all_dataset = {
            "train": ['car' 'cat' 'chair' 'horse'] + ['SDv14'],
            "val": ['car' 'cat' 'chair' 'horse'] + ['SDv14'],
            "test": ['crn', 'cyclegan', 'dalle', 'biggan', 'deepfake', 'gaugan', 'glide_50_27', 'glide_100_10', 'glide_100_27', 'guided', 'imle', 'ldm_100', 'ldm_200', 'ldm_200_cfg', 'progan', 'san', 'seeingdark', 'stargan', 'stylegan'] + ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong'] + ['Chameleon'],
            "tta": ['crn', 'cyclegan', 'dalle', 'biggan', 'deepfake', 'gaugan', 'glide_50_27', 'glide_100_10', 'glide_100_27', 'guided', 'imle', 'ldm_100', 'ldm_200', 'ldm_200_cfg', 'progan', 'san', 'seeingdark', 'stargan', 'stylegan'] + ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong'] + ['Chameleon'],
        }

        base_transform = transforms.Compose([
        transforms.Resize((img_resolution, img_resolution)),
        transforms.CenterCrop(crop_resolution)])

        preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
    
        tta_data_transform = AugMixAugmenter(base_transform, img_resolution, crop_resolution, preprocess, n_views=batch_size-1, augmix=False, dataset='Chameleon')
        self.transforms = {
            "train": transforms.Compose([
                transforms.Resize((img_resolution, img_resolution)),
                transforms.RandomCrop(crop_resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            "val": transforms.Compose([
                transforms.Resize((img_resolution, img_resolution)),
                transforms.CenterCrop(crop_resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            "test": transforms.Compose([
                transforms.Resize((img_resolution, img_resolution)),
                transforms.CenterCrop(crop_resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            "tta": tta_data_transform
        }
        
    
    def build_dataset(self, spilt_dataset, selected_subsets="all"):
        assert spilt_dataset in self.all_dataset.keys()

        if selected_subsets == "all":
            selected_subsets = self.all_dataset[spilt_dataset]
        else:
            assert isinstance(selected_subsets, list)
            for subset in selected_subsets:
                # assert subset in self.all_dataset[spilt_dataset]
                pass
        
        sub_datasets = []
        
        for subset in selected_subsets:
            if spilt_dataset == 'tta':
                subset_path = os.path.join(self.dataset_path, 'test', subset)
            else:
                subset_path = os.path.join(self.dataset_path, spilt_dataset, subset)
            # identify multi-classes subset
            
            if "0_real" in os.listdir(subset_path) and "1_fake" in os.listdir(subset_path):
                sub_datasets.append(ImageFolder(
                    subset_path,
                    self.transforms[spilt_dataset]
                ))
            elif (spilt_dataset == "test") or (spilt_dataset == "tta"):
                tmp_datasets = []
                for sub_class in os.listdir(subset_path):
                    tmp_datasets.append(ImageFolder(
                        os.path.join(subset_path, sub_class),
                        self.transforms[spilt_dataset]
                    ))
                sub_datasets.append(ConcatDataset(tmp_datasets))
            else:
                for sub_class in os.listdir(subset_path):
                    sub_datasets.append(ImageFolder(
                        os.path.join(subset_path, sub_class),
                        self.transforms[spilt_dataset]
                    ))
        
        if spilt_dataset == "test":
            return sub_datasets, selected_subsets
        
        if spilt_dataset == "tta":
            return sub_datasets, selected_subsets
        
        return ConcatDataset(sub_datasets)
    
class Dataset_Creator_Chameleon_SD:
    def __init__(self, dataset_path, batch_size, num_workers=0, img_resolution=256, crop_resolution=224):

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.all_dataset = {
            "train": ['car' 'cat' 'chair' 'horse'] + ['SDv14'],
            "val": ['car' 'cat' 'chair' 'horse'] + ['SDv14'],
            "test": ['crn', 'cyclegan', 'dalle', 'biggan', 'deepfake', 'gaugan', 'glide_50_27', 'glide_100_10', 'glide_100_27', 'guided', 'imle', 'ldm_100', 'ldm_200', 'ldm_200_cfg', 'progan', 'san', 'seeingdark', 'stargan', 'stylegan'] + ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong'] + ['Chameleon'],
            "tta": ['crn', 'cyclegan', 'dalle', 'biggan', 'deepfake', 'gaugan', 'glide_50_27', 'glide_100_10', 'glide_100_27', 'guided', 'imle', 'ldm_100', 'ldm_200', 'ldm_200_cfg', 'progan', 'san', 'seeingdark', 'stargan', 'stylegan'] + ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong'] + ['Chameleon'],
        }

        base_transform = transforms.Compose([
        transforms.Resize((img_resolution, img_resolution)),
        transforms.CenterCrop(crop_resolution)])

        preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276])
                    ])
    
        tta_data_transform = AugMixAugmenter(base_transform, img_resolution, crop_resolution, preprocess, n_views=batch_size-1, augmix=False, dataset='Chameleon')
        self.transforms = {
            "train": transforms.Compose([
                transforms.Lambda(lambda img: translate_duplicate(img, crop_resolution)),
                transforms.RandomCrop(crop_resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276]),
            ]),
            "val": transforms.Compose([
                transforms.Lambda(lambda img: translate_duplicate(img, crop_resolution)),
                transforms.CenterCrop(crop_resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276]),
            ]),
            "test": transforms.Compose([
                transforms.Lambda(lambda img: translate_duplicate(img, crop_resolution)),
                transforms.CenterCrop(crop_resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276]),
            ]),
            "tta": tta_data_transform
        }
        
    
    def build_dataset(self, spilt_dataset, selected_subsets="all"):
        assert spilt_dataset in self.all_dataset.keys()

        if selected_subsets == "all":
            selected_subsets = self.all_dataset[spilt_dataset]
        else:
            assert isinstance(selected_subsets, list)
            for subset in selected_subsets:
                # assert subset in self.all_dataset[spilt_dataset]
                pass
        
        sub_datasets = []
        
        for subset in selected_subsets:
            if spilt_dataset == 'tta':
                subset_path = os.path.join(self.dataset_path, 'test', subset)
            else:
                subset_path = os.path.join(self.dataset_path, spilt_dataset, subset)
            # identify multi-classes subset
            
            if "0_real" in os.listdir(subset_path) and "1_fake" in os.listdir(subset_path):
                sub_datasets.append(ImageFolder(
                    subset_path,
                    self.transforms[spilt_dataset]
                ))
            elif (spilt_dataset == "test") or (spilt_dataset == "tta"):
                tmp_datasets = []
                for sub_class in os.listdir(subset_path):
                    tmp_datasets.append(ImageFolder(
                        os.path.join(subset_path, sub_class),
                        self.transforms[spilt_dataset]
                    ))
                sub_datasets.append(ConcatDataset(tmp_datasets))
            else:
                for sub_class in os.listdir(subset_path):
                    sub_datasets.append(ImageFolder(
                        os.path.join(subset_path, sub_class),
                        self.transforms[spilt_dataset]
                    ))
        
        if spilt_dataset == "test":
            return sub_datasets, selected_subsets
        
        if spilt_dataset == "tta":
            return sub_datasets, selected_subsets
        
        return ConcatDataset(sub_datasets)