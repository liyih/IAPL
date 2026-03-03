from augmix_ops import augmentations
import numpy as np
import torch
import torchvision.transforms as transforms
import pdb
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from PIL import Image
import math, random, io
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    
def get_preaugment(do_resize, dataset):

    assert dataset in ['UniversalFakeDetect', 'GenImage', 'Chameleon']

    if dataset == 'UniversalFakeDetect':
        if do_resize:
            return transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.765, 0.765), ratio=(1., 1.)), # 0.765 = 224**2 / 256**2
                    transforms.RandomHorizontalFlip(),
                ])
        else:
            return transforms.Compose([
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                ])
    
    elif dataset == 'GenImage':
        if do_resize:
            return transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.765, 0.765), ratio=(1., 1.)), # 0.765 = 224**2 / 256**2
                    transforms.RandomHorizontalFlip(),
                ])
        else:
            return transforms.Compose([
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                ])
        
    elif dataset == 'Chameleon':

        return transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(1., 1.)),
                    transforms.RandomHorizontalFlip(),
                ])

def augmix(image, preprocess, aug_list, dataset, severity=1, do_resize=True):
    preaugment = get_preaugment(do_resize, dataset)
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix

class AugMixAugmenter(object):
    def __init__(self, base_transform, img_resolution, crop_resolution, preprocess, n_views=2, augmix=False, dataset='UniversalFakeDetect',
                    severity=1, threshold=2):
        self.base_transform = base_transform
        self.crop_resolution = crop_resolution
        self.preprocess = preprocess
        self.n_views = n_views
        self.img_resolution = img_resolution
        if augmix:
            self.aug_list = augmentations
        else:
            self.aug_list = []
        self.severity = severity
        self.threshold = threshold
        self.dataset = dataset
        
    def __call__(self, x):
        # x.save("aug_res/origin.png")
        if (max(x.size) >= (self.crop_resolution + 32)) and (min(x.size) >= self.crop_resolution): # 在不resize的情况下至少可以产生32种不同的视角，32是tta的bz大小。
            do_resize = False
        else:
            do_resize = True
        image = self.preprocess(self.base_transform(x))

        views = [augmix(x, self.preprocess, self.aug_list, self.dataset, self.severity, do_resize) for _ in range(self.n_views)] # 全局的view加上local的view。
        return [image] + views