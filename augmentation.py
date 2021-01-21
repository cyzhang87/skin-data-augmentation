"""
注意：
本工程还修改了C:\cyzhang\project\lidc_classify_regression\src\venv\Lib\site-packages\imgaug\augmenters\meta.py
"""
from auglib.augmentation import Augmentations
from auglib.dataset_loader import MyDataset
from torchvision.utils import save_image
from tqdm import tqdm
import os

root_dir = 'C:/cyzhang/project/ddw_skin_project/isic_data/ISIC_2017/'
img_dir = root_dir + 'ISIC-2017_Training_Data/'
seg_dir = root_dir + 'ISIC-2017_Training_Part1_GroundTruth/'

aug = [# 0 B: Saturation, Contrast, and Brightness
       {"color_contrast": 0.3,
        "color_saturation": 0.3,
        "color_brightness": 0.3,
        "color_hue": 0,
        "rotation": 0,
        "scale": (1, 1),
        "shear": 0,
        "vflip": False,
        "hflip": False,
        "random_crop": False,
        'autoaugment': False,
        'random_erasing': False,
        'piecewise_affine': False,
        'tps': False,
        'size': (224, 320),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]},
       #1 C: Saturation, Contrast, Brightness, and Hue
       {"color_contrast": 0.3,
        "color_saturation": 0.3,
        "color_brightness": 0.3,
        "color_hue": 0.1,
        "rotation": 0,
        "scale": (1, 1),
        "shear": 0,
        "vflip": False,
        "hflip": False,
        "random_crop": False,
        'autoaugment': False,
        'random_erasing': False,
        'piecewise_affine': False,
        'tps': False,
        'size': (224, 320),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]},
        #2 D: Affine
        {"color_contrast": 0,
        "color_saturation": 0,
        "color_brightness": 0,
        "color_hue": 0,
        "rotation": 90,
        "scale": (0.8, 1.2),
        "shear": 20,
        "vflip": False,
        "hflip": False,
        "random_crop": False,
        'autoaugment': False,
        'random_erasing': False,
        'piecewise_affine': False,
        'tps': False,
        'size': (224, 320),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]},
        #3 E: Flips
        {"color_contrast": 0,
        "color_saturation": 0,
        "color_brightness": 0,
        "color_hue": 0,
        "rotation": 0,
        "scale": (1, 1),
        "shear": 0,
        "vflip": True,
        "hflip": True,
        "random_crop": False,
        'autoaugment': False,
        'random_erasing': False,
        'piecewise_affine': False,
        'tps': False,
        'size': (224, 320),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]},
        #4 F: Random Crops
        {"color_contrast": 0,
        "color_saturation": 0,
        "color_brightness": 0,
        "color_hue": 0,
        "rotation": 0,
        "scale": (1, 1),
        "shear": 0,
        "vflip": False,
        "hflip": False,
        "random_crop": True,
        'autoaugment': False,
        'random_erasing': False,
        'piecewise_affine': False,
        'tps': False,
        'size': (224, 320),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]},
        #5 G: Random Erasing
        {"color_contrast": 0,
        "color_saturation": 0,
        "color_brightness": 0,
        "color_hue": 0,
        "rotation": 0,
        "scale": (1, 1),
        "shear": 0,
        "vflip": False,
        "hflip": False,
        "random_crop": False,
        'autoaugment': False,
        'random_erasing': True,
        'piecewise_affine': False,
        'tps': False,
        'size': (224, 320),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]},
        #6 H: Elastic
        {"color_contrast": 0,
        "color_saturation": 0,
        "color_brightness": 0,
        "color_hue": 0,
        "rotation": 0,
        "scale": (1, 1),
        "shear": 0,
        "vflip": False,
        "hflip": False,
        "random_crop": False,
        'autoaugment': False,
        'random_erasing': False,
        'piecewise_affine': False,
        'tps': True,
        'size': (224, 320),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]},
        #7 J:  F → D → E → C
        {"color_contrast": 0.3,
        "color_saturation": 0.3,
        "color_brightness": 0.3,
        "color_hue": 0.1,
        "rotation": 90,
        "scale": (0.8, 1.2),
        "shear": 20,
        "vflip": True,
        "hflip": True,
        "random_crop": True,
        'autoaugment': False,
        'random_erasing': False,
        'piecewise_affine': False,
        'tps': False,
        'size': (224, 320),
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]},
        # 8 K:  F → G → D → E → C
        {"color_contrast": 0.3,
         "color_saturation": 0.3,
         "color_brightness": 0.3,
         "color_hue": 0.1,
         "rotation": 90,
         "scale": (0.8, 1.2),
         "shear": 20,
         "vflip": True,
         "hflip": True,
         "random_crop": True,
         'autoaugment': False,
         'random_erasing': True,
         'piecewise_affine': False,
         'tps': False,
         'size': (224, 320),
         'mean': [0.5, 0.5, 0.5],
         'std': [0.5, 0.5, 0.5]},
         # 9 L: F → D → H → E → C
        {"color_contrast": 0.3,
         "color_saturation": 0.3,
         "color_brightness": 0.3,
         "color_hue": 0.1,
         "rotation": 90,
         "scale": (0.8, 1.2),
         "shear": 20,
         "vflip": True,
         "hflip": True,
         "random_crop": True,
         'autoaugment': False,
         'random_erasing': False,
         'piecewise_affine': False,
         'tps': True,
         'size': (224, 320),
         'mean': [0.5, 0.5, 0.5],
         'std': [0.5, 0.5, 0.5]},
       ]

augmented_dir = 'augment'

def save_images(dataset, aug_img_path, aug_seg_path, type=0):
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        file_name_split = os.path.splitext(dataset.img_filenames[i])
        image_name = file_name_split[0] + '_' + str(type) + file_name_split[1]
        img_path = os.path.join(aug_img_path, image_name)
        save_image(data[0], img_path)
        file_name_split = os.path.splitext(dataset.seg_filenames[i])
        image_name = file_name_split[0] + '_' + str(type) + file_name_split[1]
        img_path = os.path.join(aug_seg_path, image_name)
        save_image(data[1], img_path)

def augmentation(img_path, seg_path, type=0):
    augs = Augmentations(**aug[type])
    data = MyDataset(img_path, seg_path, img_transform=augs.tf_img_augment)
    aug_img_path = os.path.join(img_path, augmented_dir)
    if not os.path.exists(aug_img_path):
        os.makedirs(aug_img_path)
    aug_seg_path = os.path.join(seg_path, augmented_dir)
    if not os.path.exists(aug_seg_path):
        os.makedirs(aug_seg_path)

    save_images(data, aug_img_path, aug_seg_path, type)


if __name__ == '__main__':
    for i in range(len(aug)):
        augmentation(img_dir, seg_dir, type=i)
