#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import segmentation_models_pytorch as smp
from sklearn.preprocessing import LabelEncoder
import random
# image operations
from PIL import Image
import os
import glob
import cv2
from patchify import patchify,unpatchify
import matplotlib
# %%
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image)
#%%


# %%
#! load image into numpy

class load_images(Dataset):
    
    def __init__(self):
        image_directory = 'Train_Image_Patches/' 
        mask_directory = 'Train_Mask_Patches/'

        image_dataset = []
        mask_dataset = []


        SIZE = 256        

        images = os.listdir(image_directory)
        for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
            print(image_name)
            if (image_name.split('.')[1] == 'png'):
                #print(image_directory+image_name)
                image = cv2.imread(image_directory+image_name)
                image = Image.fromarray(image)
                image = image.resize((SIZE, SIZE))
                image_dataset.append(np.array(image))
        
 #       masks = os.listdir(mask_directory)
#        for i, image_name in enumerate(masks):
#            if (image_name.split('.')[1] == 'png'):
                #image_mask = cv2.imread(mask_directory+image_name.replace('.png','_mask.png'))
                image = cv2.imread(mask_directory+image_name)
                image_mask = Image.fromarray(image_mask)
                image_mask = image_mask.resize((SIZE, SIZE))
                mask_dataset.append(np.array(image_mask))

        self.image_dataset = image_dataset
        self.mask_dataset = mask_dataset

    def __getitem__(self,idx):
        
        return self.image_dataset[idx] , self.mask_dataset[idx]
        
    
    def __len__(self):
        return len(self.image_dataset)

train_set = load_images()
# %%
def show_sample(img, target):
    plt.imshow(torch.squeeze(img, axis=0), cmap='gray')
    print('Label:', target)

show_sample(*train_set[0])

# %%

#! sanity check
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(train_set[2][0], (256, 256)))
plt.subplot(122)
plt.imshow(np.reshape(train_set[2][1], (256, 256)))
plt.show()

# %%
#! patching images and saving

image_train = cv2.imread('Valid_Image/2edf6cc7696c91d86eb86413fa9c82d7.png')
patches = patchify(image_train,(256,256,3),step=256)
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
            single_patch = patches[i,j,:,:]
            #print(single_patch.shape)
            cv2.imwrite('Valid_Image_Patches/Val_A_'+str(i)+'_'+str(j)+'.png',single_patch.squeeze())

# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(single_patch.squeeze())

#? Repeat same with mask
#%%

image = cv2.imread('Valid_Mask/2edf6cc7696c91d86eb86413fa9c82d7_mask.png')
patches = patchify(image,(256,256,3),step=256)
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
            single_patch = patches[i,j,:,:]
            #print(single_patch.shape)
            cv2.imwrite('Valid_Mask_Patches/Val_A_'+str(i)+'_'+str(j)+'.png',single_patch.squeeze())

# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(single_patch.squeeze())

#! unpatchify:
# predicted_patches = np.array(predicted_patches)

# predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 256,256) )
# reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)

# %%
#! Sanity check, taking sample image

sample_image_train = cv2.imread('Train_Image_Patches/Train_B_6_8.png',0)
sample_image_mask = cv2.imread('Train_Mask_Patches/Train_B_6_8.png',0)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(sample_image_train,cmap='gray')

plt.subplot(122)
plt.imshow(sample_image_mask,cmap='gray')

# %%
#! figuring how to use mask

sample_image_mask_grey = cv2.cvtColor(sample_image_mask,cv2.COLOR_BGR2GRAY)

#%%
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(sample_image_train)

plt.subplot(122)
plt.imshow(sample_image_mask_grey,cmap='gray')


#%%
#* lets try from this https://github.com/RespectKnowledge/Multiclass-2D-Medical-Image-Segmentation/blob/main/Segmentationmodels_lab.ipynb

#! image,mask into dataloader
class load_image_patch(Dataset):
    """Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['Black', 'Grey', 'Blue', 'Green', 'Yellow','Red']

    
#    label_map = {'Epithelial':1, RED
#             'Lymphocyte':2, YELLOW
#             'Macrophage':4, GREEN
#             'Neutrophil':3, BLUE
#            }
    
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        
        # Unsorted
        #self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        #self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # Sorted
        self.images_fps = sorted([os.path.join(images_dir, image_id) for image_id in self.ids])
        self.masks_fps = sorted([os.path.join(masks_dir, image_id) for image_id in self.ids])

        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls) for cls in classes] # cls used instead of cls.lower()

        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # Read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = skimage.io.imread(self.images_fps[i])
        
        
        mask = cv2.imread(self.masks_fps[i], 0)
        #mask = skimage.io.imread(self.masks_fps[i])
        
        
        # Extract certain classes from mask
        #create a list of np arrays, with True, False for each class
        masks = [(mask == v) for v in self.class_values] 
        # or rather downselect certain class


        # automatic 1 hot
        mask = np.stack(masks, axis=-1).astype('float')
        
        # Add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            
            mask = np.concatenate((mask, background), axis=-1)
        
        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

# %%
image_dataset = []
mask_dataset = []
#! loading in batch
images = os.listdir('Train_Image_Patches/')
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'png'):
        #print(image_directory+image_name)
        image = cv2.imread('Train_Image_Patches/'+image_name,1)
        #image = Image.fromarray(image)
#        image = image.resize((SIZE, SIZE))
#        image = (image.astype('float32'))/255
        image_dataset.append(image)

#       masks = os.listdir(mask_directory)
#        for i, image_name in enumerate(masks):
#            if (image_name.split('.')[1] == 'png'):
        #image_mask = cv2.imread(mask_directory+image_name.replace('.png','_mask.png'))
        image_mask = cv2.imread('Train_Mask_Patches/'+image_name,0)
#        image_mask = Image.fromarray(image_mask)
#        image_mask = (image_mask.astype('float32'))/255
#        image_mask = image_mask.resize((SIZE, SIZE))
        mask_dataset.append(image_mask)

# %%

train_images = np.array(image_dataset)
train_masks = np.array(mask_dataset)

# %%
np.unique(train_masks)

# %%
#? Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)
# Black, Blue, Green, Gray
# 0, 1, 2, 3

# %%

train_mask_input = np.expand_dims(train_masks_encoded_original_shape,axis=3)

# %%
#! convert into categorical
n_classes = 4
from keras.utils import to_categorical
#train_masks_cat = to_categorical(train_mask_input, num_classes=n_classes)
train_masks_categorical = train_masks_cat.reshape((train_mask_input.shape[0], train_mask_input.shape[1], train_mask_input.shape[2], n_classes))

#https://discuss.pytorch.org/t/is-there-something-like-keras-utils-to-categorical-in-pytorch/5960
# https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py#L9-L37
# or sklearns later

#? FINALLY DATASET READY!!!!!



# %%
#! data processing for model
loss = smp.utils.losses.DiceLoss()

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

activation = 'softmax'
#%%

preprocess_input = get_preprocessing_fn('resnet34',pretrained='imagenet')



# %%


model_unet_rn34 = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=4,                      # model output channels (number of classes in your dataset)
, activation=activation
)
# %%

# %%

# %%

# %%

# %%
