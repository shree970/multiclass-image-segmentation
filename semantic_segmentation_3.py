# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
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

# %% [markdown]
# Seed Everthing for Reproducibility

# %%
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(1)

# %% [markdown]
# Since we are dealing with dataset of very high resolution, I thuoght the best approach would be sliding window approach.
# 
# Where we are creating patches of 256,256 from the dataset, and storing them in seperate folder

# %%
# Patchify

image_train = cv2.imread('Valid_Image/2edf6cc7696c91d86eb86413fa9c82d7.png')
patches = patchify(image_train,(256,256,3),step=256)
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
            single_patch = patches[i,j,:,:]
            #print(single_patch.shape)
            cv2.imwrite('Valid_Image_Patches/Val_A_'+str(i)+'_'+str(j)+'.png',single_patch.squeeze())


image = cv2.imread('Valid_Mask/2edf6cc7696c91d86eb86413fa9c82d7_mask.png')
patches = patchify(image,(256,256,3),step=256)
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
            single_patch = patches[i,j,:,:]
            #print(single_patch.shape)
            cv2.imwrite('Valid_Mask_Patches/Val_A_'+str(i)+'_'+str(j)+'.png',single_patch.squeeze())

# %% [markdown]
# For multiclass, instead of having individual colors as masks, better to scale them down to 0-5, as 6 classes.

# %%

def bin_image(mask):
    bins = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240])
    new_mask = np.digitize(mask, bins)
    return new_mask

def getSegmentationArr(image, classes, width, height):
    seg_labels = np.zeros((height, width, classes))
    img = image[:, : , 0]

    for c in range(classes):
        seg_labels[:, :, c] = (img == c ).astype(int)
    return seg_labels

def give_color_to_seg_img(seg, n_classes):
    
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)
    
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))

    return(seg_img)


#     image, mask = LoadImage(file, path)
#     mask_binned = bin_image(mask)
#     labels = getSegmentationArr(mask_binned, classes)

#     imgs.append(image)
#     segs.append(labels)

#     yield np.array(imgs), np.array(segs)

sample_image_train = cv2.imread('Train_Image_Patches/Train_B_6_8.png')
sample_image_train = cv2.cvtColor(sample_image_train,cv2.COLOR_BGR2RGB)

sample_image_mask = cv2.imread('Train_Mask_Patches/Train_B_6_8.png')
sample_image_mask = cv2.cvtColor(sample_image_mask,cv2.COLOR_BGR2RGB)

# %%

class slide_image_datagen_2(Dataset):
    def __init__(
        self, 
        images_dir, 
        masks_dir, 
        classes=None, 
        augmentation=None, 
        preprocessing=None,
        transform= None,
        custom_img_path= None,
        true_mask_for_testing = None
    ):

        self.images_dir= images_dir
        self.masks_dir=masks_dir
        self.ids = os.listdir(images_dir)

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        patches = patchify(image,(256,256,3),step=256)
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                    single_patch = patches[i,j,:,:]



        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls) for cls in classes]
        



        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.transform = transform
        self.custom_img_path = custom_img_path
        self.true_mask_for_testing = true_mask_for_testing

    def __getitem__(self):
        
    
    
    
    
    
    def __len__(self):
        return len(self.ids)

# %%

# %%

# %%

# %%
def encode_mask_pixel(mask_np,pixel_array):
    '''
    encodes pixel for mask
    '''
    for pixel in pixel_array:
        
        if pixel == 29:
            mask_np[mask_np == pixel] = 1
        
        if pixel == 76:
            mask_np[mask_np == pixel] = 2
        
        if pixel == 149:
            mask_np[mask_np == pixel] = 3
        
        if pixel == 150:
            mask_np[mask_np == pixel] = 4
        
        if pixel == 225:
            mask_np[mask_np == pixel] = 5
    
    return mask_np

# %% [markdown]
# Create our dataset using pytoch dataset class
# 
# Hardest part to figure out.
# 
# Now we load our image patches, one by one, along with masks
# 
# Image:
# 
# 1. Convert BGR to RGB
# 2. Transformations : Normalise, and totensor()
# 3. final shape : Batchx3x256x256
# 
# Mask: 
# 
# 1. Load as single channel
# 2. The values for classes were observed to be:
# 
#     Black : 0
# 
#     Blue : 29
# 
#     Red : 76
# 
#     Green: 149
# 
#     Gray: 150
# 
#     Yellow: 225
# 
# 3. Remapped the values to 0 → 5
# 4. one hot encoded the classes
# 5. Transformations: totensor()
# 6. Final shape: Batchx6x256x256
# 
# Finally, also added features for augementation transforms, model pre-processing, test_loading etc. as required.

# %%
# Dataset class
class slide_image(Dataset):
    """Slide images dataset. 
    
    Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['Black','Blue','Red','Green','Gray','Yellow']
    
    '''
    Black : 0
    Blue : 29
    Red : 76
    Green: 149
    Gray: 150
    Yellow: 225
    '''

    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            transform= None,
            custom_img_path= None,
            true_mask_for_testing = None
    ):
        self.images_dir= images_dir
        self.masks_dir=masks_dir
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.transform = transform
        self.custom_img_path = custom_img_path
        self.true_mask_for_testing = true_mask_for_testing
    
    def __getitem__(self, i):
        
        # read data
        #print(self.images_fps[i])
        if self.custom_img_path:
            image = cv2.imread(self.images_dir+'/'+self.custom_img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_dir+'/'+self.custom_img_path, 0)
        else:
            image = cv2.imread(self.images_fps[i])
        #image= cv2.imread('./Train_Image_Patches/Train_A_35_7.png')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(self.masks_fps[i])
        
            mask = cv2.imread(self.masks_fps[i], 0)


        #mask = cv2.imread('./Train_Mask_Patches/Train_A_35_7.png',0)
    
        # encoded pixel masks into 0 to 5
        
        # here it is encoding per image!!! will mess up masks

        # labelencoder = LabelEncoder()
        # h, w = mask.shape
        # print(np.unique(mask))
        # mask_encoded = labelencoder.fit_transform(mask.reshape(-1,1))
        # mask_encoded = mask_encoded.reshape(h, w)
        # mask = mask_encoded
        # print(np.unique(mask))

        # better define custom function to encode pixel
#        print(np.unique(mask))
        mask = encode_mask_pixel(mask_np=mask,pixel_array=np.unique(mask))
#        print(np.unique(mask))

        # extract certain classes from mask 
        masks = [(mask == v) for v in self.class_values] 
        
        # creates a 1-hot encoding for classes
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.transform:
            image = self.transform(image)
            mask =torch.tensor(mask.transpose(2,0,1),dtype=torch.float32)
#            image, mask = sample['image'], sample['mask']
        
        #image_tensor=torch.tensor(image.transpose(2,0,1),dtype=torch.float32)
        
        if self.true_mask_for_testing:
            mask = cv2.imread(self.masks_fps[i], 0)
            return image, mask

        return image, mask
        
    def __len__(self):
        return len(self.ids)

# %% [markdown]
# Defined paths, will change for colab

# %%
DATA_DIR = './'
x_train_dir = os.path.join(DATA_DIR, 'Train_Image_Patches')
y_train_dir = os.path.join(DATA_DIR, 'Train_Mask_Patches')

x_valid_dir = os.path.join(DATA_DIR, 'Valid_Image_Patches')
y_valid_dir = os.path.join(DATA_DIR, 'Valid_Mask_Patches')

# %% [markdown]
# Visualisation helper function

# %%

def visualize(**images):
    """PLot images in one row."""
    
    norm=plt.Normalize(0,5) # 6 classes
    map_name = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","red","green","gray","yellow"])
    #map_name = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red"])
    
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap=map_name, norm=norm)
    plt.show()


# %%
# sanity check

# defined initial transforms
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
train_dataset = slide_image(x_train_dir, y_train_dir, classes=['Black','Blue','Red','Green','Gray','Yellow'],custom_img_path = 'Train_A_1_13.png',transform=transform)

#image, mask=next(iter(train_dataset))
image, mask=train_dataset[60]
print(image.shape,mask.shape)
print(type(image),type(mask))
print(np.unique(image),np.unique(mask))


# %%
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

# %% [markdown]
# Checking the data.
# Here we are able to plot multiple segmentation maps, included example of Black, Blue, Grey.
# The "red" is just used for highlights

# %%
unorm = UnNormalize(mean =[0.5],std=[0.5])
image = unorm(image)
visualize(slide = image.numpy().transpose(1,2,0),mask_black=mask.numpy().transpose(1,2,0)[...,0].squeeze(),mask_blue = mask.numpy().transpose(1,2,0)[...,1].squeeze(),mask_grey = mask.numpy().transpose(1,2,0)[...,4].squeeze())

# %% [markdown]
# Loading dataloaders, and sanity checking, with basic transforms. Since we will be using pre-traiend backbone, the transformations will change

# %%

train_dataset = slide_image(x_train_dir, y_train_dir, classes=['Black','Blue','Red','Green','Gray','Yellow'],transform=transform)

val_dataset = slide_image(x_valid_dir, y_valid_dir, classes=['Black','Blue','Red','Green','Gray','Yellow'],transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=12)


# %%
train_image_batch,train_mask_batch=next(iter(train_loader))
print(train_image_batch.shape)
print(train_mask_batch.shape)
print('Train dataset size',len(train_dataset))
print('Valid dataset size',len(val_dataset))

# %% [markdown]
# Defined augmentations for model preprocessing pipeline

# %%
import albumentations as albu

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn: data normalization function 
        
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

# %% [markdown]
# Went on to search for various SOTA architectures, but first wanted to establish a minimal baseline. Hence used [segmenation_models](https://smp.readthedocs.io/en/latest/) given architectures  
# 
# Used transfer learning with UNET as decoder and encoder as ResNet32(that is all I could fit in my laptop). Utilised the pre-trained weights from "imagenet".  
# 
# Removed the head of ResNet, with final layer as fully convolutional, which is then passed to UNET encoder to upsample.  
# 
# Output layer of UNET was modified to give 6 feature maps as output with softmax. Same shape as label_mask earlier.
# 
# Model had total 24M parameters

# %%
#! data pre-processing and defining model

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES=['Black','Blue','Red','Green','Gray','Yellow']
ACTIVATION = 'softmax' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

model_unet_rn34 = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, 
    classes=len(CLASSES),                      # model output channels (number of classes in your dataset)
    activation=ACTIVATION
)

preprocess_input = get_preprocessing_fn(ENCODER,ENCODER_WEIGHTS)

# %% [markdown]
# The architecture

# %%
model_unet_rn34

# %% [markdown]
# Can see the encoder as ResNet, and at final layer, have added 6 channel output, with each class having its respective channel.  
# 
# %% [markdown]
# Summary of model

# %%
from torchsummary import summary
summary(model_unet_rn34, input_size=(3, 256, 256))

# %% [markdown]
# Checking the model pre-processing function

# %%
preprocess_input
#it's applying it's own normalisation
# can also use the same normalisation numbers in dataclass to avoid

# %% [markdown]
# Creating new dataloaders with models pre-processors. Since we are using pre-trained , we can discard earlier transformations.

# %%
# transform = transforms.Compose([
#             transforms.ToTensor()
#           #  transforms.Normalize([0.5], [0.5]) #since backbone is already doing normalisation
#         ])

# withuot augmentations



train_dataset_2 = slide_image(x_train_dir, y_train_dir, classes=CLASSES,preprocessing=get_preprocessing(preprocess_input))

train_loader_2 = DataLoader(train_dataset_2, batch_size=32, shuffle=True, num_workers=12)

val_dataset_2 = slide_image(x_valid_dir, y_valid_dir, classes=CLASSES,preprocessing=get_preprocessing(preprocess_input))

val_loader_2 = DataLoader(val_dataset_2, batch_size=32, shuffle=True, num_workers=12)

#!! FINALLY DATA LOADERS DONE


# %%
#! sanity check
image,mask=next(iter(train_dataset_2))
print(image.shape,mask.shape)
print(type(image),type(mask))
print(np.unique(image),np.unique(mask))

# %% [markdown]
# Loss:
# 
# Used dice loss for training.
# 
# Used metrics  = IoU , cant used accuracy because of high class imbalance.
# 
# Optimiser = Adam

# %%
loss = smp.utils.losses.DiceLoss()


# %%
loss = smp.utils.losses.DiceLoss()

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]


optimizer = torch.optim.Adam([ 
    dict(params=model_unet_rn34.parameters(), lr=0.0001),
])



# %%
train_epoch = smp.utils.train.TrainEpoch(
    model_unet_rn34, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model_unet_rn34, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)


# %%


# train model for 40 epochs, ended at 27th

max_score = 0

for i in range(0, 40):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader_2)
    valid_logs = valid_epoch.run(val_loader_2)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model_unet_rn34, './best_model_unet_rn34.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5')

# %% [markdown]
# ## Training summary
# 
# For the best model, on **validation set**:
# 
# dice_loss = 0.09517
# **dice_coeff = 0.9048**
# **IOU/Jaccard = 0.8693**
# 
# But from general observation.
# The model on 1st epoch was underfitting , as observed by IoU on train vs test.\
# 
# Then from 2nd epoch onwards model started overfitting, and once we reached training_iou ~ 0.95, all hopes were lost. Since only  = 1 - 0.95 →i.e val_iou increase of around 0.05 at best case. 
# 
# Hence model needed reguralisation techniques, like hard augmentations, higher dropouts.
# 
# PS: I missed to capture the running training loss and IoU, Hence no Plots, wanted to rerun but no time :( 
# %% [markdown]
# ### Best part in any modelling, prediction pipelines
# 
# 
# Took any random image and tried to observe the segmentation.
# 
# Had to take care of shapes, device etc
# 
# For the predicted tensor, since we get 6 segmentation maps, we have to do depthwise argmax, to get final segmentation map to compare with ground truth mask
# 

# %%
# Random testing


test_img_number = random.randint(0, len(val_dataset_2))

test_img,ground_truth = val_dataset_2[test_img_number]

image_input = torch.from_numpy(test_img).to(DEVICE).unsqueeze(0)

true_mask=ground_truth.round().transpose(1,2,0)
true_mask_argmax = np.argmax(true_mask,axis=2)


predicted_mask = model_unet_rn34.predict(image_input)
predicted_mask = (predicted_mask.squeeze().cpu().numpy().round().transpose(1,2,0))
predicted_mask_argmax = np.argmax(predicted_mask,axis=2)




plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img.transpose(1,2,0), cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(true_mask_argmax, cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_mask_argmax, cmap='jet')
plt.show()

# %% [markdown]
# ### Now I wanted to repatch our big image, and visualise
# 
# Hence first wrote a function that predicts over individual patches:

# %%
def single_patch_prediction(single_patch,single_mask,model):
    '''single patch data processing pipeline
    Args:
        
    '''
    # converting mask
#    single_patch = cv2.cvtColor(single_patch, cv2.COLOR_BGR2RGB)
    mask=single_mask
    #print(mask.shape)
    class_values = [CLASSES.index(cls) for cls in CLASSES]
    mask = encode_mask_pixel(mask_np=mask,pixel_array=np.unique(mask))
    # masks = [(mask == v) for v in class_values] 
    # mask = np.stack(masks, axis=-1).astype('float')
    print(mask.shape)
    
    mask_expand=np.expand_dims(mask,axis=2)
    print(mask_expand.shape)
    sample_patch = get_preprocessing(preprocess_input)(image= single_patch,mask = mask_expand )
    #print(np.expand_dims(mask,axis=2).shape)
    single_patch_tensor = torch.tensor(sample_patch['image'],dtype=torch.float32)
    #print(single_patch_tensor.shape)
    # single_patch_mask = sample_patch['mask']
    # print('sample patch mask shape',single_patch_mask.shape)

    predicted_patch=model.predict(single_patch_tensor.to(DEVICE).unsqueeze(0))

    predicted_patch = (predicted_patch.squeeze().cpu().numpy().round().transpose(1,2,0))
    predicted_patch_argmax = np.argmax(predicted_patch,axis=2)
    #print(predicted_patch_argmax.shape)

    return single_patch,single_mask,predicted_patch_argmax

# %% [markdown]
# Trying with single patch, without any dataloader, it works

# %%
# trying for single image pipeline, without dataloader

single_patch = cv2.imread('./Valid_Image_Patches/Val_B_22_25.png')
single_mask = cv2.imread('./Valid_Mask_Patches/Val_B_22_25.png',0)

single_patch = cv2.cvtColor(single_patch, cv2.COLOR_BGR2RGB)

single_patch,single_mask,predicted_patch_argmax = single_patch_prediction(single_patch,single_mask,model_unet_rn34)

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(single_patch, cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(mask, cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_patch_argmax, cmap='jet')
plt.show()

# %% [markdown]
# ## But not able to make it work with entire image :(
# 
# facing strange error, which I did not earlier, otherwise end -to -end pipeline was almost complete
# 
# Would apprecite your help in solving this
# 

# %%
#DIAGNOSE

large_image = cv2.imread('Valid_Image/2edf6cc7696c91d86eb86413fa9c82d7.png')
single_mask = cv2.imread('./Valid_Mask_Patches/Val_B_22_25.png',0) # dummy mask

image = cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB)

#This will split the image into small images of shape [3,3]
patches = patchify(image,(256,256,3),step=256)  #Step=256 for 256 patches means no overlap

predicted_patches = [] #create an empty list, where will fill patach predictions 1 by 1
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        print(i,j)
        
        single_patch = patches[i,j,:,:]       

        single_patch_prediction(single_patch,single_mask)
        

        predicted_patches.append(predicted_patch_argmax)

# Restich the image
predicted_patches = np.array(predicted_patches)

predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 128,128) )


reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)

plt.imshow(reconstructed_image, cmap='gray')

# %% [markdown]
# *More ToDO:  *
# 
# 1. Reguralisations: Augementations, more dropouts etc.  
# 2. GAP to remove size constraints    
# 4. Cropping as preprocessing step (remoe the white backround)  
# 5. Better patching algorithm (can see squares in output)  
# 6. Backround black different from class  "black", especially image 2 in training  
# 7. Class balance using class_weight   
# 8. Biomedical pretrained weights  
# 9. LR schedular  
# 
# 
# 
# 
# 
# 
# 
# 
# More models to try:
# 
# [https://github.com/PingoLH/FCHarDNet](https://github.com/PingoLH/FCHarDNet) 
# 
# [https://github.com/DengPingFan/PraNet](https://github.com/DengPingFan/PraNet) 
# 
# [https://github.com/jnkl314/DeepLabV3FineTuning](https://github.com/jnkl314/DeepLabV3FineTuning) 
# 
# [https://github.com/fregu856/deeplabv3](https://github.com/fregu856/deeplabv3)
# 
# [https://github.com/YudeWang/deeplabv3plus-pytorch](https://github.com/YudeWang/deeplabv3plus-pytorch) 
# 
# [https://github.com/YudeWang/semantic-segmentation-codebase/tree/main/experiment/deeplabv3%2Bvoc](https://github.com/YudeWang/semantic-segmentation-codebase/tree/main/experiment/deeplabv3%2Bvoc)
# 
# Transformer+Unet: https://github.com/Beckschen/TransUNet/blob/main/trainer.py
# 
# 
# %% [markdown]
# 

# %%



# %%



# %%



# %%



# %%



