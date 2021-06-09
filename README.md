### Semantinc segmentation with multiple classes

Images : High resolution slide pathology images with 3 channels  

Mask : Equal resolution to images, with 3 channels and 6 classes  

First appraoch was to use sliding window pre-processing operation.

Used patchify to create 254x254 non-overlapping patches.

Since one of first times dealing with such kind segmentation of dataset, 80% of time was spent on data pipelines.

Since we had to get the one segmentation map per class, we had to defined our ground truth the same.

Used extensive transformation operations as given in notebook attached.

Assumed that the backround class and “black” class was same for now.

Since less data, used a UNET with pre-trained ResNet32 encoder. And followed the transfer learning approach. Will also be trying more models in future.

For the best model, on validation set:

dice_loss = 0.09517
dice_coeff = 0.9048
IOU/Jaccard = 0.8693

