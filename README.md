# hierarchical-classification

## Dataset

https://drive.google.com/open?id=0B7EVK8r0v71pa2EyNEJ0dE9zbU0

This dataset contains apparel classified at two levels. the first level has three classes 
i.e Upper-body, Lower-body and Full-body.
These three classes contain 50 different catogory of classes.

### list_category_cloth.txt
    - The first column has name of 50 different class and the second column contains the first classification layer.
    
### list_bbox.txt
    - The first column contains image path and the next 4 columns have bounding box coordinates x1,y1,x2,y2.
    - x1 & y1 : left upper coordinate
    - x2 & y2 : right lower coordinate
    
## script.py

This makes the hierarchical directory for the dataset.

## crop_image.py

This crops the images using bounding box coordinates and then we saves the images in respective folders.

## count.py

The apparel dataset has 50 classes at the lower level but there is a large imbalance in dataset.
Thus we would only select those classes which has more than 1000 images.
Then, we are making directory from train, validation and test.

## create.py

We are selecting 100 images from each class for testing and validation.
And around 1500 images from each class for training.

## train_val.py

https://arxiv.org/abs/1709.09890
The code is implementation of branched-CNN concept which has been mentioned in the above paper.
We run the model for 60 epochs.
The initail learning rate was 0.003. After 15 epochs it was changed to 0.0005 and after 30 epochs it was changed to 0.0001.
The initial value of alpha=0.7 and beta=0.3. Then alpha and beta was changed to 0.3 and 0.7 after 12 epochs respectively.
At epoch=42, alpha=0.0 and beta=1.0

## test.py

For testing the model. The accuracy for 1st layer is 93.76% and for 2nd layer is 69.38%.
