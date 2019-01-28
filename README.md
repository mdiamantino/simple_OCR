# Partie 4 - Projet de fin d'année ULB B1-INFO

Additional features to the previous part.

## Getting Started

These instructions will help you run the project on your local machine for development and testing purposes.

### Prerequisites

The packages that are used and their version. You will eventually need to install in order to run this project if you haven't done it yet.

```
PyQt ( tested with 5.6.0 )
Numpy ( tested with 1.14.0 )
PIL ( Pillow ) ( tested with 5.0 )
Matplotlib ( tested with 2.1.2 )
OpenCv ( tested with 3.3.1 )
time
sys
os
```

The used dataset is the MNIST handwritten digits dataset, therefore ensure to have all the following ( they are **not included** ):
```
train.csv
train_small.csv
train_tiny.csv
```

### Installing


You'll find in the folder three python files:

```
main.py			=> You should run this file 
basicUI.py		=> GUI file
MPLClass.py		=> Mpl functions
```

The W1, W2 weights that you can use to speed-up predictions and verify scores. Please **do not remove** from the main folder.

```
W1.csv
W2.csv
```

A folder called 'OCRtest' in which you will find two scanned images of different handwritten digits in order to test the OCR feature. More tests will come with the report.

## Running

Make sure to have everything as expected and **run 'main.py'**.

### What you can do

There are three main functions as well as tabs

```
- Standard
- OCR
- Tools
```

#### Test 'Standard':
The Standard tab lets you run a Cross-Validation with or without pre-saved Weights, see logs, save them, save weights, see the confusion matrix.
Try to load given weights ( just click on the 'load weights' button and make sure to check the 'Use them?' checkbox, then press start ).</br>


#### Test 'OCR':
The OCR tab lets you load a picture and predict which digit there are inside.
Load the given weights, load one of the give images and type the number of digits there are written in, then click on 'Start OCR'.
```
- Image: TEST0721184.jpg
- Weights: W1.csv, W2.csv
- Digits: 7
- OCR mode: Auto

REAL: 0721184
PREDICTED: 0321184
SCORE: 85.7%
```

#### Test 'Tools':
The Tools tab lets you easily convert an array to image (and display it) or upload an image in order to convert it to array (MNIST format 1x784).
```
- Try to convert with no written arrays
- Try to convert an array with less then 784 element or without backets
- Try to display with an array from 'train.csv'
```


## Coded With

* PyCharm 2017.3.4


## Author

* **Mark Diamantino Caribé** - *B1-INFO ULB* 


