# Blood_cells
## Description
Creating a model to classify images of blood cells into 4 classes, using the Kaggle dataset [Blood Cell Images](https://www.kaggle.com/paultimothymooney/blood-cells).

## Approach
I used the ResNet50 model with with my own classification layer. I use the pretrained weights for the rest of the network, which acts as a feature extractor and only train the classifier layer on my dataset.

## Results
It achieves around 60% over the test set in 5 epochs, which could improved since I haven't done much hyperparameter tuning.

![Example](Example.png)
