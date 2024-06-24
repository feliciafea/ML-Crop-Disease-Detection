# ML-Crop-Disease-Detection
Developed ML models to detect crop diseases using the PlantVillage dataset. Models include Decision Tree, Naive CNN, and ResNet CNN that achieved an accuracy rate of 99.44%

Video results: https://www.youtube.com/watch?v=iMsAdJLSbq8

# Introduction/Background

We all depend on crops as a major part of our global food supply, and as a result, the swift detection of crop diseases is essential to maximizing the output of our fields and ensuring the health of the ingredients used in everyday products. However, manual detection of plant diseases is tedious; during the process of identification, disease continues to propagate rapidly. It also requires a high level of expertise and can be highly error prone [3]. Many of these reasons have spurred research efforts to use machine learning and deep learning to train models to identify diseased plants as well as the specific type of disease in order to aid in quick prevention of the spread of harmful blights that can ruin fields of crops. For example, Kurlkarni et al. predicted several types of diseases each for images of five species of crops using a random forest classifier and was able to achieve an average accuracy of 93% [4]. Other papers have ventured beyond traditional machine learning methods; one group used CNNs to classify a dataset of diseased and healthy plant leaves to achieve a 99.3% accuracy [1].

We plan to use the PlantVillage dataset, which contains 54,303 images of healthy and diseased plant leaves from 13 species, including apples, corn, grapes, and tomatoes, with various diseases such as black rot, common rust, and bacterial spot. It can be found here, and the below figure shows some example images from the dataset.
<p align="center">
  <img src="https://github.com/feliciafea/ML-Crop-Disease-Detection/blob/main/assets/plantexamples.png" alt="Example Image">
</p>

# Problem Definition
The health of our crops is crucial for meeting our daily needs. Unfortunately, many people lack the knowledge to accurately identify diseases in their plants. To address this issue, we are working on developing a machine learning model that can distinguish between healthy and unhealthy leaf images across various plant species and diseases. Our model could have significant applications in the agriculture industry, as detecting diseased plants is essential for ensuring food security and protecting ecosystems from harm.

# Methods
## Preprocessing Mechanisms:
### Data Augmentation
When training Resnet and our naive CNN, we employed several forms of data augmentation to supplement our current dataset, including randomly rotating images (by -180 to 180 degrees), and randomly altering the coloring of each image (by reducing/increasing brightness, contrast, saturation, and hue by up to 20%). As images in the dataset were taken in controlled conditions (with the leaves in the same orientation, same lighting, etc.), applying these transformations to the training dataset helped our CNN model become more robust in being able to classify plant diseases outside of the realm of the dataset.

### Grayscale
In order to preprocess our images for training with the decision tree, we utilized a grayscale method to create a grayscale version of the PlantVillage dataset. Opting for grayscale as our preprocessing technique allowed us to remove color bias from the model and reduce computational complexity. Below is a figure displaying an example image from the dataset in color and grayscaled.
<p align="center">
  <img src="https://github.com/feliciafea/ML-Crop-Disease-Detection/blob/main/assets/grayscaleimage.png" alt="Example Image">
</p>

## Model #1 (Decision Tree)
For image classification, we employed a decision tree model. We selected a decision tree for training as it is known for its simplicity in computation compared to other models, yet remains highly effective in hierarchically classifying images.

## Model #2 (Naive CNN)
For our naive CNN model, our architecture looks like the following:
<p align="center">
  <img src="https://github.com/feliciafea/ML-Crop-Disease-Detection/blob/main/assets/cnnlayers.jpg" alt="Example Image">
</p>
We trained our naive CNN using an Adam optimizer, the cross entropy loss function, and trained the model over 25 epochs. We chose to train our own CNN model as CNNs with convolutional and pooling layers in general are highly effective at classifying images, and we wanted to compare this model to the results of applying transfer learning using pre-trained CNNs.

## Model #3 (ResNet CNN)
For our third model, we chose to employ transfer learning of ResNet. ResNet is a popular CNN model that is known for introducing residual connections which are used to handle the issue of the vanishing/exploding gradient. In our ResNet model, we chose to use the weights of ImageNet1k, which is a subset of the Image Net dataset that contains 1,000 image classes and over 1 million training images. We then fine tuned this model on Plant Village using the Adam optimizer and the cross entropy loss function, training it for 25 epochs. We chose to apply transfer learning on a highly popular CNN model such as ResNet on the Plant Village dataset, as transfer learning is a simple way of achieving high accuracy on image datasets without having to train a CNN model from scratch. Thus, we wanted to evaluate the accuracy of transfer learning on our specific dataset. Unsurprisingly, the ResNet model was highly accurate!

# Results and Discussion
## Quantitative Metrics
### Decision Tree Classifier
- Accuracy: 0.26271
- Precision: 0.26271
- Recall: 0.26217
- F1 Score: 0.25198
### Naive CNN
- Accuracy: 0.89420
- Precision: 0.89728
- Recall: 0.89420
- F1 Score: 0.89100
- Test Loss: 0.30617
### ResNet CNN
- Accuracy: 0.99448
- Precision: 0.99481
- Recall: 0.99448
- F1 Score: 0.99454
- Test Loss: 0.01806

## Visualizations:
Decision Tree
When running our decision tree model, we trained on different percentages of the Plant Village dataset due to the computational complexity and time involved in training. Below is a visualization of the model accuracy when trained with different percentages of the Plant Village dataset:
<p align="center">
  <img src="https://github.com/feliciafea/ML-Crop-Disease-Detection/blob/main/assets/decisiontreegraph.png" alt="Example Image">
</p>
The above graph displays how the accuracy of our decision tree model improved with larger percents of the total data used. Unfortunately, we were only able to use 75% of the Plant Village dataset due to RAM limitations. These limitations are not present in the next two CNN models.

Naive CNN
Below is a confusion matrix, loss graph, and accuracy graph for our naive CNN model.
<p align="center">
  <img src="https://github.com/feliciafea/ML-Crop-Disease-Detection/blob/main/assets/confusionmatrix-naive.png" alt="Example Image">
</p>
We can see that our CNN made very few wrong classifications and did not consistently classify one label as another label.
<p align="center">
  <img src="https://github.com/feliciafea/ML-Crop-Disease-Detection/blob/main/assets/confusionmatrix-naive.png" alt="Example Image">
</p>

From the graphs, we can see that training loss decreased until about 0.5 at epoch 15 or so, and accuracy increased until about 0.9 at epoch 15. Both loss and accuracy mostly plateaued after epoch 15. The validation accuracy was slightly above the training accuracy the entire time, and we believe this is because we applied the random transformations for data augmentation (like random rotation and color jittering) only on the training set and not on the validation set. Since validation accuracy was not significantly lower than training accuracy for the entirety of training, this means our model did not overfit.

