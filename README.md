### AGRICULTURAL PROJECT

Precision agriculture empowers farmers to make informed decisions about their farming strategies. This repository presents machine-learning techniques to build predictive models for 
 a)recommending the most suitable crops to grow on a particular farm based on various parameters,
 b)classify healthy and diseased crop leaves and, determine the kind of disease
 c)weather forecasting


 CROP RECOMMENDATION ANALYSIS

 Introduction

 This repository presents a dataset designed to build predictive models for recommending the most suitable crops to grow on a particular farm based on various parameters.

Data fields
N - ratio of Nitrogen content in soil
P - ratio of Phosphorous content in soil
K - ratio of Potassium content in soil
temperature - temperature in degree Celsius
humidity - relative humidity in %
ph - ph value of the soil
rainfall - rainfall in mm

- Data preprocessing involved cleaning the dataset to remove any inconsistencies or missing values, followed by feature selection/engineering to identify relevant features for the model. The data was then normalized/standardized to ensure all features were on a similar scale and one-hot encoding was applied to categorical variables.
  
- Five classification algorithms were employed: Support Vector Machine (SVM), Random Forest, Logistic Regression, Gradient Boosting, and Gaussian Naive Bayes (GaussianNB).
  
- For each algorithm, a pipeline was created, which included a StandardScaler for feature scaling and the respective classifier.
  
- Hyperparameters for each algorithm were optimized using GridSearchCV with a 5-fold cross-validation strategy to find the best combination of parameters.
  
- The best parameters and corresponding cross-validation scores were recorded for each algorithm.
  
- The Random Forest algorithm achieved the highest accuracy score of 0.99 on the test dataset, making it the best-performing model for this dataset.
  
- The classification report provided detailed metrics for each class in the dataset, including precision, recall, f1-score, and support, showcasing the Random Forest model's strong performance across all classes.
 

PLANT DISEASE CLASSIFICATION WITH RESNET9

It consists of  images of healthy and diseased crop leaves categorized into 38 different classes. The dataset is divided into an 80/20 ratio for training and validation.

The goal is clear and straightforward: build a model that can classify healthy and diseased crop leaves and, if the crop has a disease, predict which disease it is. By achieving this, we aim to provide a reliable tool for plant disease detection to support farmers in maintaining healthy crops and managing diseases effectively.

The data was loaded from the Plant Village dataset using TensorFlow Datasets (TFDS) for a deep learning project focused on image classification.

It was split into training and validation sets, with 80% allocated for training and 20% for validation purposes.

A ResNet9 architecture was implemented, featuring convolutional blocks with batch normalization and ReLU activation, along with residual connections to enhance learning.

The model was compiled using a sparse categorical crossentropy loss function and an SGD optimizer, with early stopping employed to prevent overfitting.

A learning rate scheduler was utilized to dynamically adjust the learning rate during training, optimizing model performance.

Following training, the model's performance was evaluated on the validation dataset to compute both loss and accuracy metrics.

To provide an overview of the model's performance, metrics such as loss and accuracy were visualized across epochs for both training and validation datasets.


WEATHER FORECASTING USING CNN-LSTM

This dataset includes temperature and weather information from major cities around the world. It serves as a valuable resource for analyzing the impact of global warming on agriculture and for other weather-related agricultural tasks.

- **Model Definition**: A Sequential model was defined using TensorFlow's Keras API. The model architecture included a Conv1D layer with 32 filters, a kernel size of 5, and "causal" padding, followed by ReLU activation. This was followed by two LSTM layers with 64 units each, both returning sequences. Then, two Dense layers with 30 and 10 units respectively, both with ReLU activation, were added. Finally, a Dense layer with 1 unit (output layer) and a Lambda layer scaling the output by 400 were included.

- **Data Preparation**: The user used the `windowed_dataset` function to prepare the training data (`train_set`) in batches of size 256, with a window size of 64 and shuffling.

- **Learning Rate Schedule**: A learning rate schedule was defined using a `LearningRateScheduler` callback, which adjusted the learning rate during training.

- **Model Compilation**: The model was compiled using the Huber loss function and an SGD optimizer with a learning rate of 1e-8 and momentum of 0.9.

- **Model Training**: Training the model was successful, completing all 100 epochs using the defined learning rate schedule.

- **Forecasting**: The model_forecast function was used to generate forecasts for the trained model.
