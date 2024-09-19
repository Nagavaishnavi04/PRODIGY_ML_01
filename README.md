# PRODIGY_ML_01

 **House Prices Prediction Using Linear Regression**

 
This project aims to predict house prices using a linear regression model. The model is built using data from the Kaggle House Prices: Advanced Regression Techniques competition. By training the model on historical data, we predict house prices based on features like square footage, number of bedrooms, and number of bathrooms.

**Project Overview**

In this project, we:

1.Loaded and preprocessed the dataset from Kaggle (```train.csv``` and ```test.csv```).

2.Selected relevant features such as square footage (```GrLivArea```), number of bedrooms (```BedroomAbvGr```), and number of bathrooms (```FullBath```).

3.Trained a linear regression model to predict house prices.

4.Made predictions on the test data and formatted the results for submission as per Kaggle's requirements.


**Files in this Repository**

- **```train.csv```:** The training data provided by Kaggle, which contains features and house prices.

- **```test.csv```:** The test data for which the house prices need to be predicted.

- **```sample_submission.csv```:** The format provided by Kaggle for submitting predictions.

- **```house_price_prediction.py```:** The Python script that loads data, trains the model, and generates the predictions.

- **```my_submission.csv```:** The output file containing predictions for the test set, formatted for Kaggle submission.


**Requirements**

To run the code, you'll need the following Python libraries:

- ```pandas```
- ```numpy```
- ```scikit-learn```
- ```matplotlib```
- ```seaborn```

You can install the required libraries using:

```pip install pandas numpy scikit-learn matplotlib seaborn```

  
**How to Run:**

1.Clone this repository.

2.Place the ```train.csv```, ```test.csv```, and ```sample_submission.csv``` files in the same directory as the Python script.

3.Run the ```house_price_prediction.py``` script to train the model and generate predictions.

4.A file named ```my_submission.csv``` will be generated, ready for submission to Kaggle.

**What You Will Learn from This Project**

This project is a great learning experience for anyone looking to get started with machine learning and linear regression. Key concepts you’ll learn include:

- Data Preprocessing: How to clean and prepare datasets for modeling.
  
- Feature Selection: Identifying the most relevant features for prediction.
  
- Linear Regression: Implementing a simple machine learning algorithm to model relationships between features and target variables.
  
- Model Evaluation: Using metrics like Mean Squared Error (MSE) and R-squared score to evaluate model performance.
  
- Kaggle Submissions: Preparing and formatting predictions for submission to Kaggle competitions.


Dataset link:https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
