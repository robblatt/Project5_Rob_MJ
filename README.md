# Applying Classification Modeling Project

## The Goal

Fit three models (KNN, Logistic Regression, and Decision Tree Classifier) to predict credit card defaults and use gridsearch to find the best hyperparameters for those models to best predict credit card defaults from an edited version of the [Default Payments of Credit Card Clients in Taiwan from 2005](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset) dataset. Partnered with [zaborowm on GitHub](https://github.com/zaborowm/).

The project's final output was a .csv file to be tested against the entire class. 

## The Data
* 23,999 entries, features:
  * 'LIMIT_BAL'
  * 'SEX'
  * 'EDUCATION'
  * 'MARRIAGE'
  * 'AGE'
  * 'PAY_0'
  * 'PAY_2'
  * 'PAY_3'
  * 'PAY_4'
  * 'PAY_5'
  * 'PAY_6'
  * 'BILL_AMT1'
  * 'BILL_AMT2'
  * 'BILL_AMT3'
  * 'BILL_AMT4'
  * 'BILL_AMT5'
  * 'BILL_AMT6'
  * 'PAY_AMT1'
  * 'PAY_AMT2'
  * 'PAY_AMT3'
  * 'PAY_AMT4'
  * 'PAY_AMT5'
  * 'PAY_AMT6'
  * 'default payment next month'
  
## features.py

The feature engineering function creates new features, creating features that make sense of the features:
* Age is bucketed into four features
* Education is dummied
* The pay and bill columns are renamed to have a better understanding of how they align and a balance 
* Features added to mark people who carried a balance over 80% of their credit limit on a monthly basis
* Drops irrelevant columns

### Comparing bills to bills paid

![comparing bills to bills paid](https://raw.githubusercontent.com/robblatt/Project5_Rob_MJ/master/comparing%20bills%20to%20bills%20paid.png)

## Results

Logistic Regression had an f1 score of 81.8 %
KNN had an f1 score of 74% after hyptermarameter tuning
Decision Tree had an f1 score of 73.9% after hypterparameter tuning

### Files

#### Mod4_Rob_MJ.ipnyb
Contains the work to produce the output file.

####  Classification Assessment Mini-Project.ipynb
Contains EDA and the assignment

#### features.py
Explained above
