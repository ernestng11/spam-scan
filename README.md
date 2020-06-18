# Spam Scan

## Feature Engineering:

1. POS tagging - Dropped pronoun counts since majority of its values are 0
2. Word counts - Follows normal distribution

## Data Preparation:

1. TFIDF vectorize texts
2. Concat with numerical variables
3. Did not standardize/normalize the data (Could explore)

## Model Testing with scikit-learn: 

Evaluation Metrics: Negative mean logistic loss and ROC Score

5-fold cross validation

*neg_mean_log_loss*

1. Logistic Regression -0.04074115819049842
2. LinearSVC -0.021717561246588465 (Best performance)
3. Random Forest -0.027817987134794823
4. AdaBoost -0.04056049786250815
5. XGBoost -0.030510824323127583

*roc_auc_score* (to be tested)

1. Logistic Regression 
2. LinearSVC 
3. Random Forest 
4. AdaBoost 
5. XGBoost 

## Model Testing with Neural Networks: (to be done)

## Model Selection:

RandomForestClassifier has the second best score but I chose this over LinearSVC since ensemble method works better in practise and has more hyperparameter tuning option.

## Model Tuning:(Random Forest)

Evaluation Metrics for tuning: auc roc score TPR vs FPR

**Initial model, clf1 ** ROC Score: 0.9087301587301587

1. Tuned n_estimators (500) and max_depth (94)
2. Tuned max_features (auto) 

**Final model, clf4 ** ROC Score: 0.9146825396825398



## Model Deployment:
