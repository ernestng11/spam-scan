# Spam Scan

## Feature Engineering:

1. POS tagging - Dropped pronoun counts since majority of its values are 0
2. Word counts - Follows normal distribution

## Data Preparation:

1. TFIDF vectorize texts
2. Concat with numerical variables
3. Did not standardize/normalize the data (Could explore)

## Model Testing: 

Evaluation Metrics: Negative mean logistic loss (Can try mae/auc) 

5-fold cross validation

1. Logistic Regression -0.04074115819049842
2. LinearSVC (Best performance) -0.021717561246588465
3. Random Forest -0.027817987134794823
4. AdaBoost -0.04056049786250815
5. XGBoost -0.030510824323127583

## Model Selection:

RandomForestClassifier has the second best score but I chose this over LinearSVC since ensemble method works better in practise and has more hyperparameter tuning option.

## Model Tuning:(Random Forest)

Evaluation Metrics for tuning: auc roc score TPR vs FPR

1. Tuned n_estimators (550) and max_depth (94)
2. Tuned max_features (auto) and max_samples (1)
3. Tuned class_weight, max_leaf_nodes, min_samples_split and min_samples_leaf

## Model Deployment:
