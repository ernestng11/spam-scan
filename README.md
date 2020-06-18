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

1. Logistic Regression
2. Linear SVC (Best performance)
3. Random Forest
4. AdaBoost
5. XGBoost 

## Model Selection:

## Model Tuning:(Random Forest)

Evaluation Metrics for tuning: auc roc score TPR vs FPR

1. Tuned n_estimators (550) and max_depth (94)
2. 

## Model Deployment:
