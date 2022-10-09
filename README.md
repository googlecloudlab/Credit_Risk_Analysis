# Credit_Risk_Analysis

## Overview of the analysis 
Explain the purpose of this analysis.
In this analysis, I apply my skills in data preparation, statistical reasoning, and machine learning to solve the real-world challenge of credit card risk.

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, I employ different techniques to train and evaluate models with unbalanced classes. I use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, I use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I compare two machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. I evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Results
I this section, I use bulleted lists to describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. I use screenshots of the outputs to support the results.

**1. Naive Random Oversampling**

Balanced Accuracy Score: 0.636

Pricision Score (high risk): 0.01

Recall Score (high risk): 0.68

F1 Score (high risk): 0.02

```
                   pre       rec       spe        f1       geo       iba      sup   

  high_risk       0.01      0.68      0.59      0.02      0.63      0.41       101
   low_risk       1.00      0.59      0.68      0.74      0.63      0.40     17104

avg / total       0.99      0.59      0.68      0.74      0.63      0.40     17205
```

**2. SMOTE Oversampling**

Balanced Accuracy Score: 0.662

Pricision Score (high risk): 0.01

Recall Score (high risk): 0.63

F1 Score (high risk): 0.02

```
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.63      0.69      0.02      0.66      0.44       101
   low_risk       1.00      0.69      0.63      0.82      0.66      0.44     17104

avg / total       0.99      0.69      0.63      0.81      0.66      0.44     17205
```

**3. Undersampling**

Balanced Accuracy Score: 0.544

Pricision Score (high risk): 0.01

Recall Score (high risk): 0.69

F1 Score (high risk): 0.01

```
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.69      0.40      0.01      0.52      0.28       101
   low_risk       1.00      0.40      0.69      0.57      0.52      0.27     17104

avg / total       0.99      0.40      0.69      0.56      0.52      0.27     17205
```

**4. Combination (Over and Under) Sampling**

Balanced Accuracy Score: 0.645

Pricision Score (high risk): 0.01

Recall Score (high risk): 0.72

F1 Score (high risk): 0.02

```
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.72      0.57      0.02      0.64      0.42       101
   low_risk       1.00      0.57      0.72      0.72      0.64      0.40     17104

avg / total       0.99      0.57      0.72      0.72      0.64      0.40     17205
```

**5. Ensemble with Balanced Random Forest Clasifier**

Balanced Accuracy Score: 0.789

Pricision Score (high risk): 0.03

Recall Score (high risk): 0.70

F1 Score (high risk): 0.06

```
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.03      0.70      0.87      0.06      0.78      0.60       101
   low_risk       1.00      0.87      0.70      0.93      0.78      0.62     17104

avg / total       0.99      0.87      0.70      0.93      0.78      0.62     17205
```

Top 5 features by importance
```
[(0.07876809003486353, 'total_rec_prncp'),
 (0.05883806887524815, 'total_pymnt'),
 (0.05625613759225244, 'total_pymnt_inv'),
 (0.05355513093134745, 'total_rec_int'),
 (0.0500331813446525, 'last_pymnt_amnt')
 ...
]
```

**6. Ensemble with Easy Ensemble AdaBoost Classifier**

Balanced Accuracy Score: 0.932

Pricision Score (high risk): 0.09

Recall Score (high risk): 0.92

F1 Score (high risk): 0.16

```
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.09      0.92      0.94      0.16      0.93      0.87       101
   low_risk       1.00      0.94      0.92      0.97      0.93      0.87     17104

avg / total       0.99      0.94      0.92      0.97      0.93      0.87     17205
```


## Summary
In this section I summarize the results of the machine learning models and include a recommendation on the model to use.  

The following table summarizes the results from the various sampling techniques that were applied.

Technique            | Balanced Accuracy | Precision | Recall | F1 |
---------------------|-------------------|-----------|--------|----|
Random Oversampling  | 0.636             |0.01       |0.68    |0.02|
SMOTE Oversampling   | 0.662             |0.01       |0.63    |0.02|
Undersampling        | 0.544             |0.01       |0.69    |0.01|
Combination          | 0.645             |0.01       |0.72    |0.02|
Ensemble (BRF)       | 0.789             |0.03       |0.70    |0.06|
Ensemble (AdaBoost)  | 0.932             |0.09       |0.92    |0.16|


**Recommendation**

Let's go over the results in the Summary table above for the various sampling techniques and the results from the classification reports:

- **Accuracy**: All models have an accuracy rate that is greater than 50%. The Ensemble with Easy Ensemble AdaBoost Classifier method yielded the best accuracy score of 93.2%.  

- **Precision**: Precision is the measure of how reliable a positive classification is. From our results, the precision for the high risk loans can be determined by the ratio TP/(TP + FP). A low precision is indicative of a large number of false positives—of the loans we predicted to be high risk loans that turn out to be low risk loans. 
All of our analysis techniques resulted in low Precision scores. This means that there is a high rate of false-positives.  The Ensemble with Easy Ensemble AdaBoost Classifier method yielded the best precision score of 0.09.  

- **Recall**: Recall is the ability of the classifier to find all the positive samples. It can be determined by the ratio: TP/(TP + FN). A low recall is indicative of a large number of false negatives.
Recall was generally high in our analysis for all models with values over 0.60 in all models.  The Ensemble with Easy Ensemble AdaBoost Classifier method yielded the best recall score of 0.92.   

- **F1 score**: F1 score is a weighted average of the true positive rate (recall) and precision, where the best score is 1.0 and the worst is 0.0.
All of our analysis techniques resulted in low F1 scores due to the low Precision scores.  The Ensemble with Easy Ensemble AdaBoost Classifier method yielded the best F1 score of 0.16.  

- **Support**: Support is the number of actual occurrences of the class in the specified dataset. For our results, there are 101 actual occurrences for the high risk loans and 17104 actual occurrences for low-risk loans.

In summary, this model may not be the best one for preventing high-risk loans because the low precision score in all models. The Ensemble with Easy Ensemble AdaBoost Classifier method would be the best technique to apply due to it's high accuracy and high sensitivity.  Modeling is an iterative process: you may need more data, more cleaning, another model parameter, or a different model. It's also important to have a goal that's been agreed upon, so that you know when the model is good enough.


