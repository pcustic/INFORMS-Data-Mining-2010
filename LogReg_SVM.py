"""
    Izradili Ante Mijoc i Porin Custic u sklopu projekta za kolegij Strojno ucenje.
    Skripta radi lagged varijable na TargetVariable sa koracima 1-12 i pokrece logisticku
    regresiju i SVM na svim podacima.
"""

import pandas as pd
from sklearn import metrics, linear_model, svm

training = pd.read_csv("TrainingData.csv")
test = pd.read_csv('TestData.csv')

for i in range(1,13):
    training['TargetVariable_lag' + str(i)] = training['TargetVariable'].shift(-i)
    test['TargetVariable_lag' + str(i)] = test['TargetVariable'].shift(-i)

training.fillna(0, inplace=True)
test.fillna(0, inplace=True)

training_target = training['TargetVariable']
del training['TargetVariable']
del training['Timestamp']

test_target = test['TargetVariable']
del test['TargetVariable']
del test['Timestamp']

logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(training, training_target)
predicted = logistic.predict(test)
print metrics.roc_auc_score(test_target, predicted)

