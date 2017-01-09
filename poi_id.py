# %load poi_id.py
#How : http://stackoverflow.com/questions/21034373/how-to-load-edit-run-save-text-files-py-into-an-ipython-notebook-cell
#!/usr/bin/python
import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from time import time
import re
from numpy import log
from numpy import sqrt
from numpy import float64
from numpy import nan
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
sys.path.append("../final_project/")
from tester import dump_classifier_and_data, test_classifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi", "salary", "bonus", 'total_payments', 'loan_advances', 'total_stock_value',\
                 'expenses', 'exercised_stock_options',\
                 'long_term_incentive', 'restricted_stock', 'director_fees'] 

features_list_new = ["poi", "salary", "bonus", 'total_payments', \
'loan_advances', 'total_stock_value', 'expenses','exercised_stock_options',\
'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock',\
'director_fees','employee_perks', 'perks_ratio']


# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop("LOCKHART EUGENE E", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
### 2.1 Correctwrong entries found after plotting deferral payments:
data_dict["BHATNAGAR SANJAY"]['restricted_stock_deferred'] = "-2604490"
data_dict["BHATNAGAR SANJAY"]['restricted_stock'] = "2604490"
data_dict["BHATNAGAR SANJAY"]['exercised_stock_options'] = "15456290"
data_dict["BHATNAGAR SANJAY"]['total_stock_value'] = "15456290"
data_dict["BHATNAGAR SANJAY"]['expenses'] = "137864"


data_dict["BELFER ROBERT"]['deferred_income'] = "-102500"
data_dict["BELFER ROBERT"]['deferral_payments'] = "NaN"
data_dict["BELFER ROBERT"]['expenses'] = "3285"
data_dict["BELFER ROBERT"]['director_fees'] = "102500"
data_dict["BELFER ROBERT"]['total_paymens'] = "3285"
data_dict["BELFER ROBERT"]['exercised_stock_options'] = "NaN"
data_dict["BELFER ROBERT"]['restricted_stock'] = "44093"
data_dict["BELFER ROBERT"]['restricted_stock_deferred'] = "-44093"
data_dict["BELFER ROBERT"]['total_stock_value'] = "NaN"


### Task 3: Create new feature(s)

# From https://github.com/DariaAlekseeva/Enron_Dataset/blob/master/Enron%20POI%20Detector%20Project%20Assignment.ipynb

#Get rid of NAN


for feature in features_list[1:]:
    for key in data_dict.keys():
        if data_dict[key][feature] == 'NaN':
            data_dict[key][feature] = 0.


for name in data_dict:

# Add ratio of POI messages to total and employee perks (https://github.com/sebasibarguen/udacity-nanodegree-machinelearning/blob/master/final_project/poi_id.py)
    try:
        total_messages = data_dict[name]['from_messages'] + data_dict[name]['to_messages']
        poi_related_messages = data_dict[name]["from_poi_to_this_person"] +\
                                data_dict[name]["from_this_person_to_poi"] +\
                                data_dict[name]["shared_receipt_with_poi"]
        poi_ratio = 1.* poi_related_messages / total_messages
        data_dict[name]['poi_ratio_messages'] = poi_ratio
    except:
        data_dict[name]['poi_ratio_messages'] = 'NaN'
    try:
        employee_perks = float(data_dict[name]["salary"])+float(data_dict[name]["bonus"]) \
+float(data_dict[name]["long_term_incentive"])\
+float(data_dict[name]["total_stock_value"])
        data_dict[name]["employee_perks"] = employee_perks
    except:
        data_dict[name]['employee_perks'] = 'NaN'
    try:
        total_earnings = float(data_dict[name]["total_payments"])+float(data_dict[name]["total_stock_value"])
        data_dict[name]["perks_ratio"] = employee_perks/total_earnings
    except:
        data_dict[name]["perks_ratio"] = "NaN"


###Update the features list (https://github.com/austinjalexander/udacity_intro_ml_project/blob/master/poi_id.py)
for key in data_dict.keys():
    features_list.append("poi_ratio_messages")
    features_list.append("employee_perks")
    features_list.append("perks_ratio")


### Store to my_dataset for easy export below.
my_dataset = data_dict

###Choose the best features Kbest

### Extract features and labels from dataset for local testing
### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list_new, sort_keys = True)
labels, features = targetFeatureSplit(data)
cv = StratifiedShuffleSplit(labels, n_iter=1000, random_state = 42)
for i_train, i_test in cv:
    features_train, features_test = [features[i] for i in i_train], [features[i] for i in i_test]
    labels_train, labels_test = [labels[i] for i in i_train], [labels[i] for i in i_test]
    for ii in i_train:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in i_test:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )
        
     
# We will test for the best features and then create a new features list
# 
from sklearn.svm import SVC

# 1000 test and train splits
scaler = MinMaxScaler() 
skb = SelectKBest(f_classif)
pca = PCA()
#clf_GNB = GaussianNB()
svm = SVC()

#feature selection
#pca_params = {"PCA__n_components":[3,4,5]}
#kbest_params = {"SKB__k":[5,6,7]}
#SVC_params = {'svc__kernel':('linear', 'rbf','poly'),
#              'svc__C':(1,5,10),
#              'svc__decision_function_shape':('ovo','over','None'),
#              'svc__tol':(1e-3,1e-4,1e-5)
#}
#pca_params.update(kbest_params)
#pca_params.update(SVC_params)
#pipe= Pipeline([('scaler',scaler),("SKB", skb),("PCA", pca),("svc", svm)])
#clf = GridSearchCV(pipe,pca_params, scoring='f1')
#new_clf = clf.fit(features_train, labels_train)
#new_pred = clf.predict(features_test)
#Result: ['salary', 'bonus', 'total_stock_value', 'exercised_stock_options', 'restricted_stock', 'employee_perks']
# you'll want to see the model selected:
#clf_top = new_clf.best_estimator_
#print clf_top
#features_selected_bool = new_clf.best_estimator_.named_steps['SKB'].get_support()
#features_selected_list = [x for x, y in zip(features_list_new[1:], features_selected_bool) if y]

#print features_selected_list

#Create new list with features and split it just as we did with features_list_new
features_list_best = ["poi","salary", "bonus", 'exercised_stock_options',
                     'restricted_stock','employee_perks', 'total_stock_value']
data1 = featureFormat(data_dict, features_list_best, sort_keys = True)
labels1, features1 = targetFeatureSplit(data1)
cv1 = StratifiedShuffleSplit(labels, n_iter=1000, random_state = 42)
for i_train, i_test in cv1:
    features_train1, features_test1 = [features[i] for i in i_train], [features[i] for i in i_test]
    labels_train1, labels_test1 = [labels[i] for i in i_train], [labels[i] for i in i_test]
    for ii in i_train:
        features_train1.append( features[ii] )
        labels_train1.append( labels[ii] )
    for jj in i_test:
        features_test1.append( features[jj] )
        labels_test1.append( labels[jj] )
### use KFold for split and validate algorithm

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
#Feature selection:
   
# Running a basic classifier first in order to get a feel for the process and tuning

NB = GaussianNB()
NB.fit(features_train1, labels_train1)
predNB = NB.predict(features_test1)

test_classifier(NB, my_dataset, features_list_best)



### Alternatives 2 and 3 BEST RESULTS DON'T CHANGE


#classifier number 2 - Logistic regression tuning
#LogisticR = LogisticRegression()
#parameters = {'LogisticRegression__C':(40,50,60),
#              'LogisticRegression__class_weight':('balanced',None),
#              'LogisticRegression__intercept_scaling':(1,2,3),
#              "LogisticRegression__max_iter": (100, 200,300),
 #             "LogisticRegression__tol": (1e-05,1e-06,1e-07)
 #             }
#pca_params = {"PCA__n_components":[1,2,3,4,5]}

#parameters.update(pca_params)
#pipeline2 = Pipeline([('scaler',scaler),('PCA', pca),('LogisticRegression', LogisticR )])
#clf1 = GridSearchCV(pipeline2, parameters, scoring='f1')
#clf1.fit(features_train1, labels_train1)
#pred1 = clf1.predict(features_test1)
#scaler - necessary step before PCA 
#PCA - reduces dimensionality based on 
#skb - picks best feture based on anova
#LogisticR picks a feature based on overall fit to the model


#you'll want to see the model selected:
#clf_top1 = clf1.best_estimator_
#print clf_top1
#Pipeline(steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('PCA', PCA(copy=True, n_components=5, whiten=False)), ('LogisticRegression', LogisticRegression(C=40, class_weight='balanced', dual=False,
#          fit_intercept=True, intercept_scaling=1, max_iter=100,
#          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
#          solver='liblinear', tol=1e-05, verbose=0, warm_start=False))])
    

### Alternatives 2 and 3 BEST RESULTS DON'T CHANGE
#combined_features = FeatureUnion([("pca", PCA(n_components=5)), ("univ_select", SelectKBest(k=5))])
#LogisticR = LogisticRegression(C=40, class_weight='balanced', dual=False,
#          fit_intercept=True, intercept_scaling=1, max_iter=100,
#          multi_class='ovr', random_state=None,
#         solver='liblinear', tol=1e-05, verbose=0 )

#clf_log = Pipeline([("minmax", scaler), ("features", combined_features), ("clf", LogisticR)])

#test_classifier(clf_log, my_dataset, features_list)
#Results on the first list with featuresAccuracy: 0.74433	Precision: 0.30179	Recall: 0.69850	F1: 0.42148	F2: 0.55309
#	Total predictions: 15000	True positives: 1397	False positives: 3232	False negatives:  603	True negatives: 9768
pca = PCA(n_components=1)
LogisticR = LogisticRegression(C=40, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=5, max_iter=100,
          multi_class='ovr', n_jobs=1, random_state=None, penalty='l2',
          solver='liblinear', tol=1e-05, verbose=0, warm_start=False )

clf_log = Pipeline([("minmax", scaler), ("PCA", pca), ("clf", LogisticR)])
clf_log.fit(features_test1, labels_test1)

test_classifier(clf_log, my_dataset, features_list_best)
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#Result according to the model selected by GridSearch:
#Accuracy: 0.72846	Precision: 0.26577	Recall: 0.43400	F1: 0.32966	F2: 0.38523
#	Total predictions: 13000	True positives:  868	False positives: 2398	False negatives: 1132	True negatives: 8602

### Alternatives 2 and 3 BEST RESULTS DON'T CHANGE
#Next is the DecisionTree
tree = DecisionTreeClassifier()
pca = PCA(n_components=5)
# classifier number 2 - Decision Tree tuning
#parameters = {'DecisionTreeClassifier__min_samples_split':(10,20),
#              'DecisionTreeClassifier__min_samples_leaf':(2,4),
#              'DecisionTreeClassifier__max_leaf_nodes':(20,30,40,50),
#              "DecisionTreeClassifier__min_weight_fraction_leaf": (0.3,0.4,0.5),
#              "DecisionTreeClassifier__random_state": (20,40),
#              "DecisionTreeClassifier__criterion": ('gini','entropy')
#              }

#pipeline1 = Pipeline([('DecisionTreeClassifier', tree )])
#scaler - necessary step before PCA 
#PCA - reduces dimensionality based on 
#skb - picks best feture based on anova
#decision tree picks a feature based on overall fit to the model
#
#clf1 = GridSearchCV(pipeline1, parameters, scoring='f1')
#clf1.fit(features_train1, labels_train1)
#pred1 = clf1.predict(features_test1)

# you'll want to see the model selected:
#clf_top1 = clf1.best_estimator_

#print clf_top1
#GridSearch model result:
#Results: Pipeline(steps=[('DecisionTreeClassifier', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
#            max_features=None, max_leaf_nodes=20, min_samples_leaf=2,
#            min_samples_split=10, min_weight_fraction_leaf=0.3,
#            presort=False, random_state=20, splitter='best'))])



clf_tree = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=20, min_samples_leaf=2,
            min_samples_split=10, min_weight_fraction_leaf=0.3,
            presort=False, random_state=20, splitter='best')
clf = Pipeline([('DecisionTreeClassifier', clf_tree )])
clf.fit(features_train1, labels_train1)
test_classifier(clf, my_dataset, features_list_best)
#Pipeline(steps=[('DecisionTreeClassifier', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
#            max_features=None, max_leaf_nodes=20, min_samples_leaf=4,
#            min_samples_split=10, min_weight_fraction_leaf=0.3,
#            presort=False, random_state=20, splitter='best'))])
#	Accuracy: 0.82043	Precision: 0.35879	Recall: 0.32650	F1: 0.34188	F2: 0.33248
#	Total predictions: 14000	True positives:  653	False positives: 1167	False negatives: 1347	True negatives: 10833
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list_best)