import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time
import warnings

def result_plot(results,names):
    fig = plt.figure()
    fig.suptitle('Performance Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

print('\n\n----------Breast Cancer Prediction---------\n\n')
# class 2->benign 4->malignant
# -------Step 1: Reading the data set
col_names=['Id','Cl.thickness','Cell.size','Cell.shape','Marg.adhesion','Epith.c.size','Bare.nuclei','Bl.cromatin','Normal.nucleoli','Mitoses','Class']
data = pd.read_csv('C:/Users/Pranave/Desktop/DM Project/data_569_32.csv', index_col=False,header=None,names=col_names)
data['Class']=data['Class'].apply(lambda x:'1' if x==4 else '0')

#--------Step 2: Preprocessing - Filling the unfilled data
for name in col_names:
    if not name is 'Id' :
        data[name] = data[name].apply(lambda x:5 if x=='?' else x)
        data[name] = data[name].apply(lambda x:5 if x=='' else x)
        data[name] = data[name].apply(lambda x:5 if x==' ' else x)
        
#data['Bare.nuclei'] = data['Bare.nuclei'].apply(lambda x:5 if x=='?' else x)
#print(data.groupby('Bare.nuclei').size())
grp = data.groupby('Class').size()

# -------Step 3: Splitting into training and test data set
# 20% of whole data is taken as test data
Y = data['Class'].values
X = data.drop('Class',axis=1).values
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.20, random_state=21)
#print(X_train,Y_train)

# -------Step 4: Taking different classifiers and finding the top 2 classifciation models which gives good accuracy and low error rate
#----------------Compare Decision tree, Gaussian Naive Bayes, K-Nearest Neighbours, Support Vector Machine
models_list = []
models_list.append(('CART', DecisionTreeClassifier()))
models_list.append(('SVM', SVC()))
models_list.append(('NB', GaussianNB()))
models_list.append(('KNN', KNeighborsClassifier()))
#print(models_list)

#--------Step 5: 10 fold cross validation - for finding which clssification model works well
num_folds = 10
results = []
names = []
for name, model in models_list:
    kfold = KFold(n_splits=num_folds, random_state=123)
    start = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')   
    end = time.time()
    results.append(cv_results)
    names.append(name)
    #print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))
#result_plot(results,names)

#---------Step 6: Standardize the dataset
#Centering and scaling happen independently on each feature 
#by computing the relevant statistics on the samples in the training set.
pipelines = []
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC( ))])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsClassifier())])))
results = []
names = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kfold = KFold(n_splits=num_folds, random_state=123)
    for name, model in pipelines:
        start = time.time()
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        #print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))
#result_plot(results,names)

#---------Found that SVM and Naive Bayes provides better results

#---------Step 7---SVM -- Finding good c and kernel values
# The larger C is the less the final training error will be. But if you increase C too much you risk
# losing the generalization properties of the classifier, because it will try to fit as best as possible all the training points 
warnings.simplefilter("ignore") # To suppress the warnings
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()
kfold = KFold(n_splits=num_folds, random_state=21)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
'''
for mean,+ stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
'''
#---------Step 8 - prepare the model with training data set
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    model = SVC(C=0.3, kernel='linear')
    start = time.time()
    model.fit(X_train_scaled, Y_train) 
    end = time.time()
    #print( "Run Time: %f" % (end-start))

#---------Step 9 - Test the model with test data set
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)

#---------Step 10 - Used Naive Bayes to reduce false negatives
gnb = GaussianNB()
y_pred1 = gnb.fit(X_train,Y_train)
y_pred2 = y_pred1.predict(X_test)
#print(y_pred2)

#---------Step 11 - Finding accuracy of the built model
y_final =y_pred2
#count = 0
for y in range(0,len(y_pred2)):
    if y_pred2[y] == '1' or predictions[y] == '1' :
        y_final[y] = '1'
        #count = count + 1
    else :
        y_final[y] = '0'

print("Accuracy score of the built model %f" % accuracy_score(Y_test, y_final))
print("Accuracy score of SVM %f" % accuracy_score(Y_test, predictions))
print("Accuracy score of Naive Bayes %f" % accuracy_score(Y_test, y_pred2))

#print(classification_report(Y_test, y_final))
#print(count)
#print(y_pred2)
#print(predictions)
#print(y_final)

#----------Step 12 - Calcutating the false -ve's
#------Calcutating the false -ve's for Final output
count_f = 0
for y in range(0,len(y_pred2)):
    if Y_test[y] == '1' and y_final[y] == '0' :
        #print(y_pred2[y],predictions[y])
        count_f = count_f + 1
        
#------Calcutating the false -ve's for SVM output        
count_s = 0
for y in range(0,len(y_pred2)):
    if Y_test[y] == '1' and predictions[y] == '0':
        count_s = count_s + 1

#------Calcutating the false -ve's for NB output        
count_n = 0
for y in range(0,len(y_pred2)):
    if Y_test[y] == '1' and y_pred2[y] == '0':
        count_n = count_n + 1
N = grp[1]
print("Percent of False Negatives in SVM, NB, F: ",(count_s/N)*100,(count_n/N)*100,(count_f/N)*100)