# -*- coding: utf-8 -*-
# Author: Chen Ling

import data_helper as dh
import numpy as np

# Sklearn Section
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


X, y = [], []
user_X = []; user_list = []

for i in range(74):
    temp, temp_user, temp_tweet = dh.get_training_vector('dataset_1/{}/'.format(i))
    
    for i in range(len(temp)):
        X.append([len(j) for j in temp[i]])
        if i == 1:
            y.append(1)
        else:
            y.append(0)
    
    for i in range(len(temp_user)):
        tt = []
        for j in range(len(temp_user[i])):
            aa = list(set(temp_user[i][j]))
            tt.append(len(list(set(aa)-set(user_list))))
            user_list.extend(list(set(temp_user[i][j])))
            user_list = list(set(user_list))
        user_X.append(tt)
            

tweet_X = []

for i in X:
    temppp = []
    for j in range(len(i)):
        if j == 0:
            temppp.append(i[0])
        else:
            temppp.append(i[j] - i[j-1])
    tweet_X.append(temppp)    


X = Normalizer().fit_transform(X) # fit does nothing.
tweet_X = Normalizer().fit_transform(tweet_X) # fit does nothing.
user_X = Normalizer().fit_transform(user_X) # fit does nothing.



training_list = []
for i in range(len(X)):
    temp = np.stack((X[i], user_X[i], tweet_X[i]))
    training_list.append(temp)
   
training_list = np.array(training_list)  
nsamples, nx, ny = training_list.shape
training_set = training_list.reshape((nsamples,nx*ny))  

'''Generate evenly distributed test set'''
disturbed_list, undisturbed_list = [], []
for i, x in enumerate(y):
    if x == 1:
        disturbed_list.append(i)
    else:
        undisturbed_list.append(i)

aaa = np.random.choice(disturbed_list, 37, False)
bbb = np.random.choice(undisturbed_list, 37, False)

aaaa = ([v for i,v in enumerate(disturbed_list) if v not in aaa])
bbbb = ([v for i,v in enumerate(undisturbed_list) if v not in bbb])


x_train = [training_set[i] for i in aaaa] + [training_set[i] for i in bbbb]
y_train = [y[i] for i in aaaa] + [y[i] for i in bbbb]

x_test = [training_set[i] for i in aaa] + [training_set[i] for i in bbb]
y_test = [y[i] for i in aaa] + [y[i] for i in bbb]

'''
x_train, x_test, y_train, y_test = train_test_split(
        training_set, y, test_size=0.2, stratify = y)


x_total = Normalizer().fit_transform(user_X) # fit does nothing.
x_train = Normalizer().fit_transform(X_train) # fit does nothing.
x_test = Normalizer().fit_transform(X_test) # fit does nothing.
'''
'''
X_train = Normalizer().fit_transform(X)

clf_svm = LinearSVC(random_state=0, tol=1e-5)
clf_lr = LinearRegression()

scores = cross_val_score(clf_lr, X_train, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

'''
clf_lr = LogisticRegressionCV(random_state=0, solver='sag', tol=1e-5).fit(x_train, y_train)
clf_svm = LinearSVC(random_state=0, tol=1e-5).fit(x_train, y_train)
clf_lda = LinearDiscriminantAnalysis().fit(x_train, y_train)
neigh = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)

'''
kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=6, max_iter=8).fit_predict(training_set)
agg_clustering = AgglomerativeClustering().fit_predict(training_set)
sc = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit_predict(training_set)
'''


y_pred_lr = clf_lr.predict(x_test)
y_pred_svm = clf_svm.predict(x_test)
y_pred_lda = clf_lda.predict(x_test)
y_pred_knn = neigh.predict(x_test)


#y_pred_kmeans = kmeans.predict(x_test)

print('Classifier\'s Accuracy for linear svm: %0.5f\n' % accuracy_score(y_test, y_pred_svm))

print('Classifier\'s Accuracy for Logistic Regression: %0.5f\n' % accuracy_score(y_test, y_pred_lr))

print('Classifier\'s Accuracy for LDA: %0.5f\n' % accuracy_score(y_test, y_pred_lda))

print('Classifier\'s Accuracy for kNN: %0.5f\n' % accuracy_score(y_test, y_pred_knn))


