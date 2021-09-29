from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

X,y=load_iris(150)
labels=KMeans(3,random_state=0).fit_predict(X)
plt.scatter(X[:,0],X[:,1],s=40,c=labels,cmap='nipy_spectral')

liste=np.zeros((150,5))
for i in range(150):
    liste[i,0]=X[i,0]
    liste[i,1]=X[i,1]
    liste[i,2]=X[i,2]
    liste[i,3]=X[i,3]
    liste[i,4]=labels[i]

Xx = liste[:, [0,1,2,3]]
yy = liste[:, 4]

from sklearn.model_selection import train_test_split

print('Verinin boyutu= ' + str(np.shape(Xx)))
print('Etiketlerin boyutu= ' + str(np.shape(yy)))

X_train, X_test, y_train, y_test = train_test_split(Xx,yy, train_size = 0.45, test_size = 0.55, random_state = 100, stratify = yy)
    
print('Öğrenme verisi= ' + str(np.shape(X_train)))
print('Sınama verisi = ' + str(np.shape(X_test)))

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 100)
dt.fit(X_train,y_train)

y_pred_dt = dt.predict(X_test)
y_pred_proba_dt = dt.predict_proba(X_test)

from sklearn.metrics import roc_auc_score, accuracy_score  
print('isabet değeri : ' + str(accuracy_score(y_pred_dt, y_test)))

import pickle
decision_tree_pkl_filename = 'decision_tree1.pkl'
decision_tree_model_pkl = open(decision_tree_pkl_filename, 'wb')
pickle.dump(dt, decision_tree_model_pkl)
decision_tree_model_pkl.close()
