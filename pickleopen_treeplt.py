from sklearn.datasets import load_iris

iris=load_iris()

import pickle
decision_tree_model_pkl = open("decision_tree_001.pkl", 'rb')
decision_tree_model = pickle.load(decision_tree_model_pkl)
print ("Loaded Decision tree model :: ", decision_tree_model)

from sklearn import tree
from matplotlib import pyplot as plt

plt.figure(figsize=(20,10),dpi=100)
tree.plot_tree(decision_tree_model,
               feature_names=iris.feature_names,
               rounded=True,
               filled=True,
               class_names=iris.target_names,
               impurity=True)
