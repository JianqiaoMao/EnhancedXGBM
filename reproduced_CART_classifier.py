# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:24:40 2023

@author: NickMao
"""

import numpy as np
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

# Get features and label
data = iris.data
feature_name = iris.feature_names
label = iris.target

# Convert to Dataframe to check the data
dataset_df = pd.DataFrame(data, columns=feature_name)
dataset_df['class'] = label
dataset_df.head(5)



class Node:
    
    def __init__(self, depth, data, labels):
        
        self.depth = depth
        
        self.data = data
        self.labels = labels
        
        self.left = None
        self.right = None
        self.feature = None
        self.value  = None

#root = Node(depth = 0, idx = np.ones(data.shape[0])==1)

class Tree:
    
    def __init__(self, max_depth, max_gini_to_split, min_sample, num_bins):
        
        self.max_depth = max_depth
        self.num_bins = num_bins
        self.max_gini_to_split = max_gini_to_split
        self.min_sample = min_sample
        self.X = None
        self.y = None
        self.feature_importance_ = None
        self.root = None
    
    def if_split(self, node):
        max_depth_criterion = node.depth < self.max_depth
        sample_size_criterion = node.labels.shape[0] >= self.min_sample
        gini_impurity_criterion = self.Gini_impurity(node.labels) <=  self.max_gini_to_split
        if max_depth_criterion and sample_size_criterion and gini_impurity_criterion:
            return True
        else:
            return False
        
    def Gini_impurity(self, y):
        size = y.shape[0]
        G = 1
        for value in np.unique(y):
            G-= (np.sum(y==value)/size)**2
        return G
    
    def percentile_value(self, x):
        percentails = [int(100/self.num_bins)*i for i in range(1,self.num_bins+1)]
        x = np.array(x)
        return np.percentile(x, percentails)
    
    def data_split(self, X, split_feature, split_value):
        
        left_index = np.where(X[:,split_feature] <= split_value)[0]
        right_index = np.where(X[:,split_feature] > split_value)[0]
        
        return left_index, right_index

    def split_decision(self, X, y, node):   
        
        numFeatures = X.shape[1]
        dataSize = X.shape[0]
        bestGini_gain = 1
        best_feature = 0
        best_left_y = np.array(None)
        best_right_y = np.array(None)
        best_left_X = np.array(None)
        best_right_X = np.array(None)
    
        for i in range(numFeatures):
            uniqueVals = self.percentile_value(X[:,i])
            Gini_gain = {}
            
            for value in uniqueVals:
    
                left_index, right_index = self.data_split(X, i, value)
                
                left_X = X[left_index, :]
                right_X = X[right_index, :]
                
                left_y = y[left_index]
                right_y = y[right_index]
    
                prob_left = left_X.shape[0] / dataSize
                prob_right = right_X.shape[0] / dataSize
                
                Gini_left = self.Gini_impurity(left_y)
                Gini_right = self.Gini_impurity(right_y)
                
                Gini_gain[value] = prob_left * Gini_left + prob_right * Gini_right
    
                if Gini_gain[value] < bestGini_gain:
                    bestGini_gain = Gini_gain[value]
                    best_feature = i
                    best_value = value
                    best_left_y = left_y
                    best_right_y = right_y
                    best_left_X = left_X
                    best_right_X = right_X
                    
        node.left = Node(node.depth+1, best_left_X, best_left_y)
        node.right = Node(node.depth+1, best_right_X, best_right_y)
        
        self.feature_importance_[best_feature] += bestGini_gain
        
        return best_feature, best_value, bestGini_gain    
    
    def build_prepare(self):
        self.depth = 0
        self.feature_importance_ = {}
        for i in range(self.X.shape[1]):
            self.feature_importance_[i] = 0
        self.root = Node(depth = 0, data = self.X, labels = self.y)
        
    def expand_tree(self, cur_node):
        
        if cur_node is None:
            return None
        if not self.if_split(cur_node):
            return cur_node
         
        split_feature, split_value, gini_Gain = self.split_decision(cur_node.data, cur_node.labels, cur_node)
        
        cur_node.feature = split_feature
        cur_node.value = split_value
        
        self.expand_tree(cur_node.left)
        self.expand_tree(cur_node.right)
        
    def search_prediction(self, node, x):
        
        if node.left is None and node.right is None:
            return np.argmax(np.bincount(node.labels))
        if x[node.feature] <= node.value:
            node = node.left
        else:
            node = node.right
        return self.search_prediction(node, x)
    
    def predict(self, x):
        return self.search_prediction(self.root, x)
    
class cartClassifier:
    
    def __init__(self, max_depth, max_gini_to_split, min_sample, num_bins):
        self.tree = Tree(max_depth = max_depth, max_gini_to_split = max_gini_to_split, min_sample = min_sample, num_bins = num_bins)
        
    def fit(self, X, y):
        
        self.tree.X = np.array(X)
        self.tree.y = np.array(y)
        
        self.tree.build_prepare()
        self.tree.expand_tree(self.tree.root)
        self.feature_importance_ = self.tree.feature_importance_
        
        return self
    
    def predict(self, X):
        
        return  np.array([self.tree.predict(x) for x in X])
        

cart_classifier = cartClassifier(max_depth = 6, max_gini_to_split = 1, min_sample = 2, num_bins = 20)
cart_classifier.fit(data, label)
pred = cart_classifier.predict(data)





    




    
