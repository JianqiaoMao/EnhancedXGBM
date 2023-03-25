# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:45:26 2023

@author: NickMao
"""
import numpy as np


class xgbBaseTree:
    
    class Node:
        
        def __init__(self, depth, data, g, h, score_gain):
            
            self.depth = depth
            
            self.data = data
            self.g = g
            self.h = h
            if np.sum(h) == 0:
                self.y_hat = 0
            else:
                self.y_hat = -np.sum(g)/np.sum(h)
            
            self.score_gain = score_gain
            
            self.left = None
            self.right = None
            self.feature = None
            self.value  = None
    
    def __init__(self, max_depth, min_sample, num_bins, min_score_gain):
        
        self.max_depth = max_depth
        self.num_bins = num_bins
        self.min_score_gain = min_score_gain
        self.min_sample = min_sample
        self.X = None
        self.g = None
        self.h = None
        self.feature_importance_ = None
        self.root = None
    
    def if_split(self, node):
        max_depth_criterion = node.depth < self.max_depth
        sample_size_criterion = node.data.shape[0] >= self.min_sample
        
        if max_depth_criterion and sample_size_criterion:
            return True
        else:
            return False
        
    def score(self, g, h):
        G = np.sum(g)
        H = np.sum(h)
        if H == 0:
            return 0
        return -G**2/(2*H)
    
    def percentile_value(self, x):
        percentails = [int(100/self.num_bins)*i for i in range(1,self.num_bins+1)]
        x = np.array(x)
        return np.percentile(x, percentails)
    
    def data_split(self, X, split_feature, split_value):
        
        left_index = np.where(X[:,split_feature] <= split_value)[0]
        right_index = np.where(X[:,split_feature] > split_value)[0]
        
        return left_index, right_index

    def split_decision(self, X, g, h, node):   
        
        numFeatures = X.shape[1]
        bestScore_gain = 0
        best_feature = 0
        best_left_g = np.array(None)
        best_left_h = np.array(None)
        best_right_g = np.array(None)
        best_right_h = np.array(None)
        best_left_X = np.array(None)
        best_right_X = np.array(None)
    
        for i in range(numFeatures):
            uniqueVals = self.percentile_value(X[:,i])
            score_gain = {}
            
            for value in uniqueVals:
    
                left_index, right_index = self.data_split(X, i, value)
                
                left_X = X[left_index, :]
                right_X = X[right_index, :]
                
                left_g = g[left_index]
                left_h = h[left_index]
                right_g = g[right_index]
                right_h = h[right_index]
                
                # prob_left = left_X.shape[0] / dataSize
                # prob_right = right_X.shape[0] / dataSize
                
                score_left = self.score(left_g, left_h)
                score_right = self.score(right_g, right_h)
                
                score_gain[value] = self.score(g,h) -  score_left - score_right

                if score_gain[value] >= bestScore_gain:
                    bestScore_gain = score_gain[value]
                    best_feature = i
                    best_value = value
                    best_left_g = left_g
                    best_left_h = left_h
                    best_right_g = right_g
                    best_right_h = right_h
                    best_left_X = left_X
                    best_right_X = right_X
        
        node.score_gain = bestScore_gain

        node.left = self.Node(node.depth+1, best_left_X, best_left_g, best_left_h, score_gain = 0)
        node.right = self.Node(node.depth+1, best_right_X, best_right_g, best_right_h, score_gain = 0)
        
        self.feature_importance_[best_feature] += bestScore_gain
        
        return best_feature, best_value, bestScore_gain    
    
    def build_prepare(self):
        self.depth = 0
        self.feature_importance_ = {}
        for i in range(self.X.shape[1]):
            self.feature_importance_[i] = 0
        self.root = self.Node(depth = 0, data = self.X, g = self.g,  h=self.h, score_gain = 0)
        
    def expand_tree(self, cur_node):
        
        if cur_node is None:
            return None
        if not self.if_split(cur_node):
            return cur_node
         
        split_feature, split_value, bestScore_gain = self.split_decision(cur_node.data, cur_node.g, cur_node.h, cur_node)
        
        cur_node.feature = split_feature
        cur_node.value = split_value
        
        self.expand_tree(cur_node.left)
        self.expand_tree(cur_node.right)
    
    def search_prediction(self, node, x):
        
        if node.left is None and node.right is None:
            return node.y_hat
        if x[node.feature] <= node.value:
            node = node.left
        else:
            node = node.right
        return self.search_prediction(node, x)
    
    def predict(self, x):
        return self.search_prediction(self.root, x)
        
class xgbTreeEstimator:
    
    def __init__(self, max_depth, min_sample, num_bins, min_score_gain):
        self.tree = xgbBaseTree(max_depth = max_depth, min_score_gain = min_score_gain, min_sample = min_sample, num_bins = num_bins)
        
    def fit(self, X, g, h):
        
        self.tree.X = np.array(X)
        self.tree.g = g
        self.tree.h = h
        
        self.tree.build_prepare()
        self.tree.expand_tree(self.tree.root)
        self.feature_importance_ = self.tree.feature_importance_
        
        return self
    
    def predict(self, X):
        
        return  np.array([self.tree.predict(x) for x in X])


class XGboosting:
    
    def __init__(self, n_estimator, max_depth, min_sample, num_bins, min_score_gain, gamma, _lambda):
        
        self.n_estimator = n_estimator
        self.max_depth =  max_depth
        self.min_sample = min_sample
        self.num_bins = num_bins
        self.min_score_gain = min_score_gain
        self.gamma = gamma
        self._lamda = _lambda
        




