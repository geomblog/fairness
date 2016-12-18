---
subtitle: "Lecture 3: Fairness Mechanisms"
title: "GIAN Course on Fairness, Accuracy and Transparency in Machine Learning"
author: "Suresh Venkatasubramanian"
date: "Dec 14, 2016"


---

# Introduction

In the last lecture we discussed strategies for ensuring fairness by modifying the input data. Such an approach is reasonable if you a) want to use a classifier unconstrained by considerations of fairness or b) do not have access to the classifier. But if you can build a fairness-aware classifier, it is possible to employ more direct approaches. 

In this lecture we will survey a number of different ways of building fairness-aware models. The methods used are customized to particular models, so for each kind of model and each definition of fairness we could potentially have a different algorithm. We will review a number of ideas in the literature that cover some of these possible pairings: in the process we will also see the emergence of general principles one might use to design a fairness-aware classifier. 

## Decision Trees

We start with decision trees and statistical parity. The goal is to build a decision tree that respects statistical parity while still being an accurate classifier. Recall that one standard method to build decision trees uses entropy to measure information gain in the tree. As usual, let the input $D$ consist of pairs $(\vec{x}, y)$, and let $g(x)$ denote the protected attribute. The class entropy $H_c(D)$ is constructed by computing the fractions $p_i = |\{ (\vec{x}, y) \in D\mid  y = i\} |/|D|$, and then setting 

$$ H_c(D) = \sum_i p_i  \log \frac{1}{p_i} $$

The *information gain* idea is to determine the a splitting attribute and value so that  if we look at the two data sets $D_1, D_2$ obtained after the split, then the information gain 

$$ I_c(D) = H_c(D) - \sum p_i H_c(D_i)$$ 

is maximized, where $p_i = |D_i|/|D|$. 

The new idea here is to incorporate statistical bias considerations into the split function.  Let us define a notion of entropy for the group variable $g(x)$ as before. 

$$ H_g(D) = \sum p_i \log \frac{1}{p_i} $$

where now $p_i$ represents the fraction of items in the data with *group label* $g(x) = i$. We can similarly define the information gain $I_g(D)$. 

The first approach is to try and decouple the class labels from the protected attribute. In other words, we would like a split where the information gain about the group label is *minimized*. The simplest way to do this is to write down a new information gain function that combines the two notions above. In particular, the authors experiment with information gains of the form $I_c(D) - I_g(D)$ and $I_c(D)/I_g(D)$. 

The second approach, somewhat paradoxically, is to do the opposite, and maximize information gain on both variables by using a split function of the form $I_g(D) + I_c(D)$. Once the tree is built, the algorithm then relabels leaf nodes to balance the accuracy/fairness tradeoff. 

Specifically, the algorithm computes, for each leaf, how accuracy will change if the node label is flipped, and how the bias of the tree will change. Once these nunmbers are computed for each node, the algorithm runs a KNAPSACK algorithm to find a set of leaves that together reduce statistical disparity to below the desired threshold while reducing accuracy as little as possible. 

Interesting, the authors find that this latter strategy (build a model and then fix it) appears to work better than baking the fairness criterion into the tree construction. 

## Naive Bayes

Naive Bayes based method: very specific to NB, but has one interesting idea: 

- train two classifiers, one for one group and one for another. And then run separate decisions for each group. This could very well be illegal. 

## Using a regularizer

Regularize with a "prejudice penality" that measures information flow beterrn sensitive attribute $S$ and class variable $C$. 

## SVMs

Kun et al generalize the idea of shifting the decision boundary with theoretical backing. They argue that by shifting the boundary by a tuned parameter $\lambda$ in order to maintain statistical parity, they retain certain accuracy characteristics of the original data, relying on results from boosting. 