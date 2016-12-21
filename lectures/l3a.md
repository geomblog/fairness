---
subtitle: "Lecture 3: Fairness Mechanisms"
title: "GIAN Course on Fairness, Accuracy and Transparency in Machine Learning"
author: "Suresh Venkatasubramanian"
date: "Dec 14, 2016"


---

# Introduction

In the last lecture we discussed strategies for ensuring fairness by modifying the input data. Such an approach is reasonable if you a) want to use a classifier unconstrained by considerations of fairness or b) do not have access to the classifier. But if you can build a fairness-aware classifier, it is possible to employ more direct approaches. 

In this lecture we will survey a number of different ways of building fairness-aware models. The methods used are customized to particular models, so for each kind of model and each definition of fairness we could potentially have a different algorithm. We will review a number of ideas in the literature that cover some of these possible pairings: in the process we will also see the emergence of general principles one might use to design a fairness-aware classifier. 

All models reviewed in this lecture will focus on statistical discrepancy or its variants. In the next lecture we will look at methods that build models for other notions of fairness. 

# Decision Trees

We start with decision trees and statistical parity. The goal is to build a decision tree that respects statistical parity while still being an accurate classifier. Recall that one standard method to build decision trees uses entropy to measure information gain in the tree. As usual, let the input $D$ consist of pairs $(\vec{x}, y)$, and let $g(x)$ denote the protected attribute. The class entropy $H_c(D)$ is constructed by computing the fractions $p_i = |\{ (\vec{x}, y) \in D\mid  y = i\} |/|D|$, and then setting 

$$ H_c(D) = \sum_i p_i  \log \frac{1}{p_i} $$

The *information gain* idea is to determine the splitting attribute and value so that  if we look at the two data sets $D_1, D_2$ obtained after the split, then the information gain 

$$ I_c(D) = H_c(D) - \sum p_i H_c(D_i)$$ 

is maximized, where $p_i = |D_i|/|D|$. 

The new idea here is to incorporate statistical bias considerations into the split function.  Let us define a notion of entropy for the group variable $g(x)$ as before. 

$$ H_g(D) = \sum p_i \log \frac{1}{p_i} $$

where now $p_i$ represents the fraction of items in the data with *group label* $g(x) = i$. We can similarly define the information gain $I_g(D)$. 

The first approach is to try and decouple the class labels from the protected attribute. In other words, we would like a split where the information gain about the group label is *minimized*. The simplest way to do this is to write down a new information gain function that combines the two notions above. In particular, the authors experiment with information gains of the form $I_c(D) - I_g(D)$ and $I_c(D)/I_g(D)$. 

The second approach, somewhat paradoxically, is to do the opposite, and maximize information gain on both variables by using a split function of the form $I_g(D) + I_c(D)$. Once the tree is built, the algorithm then relabels leaf nodes to balance the accuracy/fairness tradeoff. 

Specifically, the algorithm computes, for each leaf, how accuracy will change if the node label is flipped, and how the bias of the tree will change. Once these nunmbers are computed for each node, the algorithm runs a KNAPSACK algorithm to find a set of leaves that together reduce statistical disparity to below the desired threshold while reducing accuracy as little as possible. 

Interesting, the authors find that this latter strategy (build a model and then fix it) appears to work better than baking the fairness criterion into the tree construction. 

# Naïve Bayes

Let us consider the Naïve Bayes model now. Recall that the goal of such a model is to compute the probaiblities $\Pr(C \mid x)$. If an input $x$ can be written as the set of features $x = x_1x_2\ldots x_n$, then the Naïve Bayes model uses Bayes' theorem as well as the assumption of class-conditioned variable independence to write this probability as 

$$ \Pr(C \mid x) = \frac{\Pr(C) \prod_i \Pr(x_i|C)}{\Pr(x)}$$

The approaches proposed by the author all work by modifying the assumed generative structure given above. 

## Making $C$ depend directly on $S$. 

Discrimination in such a model reveals itself in the probabilities $\Pr(C \mid S)$, where $S$ is some protected attribute. In the standard model therefore, the way one might try to fix this is to modify the probabilities $\Pr(S |C)$. The authors point out that the obvious way to do this would increase (or decrease) the number of positive outcomes. Their goal is thus to remove statistical bias while holding $\Pr(C = 1)$ fixed. They do this by modifying the generative model. Rather than the class label controlling all attribute generation via the conditional probabilities $\Pr(x_i|C)$, they posit a *biased* model where the class label $C$ is generated from the sensitive attribute $S$ and then generates other attributes independently as before. This means that the joint distribution of all the variables can written as 

$$ \Pr(C, x) = \Pr(S) \Pr(C\mid S) \prod_i \Pr(x_i|C)$$. 

This change of model means that removing statistical bias can be controlled directly by changing the probabilities $\Pr(C\mid S)$ while still holding $\Pr(C = 1)$ constant

[^1]: This boils down to changing entries in the $2\times 2$ table of values for the joint distribution of $C$ and $S$ so that the marginal probabilities remain the same. 

That is exacly what their first algorithm does. Whle the bias in the model is above some threshold the algorithm updates the probabilities in the joint distribution of $C$ and $S$ so that the marginals remain fixed but the probabilities $\Pr(C = 1|S - 0)$ change. 

## Decoupling the models for different values of $S$. 

The above approach decouples $S$ and $C$ by explicitly adjusting model probabilities so that $P(C\mid S)$ is essentially fixed, which reduces the mutual information between $S$ and $C$ to near zero. But this does not rule out indirect influence of $S$ on $C$ via a different feature $x_i$. To eliminate this influence, we must also decouple $x_i$ and $S$. The idea proposed in this paper is to (again) change the generative model by postulating that $S$ influences *all* other variables, and requiring that in each case the conditional influence does not depend on the value of $S$. The way they achieve this is to create two *copies* of the model, one for $S = 0$ and one for $S=1$. By training two separate models, one for each value of $S$, the algorithm guarantees that any variation in output cannot depend on $S$. At labeling time, the algorithm then picks the appropriate model to use based on the value of $S$. 

The paper also introduces a third variation that introduces a latent variable $L$ that is assumed to be the root cause of the classification (and is independent of $S$). The goal is then to determine $L$ and its influence on the outcome. We will not discuss this technique further. 

# Using a regularizer

In machine learning, a standard way of findiing a good model is to define a loss function that captures the model's effectiveness at classification and then find the model that minimizes cost. To add a constraint, we can either add it as a hard constraint or as a penalty term in the loss function — in this latter form the constraint is called a *regularizer*. This is a flexible strategy: for virtually any model build by cost minimization we can add a fairness-regularizer. The paper by Kamishima et al (*cite*) takes this approach to ensuring fairness in logistic regression. 

I'll focus primarily on how the term in the loss function for fairness is introduced. The actual loss function includes other terms for other desired characteristics that are not as relevant to our discussion. 

The key idea is to quantify the relationship between the sensitive attribute $S$ and the outcome $Y$ via their *mutual information*:

$$ I(Y;S) = \sum_y \sum_s \Pr(y,s)\log \frac{\Pr(y,s)}{Pr(y)Pr(s)} $$

which can further be simplified as 

$$ I(Y;S) = \sum_y \sum_s \Pr(y,s)\log \frac{\Pr(y\mid s)}{Pr(y)} $$

We can express the joint and conditional probabilities in terms of the model. If the output of the model is given by predictions $M(Y \mid S, X, \Theta)$, where $X$ are the remaining attributes and $\Theta$ represents the parameters of the model, then we can write

$$ \Pr(Y, S) = M(Y \mid S, X, \Theta) \Pr(X, S)$$

where $\Pr(X, S)$ represents the distribution of the inputs and can be estimated from the data. Kamishima et al suggest  that we approximate $\Pr(Y |S)$ by the prediction of the model $M(Y|X_S, S, \Theta)$ on the average $X_S$ over all inputs that have a fixed value of $S$. Once this is done, an approximation of $\Pr(Y)$ can be computed by the identity $\Pr(Y) = \sum_s \Pr(S = s) \Pr(Y | S = s)$ with the conditional probability replaced by its approximation. 

Assembling all of this together, we get a single estimated expression for the mutual information $I(Y;S)$ which can then be added as a regularization term to the cost minimization. 

# SVMs

Kun et al generalize the idea of shifting the decision boundary with theoretical backing. They argue that by shifting the boundary by a tuned parameter $\lambda$ in order to maintain statistical parity, they retain certain accuracy characteristics of the original data, relying on results from boosting. 