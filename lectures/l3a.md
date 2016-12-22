---
subtitle: "Lecture 3: Fairness Mechanisms"
title: "GIAN Course on Fairness, Accuracy and Transparency in Machine Learning"
author: "Suresh Venkatasubramanian"
date: "Dec 14, 2016"
bibliography: fatcs.bib

---

# Introduction

In the last lecture we discussed strategies for ensuring fairness by modifying the input data. Such an approach is reasonable if you a) want to use a classifier unconstrained by considerations of fairness or b) do not have access to the classifier. But if you can build a fairness-aware classifier, it is possible to employ more direct approaches.

In this lecture we will survey a number of different ways of building fairness-aware models. The methods used are customized to particular models, so for each kind of model and each definition of fairness we could potentially have a different algorithm. We will review a number of ideas in the literature that cover some of these possible pairings: in the process we will also see the emergence of general principles one might use to design a fairness-aware classifier.

All models reviewed in this lecture will focus on statistical discrepancy or its variants. In the next lecture we will look at methods that build models for other notions of fairness.

<a name="section1">

# Changing the model construction algorithm

</a>

We start with decision trees and statistical parity. The goal is to build a decision tree that respects statistical parity while still being an accurate classifier. Recall that one standard method to build decision trees uses entropy to measure information gain in the tree. As usual, let the input $D$ consist of pairs $(\vec{x}, y)$, and let $g(x)$ denote the protected attribute. The class entropy $H_c(D)$ is constructed by computing the fractions $p_i = |\{ (\vec{x}, y) \in D\mid  y = i\} |/|D|$, and then setting

$$ H_c(D) = \sum_i p_i  \log \frac{1}{p_i} $$

The *information gain* idea is to determine the splitting attribute and value so that if we look at the two data sets $D_1, D_2$ obtained after the split, then the information gain

$$ I_c(D) = H_c(D) - \sum p_i H_c(D_i)$$

is maximized, where $p_i = |D_i|/|D|$.

The new idea here, proposed by Kamiran et al. is to incorporate statistical bias considerations into the split function. [@kamiran_discrimination_2010] Let us define a notion of entropy for the group variable $g(x)$ as before.

$$ H_g(D) = \sum p_i \log \frac{1}{p_i} $$

where now $p_i$ represents the fraction of items in the data with *group label* $g(x) = i$. We can similarly define the information gain $I_g(D)$.

Recall from last lecture that it is possible to have statistical discrepancy with respect to the protected variables only when we can predict the protected variable from the output labels. The first approach is to try to decouple the class labels from the protected attribute. In other words, we would like a split where the information gain about the group label is *minimized*. The simplest way to do this is to write down a new information gain function that combines the two notions above. In particular, the authors experiment with information gains of the form $I_c(D) - I_g(D)$ and $I_c(D)/I_g(D)$.

The second approach, somewhat paradoxically, is to do the opposite, and maximize information gain on both variables by using a split function of the form $I_g(D) + I_c(D)$. Once the tree is built, the algorithm then relabels leaf nodes to balance the accuracy/fairness tradeoff.

Specifically, the algorithm computes, for each leaf, how accuracy will change if the node label is flipped, and how the bias of the tree will change. Once these numbers are computed for each node, the algorithm runs a KNAPSACK algorithm to find a set of leaves that together reduce statistical disparity to below the desired threshold while reducing accuracy as little as possible.

Interestingly, the authors find that this latter strategy (build a model and then fix it) appears to work better than baking the fairness criterion into the tree construction.

# Changing the underlying probabilistic model

Let us consider the Naïve Bayes model now. Recall that the goal of such a model is to compute the probabilities $\Pr(C \mid x)$. If an input $x$ can be written as the set of features $x = x_1x_2\ldots x_n$, then the Naïve Bayes model uses Bayes' theorem as well as the assumption that variables are independent conditioned on class to write the probabilities as

$$ \Pr(C \mid x) = \frac{\Pr(C) \prod_i \Pr(x_i|C)}{\Pr(x)}$$

The approaches proposed by the Calders et al. work by modifying the assumed generative structure given above. [@calders_three_2010]

## Making $C$ depend directly on $S$

Discrimination in such a model reveals itself in the probabilities $\Pr(C \mid S)$, where $S$ is some protected attribute. In the standard model therefore, the way one might try to fix this is to modify the probabilities $\Pr(S |C)$. The authors point out that the obvious way to do this would increase (or decrease) the number of positive outcomes. Their goal is thus to remove statistical bias while holding $\Pr(C = 1)$ fixed. They do this by modifying the generative model. Rather than the class label controlling all attribute generation via the conditional probabilities $\Pr(x_i|C)$, they propose a *biased* model where the class label $C$ is generated from the sensitive attribute $S$ and then generates other attributes independently as before. This means that the joint distribution of all the variables can be written as

$$ \Pr(C, x) = \Pr(S) \Pr(C\mid S) \prod_i \Pr(x_i|C)$$.

This change of model means that removing statistical bias can be controlled directly by changing the probabilities $\Pr(C\mid S)$ while still holding $\Pr(C = 1)$ constant. [^1]
That is exactly what their first algorithm does. While the bias in the model is above some threshold the algorithm updates the probabilities in the joint distribution of $C$ and $S$ so that the marginals remain fixed but the probabilities $\Pr(C = 1|S = 0)$ change.

[^1]: This boils down to changing entries in the $2\times 2$ table of values for the joint distribution of $C$ and $S$ so that the marginal probabilities remain the same.


## Decoupling the models for different values of $S$

The above approach decouples $S$ and $C$ by explicitly adjusting model probabilities so that $P(C\mid S)$ is essentially fixed, which reduces the mutual information between $S$ and $C$ to near zero. But this does not rule out indirect influence of $S$ on $C$ via a different feature $x_i$. To eliminate this influence, we must also decouple $x_i$ and $S$. The idea proposed in this paper is to (again) change the generative model by postulating that $S$ influences *all* other variables, and requiring that in each case the conditional influence does not depend on the value of $S$. The way they achieve this is to create two *copies* of the model, one for $S = 0$ and one for $S=1$. By training two separate models, one for each value of $S$, the algorithm guarantees that any variation in output cannot depend on $S$. At labeling time, the algorithm then picks the appropriate model to use based on the value of $S$.

The paper also introduces a third variation that introduces a latent variable $L$ that is assumed to be the root cause of the classification (and is independent of $S$). The goal is then to determine $L$ and its influence on the outcome. We will not discuss this technique further.

# Using a regularizer

In machine learning, a standard way of finding a good model is to define a loss function that captures the model's effectiveness at classification and then find the model that minimizes cost. To add a constraint, we can either add it as a hard constraint or as a penalty term in the loss function — in this latter form the constraint is called a *regularizer*. This is a flexible strategy, for virtually any model build by cost minimization we can add a fairness-regularizer. The paper by Kamishima et al. [@kamishima_fairness-aware_2011] takes this approach to ensuring fairness in logistic regression.

I'll focus primarily on how the term in the loss function for fairness is introduced. The actual loss function includes other terms for other desired characteristics that are not as relevant to our discussion.

The key idea is to quantify the relationship between the sensitive attribute $S$ and the outcome $Y$ via their *mutual information*:

$$ I(Y;S) = \sum_y \sum_s \Pr(y,s)\log \frac{\Pr(y,s)}{Pr(y)Pr(s)} $$

which can further be simplified as

$$ I(Y;S) = \sum_y \sum_s \Pr(y,s)\log \frac{\Pr(y\mid s)}{Pr(y)} $$

We can express the joint and conditional probabilities in terms of the model. If the output of the model is given by predictions $M(Y \mid S, X, \Theta)$, where $X$ are the remaining attributes and $\Theta$ represents the parameters of the model, then we can write

$$ \Pr(Y, S) = M(Y \mid S, X, \Theta) \Pr(X, S)$$

where $\Pr(X, S)$ represents the distribution of the inputs and can be estimated from the data. Kamishima et al suggest  that we approximate $\Pr(Y |S)$ by the prediction of the model $M(Y|X_S, S, \Theta)$ on the average $X_S$ over all inputs that have a fixed value of $S$. Once this is done, an approximation of $\Pr(Y)$ can be computed by the identity $\Pr(Y) = \sum_s \Pr(S = s) \Pr(Y | S = s)$ with the conditional probability replaced by its approximation.

Assembling all of this together, we get a single estimated expression for the mutual information $I(Y;S)$ which can then be added as a regularization term to the cost minimization.

# Shifting the decision boundary

In Section [one](#section1), one of the strategies employed was to build a classifier and then change labels at the leaves. A similar idea (changing the decision boundary) can be employed in a more geometric setting with any technique where the decision boundary can be expressed as a sign of a *confidence* function. The method we will describe applies for example to methods like logistic regression, AdaBoost as well as SVMs.

At a high level, the idea [@zafar_fairness_2016] is the following. Suppose that the classifier can be expressed as $y = \text{sign}(\text{conf}(x))$ where $\text{conf}(x)$ is some function from $\mathbb{R}^d$ to $\mathbb{R}$. Fix a parameter $\lambda$ and define a new classifier

$$ h_\lambda(x) = \begin{cases} 1 & \text{conf}(x) \ge -\lambda \\ \text{sign}(\text{conf}(x)) & \text{otherwise} \end{cases} $$

The parameter $\lambda$ *shifts* the decision boundary away from $0$. Now we can choose $\lambda$ so as to ensure statistical parity. The key insight behind this method is that the generalization accuracy of such a classification procedure depends on the *margin*: the gap between in $\text{conf}(x)$ between the last point classified as $-1$ and the first point classified as $+1$. Shifting the decision boundary in this manner changes the margin, but does so in a way that we can still prove bounds on the accuracy of such a shifted classifier, obtaining a clearer tradeoff between fairness and accuracy.
