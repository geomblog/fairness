---
subtitle: "Lecture 3: Fairness Mechanisms (continued)"
title: "GIAN Course on Fairness, Accuracy and Transparency in Machine Learning"
author: "Suresh Venkatasubramanian"
date: "Dec 14, 2016"


---

# Introduction

In the last lecture we discussed strategies for ensuring fairness 

- LFR: define three terms in the cost function. 
  - The first one handles statistical parity: difference between assignments to prototype class $k$ from positive data and assignments to class $k$ from negative group ID are similar for all k. This is averaged over the points, and the difference is averaged over the classes. Note that this is based on a "softmax" assignment of points to clusters. 
  - Second term asks that representation of x in prototype is good. 
  - Third term asks that predictions are accurate. The key is that each predicted value is computed by fixing a prediction for each class, and then averaging over membership in the class. 
- Gummadi et al: an interesting idea of covariance to capture mutual information between sensitive attribute and decision boundary. And a neat way to encode disparate mistreatment via this. 
- Roth et al, and contextual fairness. 
