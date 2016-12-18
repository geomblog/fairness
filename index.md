### Overview

Machine learning has taken over our world, in more ways than we realize. You might get book recommendations, or an efficient route to your destination, or even a winning strategy for a game of Go. But you might also be admitted to college, granted a loan, or hired for a job based on algorithmically enhanced decision-making. We believe machines are neutral arbiters: cold, calculating entities that always make the right decision, that can see patterns that our human minds can’t or won’t. But are they? Or is decision-making-by-algorithm a way to amplify, extend and make inscrutable the biases and discrimination that is prevalent in society?

To answer these questions, we need to go back — all the way to the original ideas of justice and fairness in society. We also need to go forward — towards a mathematical framework for talking about justice and fairness in machine learning. I will talk about the growing landscape of research in algorithmic fairness: how we can reason systematically about biases in algorithms, and how we can make our algorithms fair(er).

### Course Mechanics
This is a short (and intense) course. We'll cover material in two lecture chunks each day. But this is also a **discussion**, on a topic that's still very new and that has fluid boundaries and evolving formalisms. I've provided readings that are technical and non-technical in nature, and I expect (and hope!) that the presentations will provoke discussion, arguments, and new ideas. 

So **please read the provided materials** ahead of the lecture and come prepared with your questions, comments and critiques. You'll benefit the most from the material if you have time to engage with it. 

### Syllabus
*   Dec 11: **Preliminaries**

      Basics of machine learning -- supervised and unsupervised learning, empirical risk minimization, classifiers, regression, training and generalization. 

      *Readings*:
    * Chapters 1 [(PDF)](http://ciml.info/dl/v0_9/ciml-v0_9-ch01.pdf), 3.1-3.4 [(PDF)](http://ciml.info/dl/v0_9/ciml-v0_9-ch03.pdf) and 4 [(PDF)](http://ciml.info/dl/v0_9/ciml-v0_9-ch04.pdf) of Hal Daumé's [excellent book on ML](http://ciml.info). 
    * Hal's [post on the ML development pipeline](http://nlpers.blogspot.com/2016/08/debugging-machine-learning.html). This is framed in terms of the way errors can creep into the modeling process, but it doubles as an excellent explanation of the pipeline itself. 

*   Dec 12: **Automated Decision Making**

	Case studies of the use of machine learning in applications. An introduction to different formal notions of fairness.

	*Readings* (applications):

    * Criminal Justice:
      * Risk Assessment: ([Primer](http://www.datacivilrights.org/pubs/2015-1027/Courts_and_Predictive_Algorithms.pdf), [discussion](http://www.datacivilrights.org/pubs/2015-1027/WDN-Courts_and_Predictive_Algorithms.pdf) from Data and Civil Rights Workshop)
      * Predictive Policing: ([Primer](http://www.datacivilrights.org/pubs/2015-1027/Predictive_Policing.pdf) and [discussion](http://www.datacivilrights.org/pubs/2015-1027/WDN-Predictive_Policing.pdf))
      * Julia Angwin, Jeff Larson, Surya Mattu, Lauren Kirchner, [“Machine Bias"](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
      * [Video showing predictive policing in action](http://fusion.net/story/283896/real-future-episode-12-predictive-policing/)
    * Hiring:
      * [On how AI can be used in hiring](http://venturebeat.com/2016/11/09/ai-is-helping-job-candidates-bypass-resume-bias-and-black-holes/) (by one company that provides solutions). [Another perspective](http://www.ca.com/us/rewrite/articles/application-economy/can-artificial-intelligence-find-the-perfect-hire.html) 
      * [Predicting Voice-elicited emotions](http://delivery.acm.org/10.1145/2790000/2788619/p1969-li.pdf?ip=71.195.244.110&id=2788619&acc=OA&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4037F4931E565B6B&CFID=872166982&CFTOKEN=74413255&__acm__=1480927397_b769b575e3f06e480d52f70766f3a596) (from KDD 2015)
    * Credit Scoring and Loans
      * China's new [citizen scoring system](https://www.washingtonpost.com/world/asia_pacific/chinas-plan-to-organize-its-whole-society-around-big-data-a-rating-for-everyone/2016/10/20/1cd0dd9c-9516-11e6-ae9d-0030ac1899cd_story.html?utm_term=.f8184eeef71d)
    * Education:
      * ["19 Ways Data Analysis Empowered Students and Schools"](https://fpf.org/wp-content/uploads/2016/03/Final_19Times-Data_Mar2016-1.pdf) pages 21-26.
      * [Promise and limits of learning analytics](http://www.chronicle.com.libproxy.ocean.edu:2048/article/This-Chart-Shows-the-Promise/234573)

    *Readings* (notions of fairness):
    * [Discrimination-aware data mining](Discrimination-aware data min- ing).  (*discriminatory classifiers*)
    * [Data preprocessing techniques for classification without discrimination](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=0ahUKEwiLtYnNit7QAhWHiVQKHcUaAE8QFggkMAE&url=https%3A%2F%2Fpdfs.semanticscholar.org%2F1a43%2Fd5a8f3dd82a138c92911befba05ae98add27.pdf&usg=AFQjCNHwZ1vsGzJRLsbv4QoW-gLX3DIyCg&sig2=0TikurXGq184Xoqi7O6eMw).  (*statistical parity*)
    * [Fairness through awareness](https://arxiv.org/abs/1104.3913).  (*individual fairness*)
    * [Certifying and removing disparate impact](https://arxiv.org/abs/1412.3756) (*disparate impact*)
    * [Equality of opportunity in supervised learning](https://arxiv.org/abs/1610.02413) and [Fairness Beyond Disparate Treatment & Disparate Impact: Learning Classification without Disparate Mistreatment](https://arxiv.org/abs/1610.08452) (*equalizing odds*)
    * [Fairness in Classic and Contextual Bandits](https://papers.nips.cc/paper/6355-fairness-in-learning-classic-and-contextual-bandits.pdf) (*fairness in sequential learning*)

* Dec 13: **Fairness Mechanisms**

	Understanding the different techniques for ensuring fairness in classification.

    *Readings* (preprocessing and detection)
    * [Detecting discriminatory rules in a rule-based system](http://pages.di.unipi.it/ruggieri/Papers/tkdd.pdf)
    * [Detecting discriminatory black box decision-making](https://arxiv.org/abs/1412.3756) (and repairing it)
    * [Data preprocessing techniques for classification without discrimination](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=0ahUKEwiLtYnNit7QAhWHiVQKHcUaAE8QFggkMAE&url=https%3A%2F%2Fpdfs.semanticscholar.org%2F1a43%2Fd5a8f3dd82a138c92911befba05ae98add27.pdf&usg=AFQjCNHwZ1vsGzJRLsbv4QoW-gLX3DIyCg&sig2=0TikurXGq184Xoqi7O6eMw). 

*   Dec 14: **Fairness Mechanisms** (continued)

   *Readings* (building a better classifier)

    - [Discrimination Aware Decision Tree Learning](http://wwwis.win.tue.nl/~tcalders/pubs/ICDM2010KCP.pdf)
    - [Three Naive Bayes Approaches for Discrimination-Free Classification](https://pdfs.semanticscholar.org/a087/d3893af0276fe3b41924087670b03997f7af.pdf)
    - [Fairness-aware Learning through Regularization Approach](http://ieeexplore.ieee.org/document/6137441/)
    - [Learning Fair Representations](https://www.cs.toronto.edu/~toni/Papers/icml-final.pdf)
    - [A Confidence-Based Approach for Balancing Fairness and Accuracy](https://arxiv.org/abs/1601.05764)
    - [Fairness Beyond Disparate Treatment & Disparate Impact: Learning Classification without Disparate Mistreatment](https://arxiv.org/abs/1610.08452)
    - [Fairness in Classic and Contextual Bandits](https://papers.nips.cc/paper/6355-fairness-in-learning-classic-and-contextual-bandits.pdf)

*   Dec 15: **Accountability via Influence Estimation**

	Probing black-box decision-makers: estimating influence of features.

    *Readings*:
    * Breiman's idea for testing classifiers ([Section 10](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf))
    * [A peek into the black box](http://link.springer.com/article/10.1007/s10618-014-0368-8)
    * [Algorithmic transparency via quantitative input influence](https://www.andrew.cmu.edu/user/danupam/datta-sen-zick-oakland16.pdf)
    * [Auditing Black-box Models for Indirect Influence.](http://sorelle.friedler.net/papers/auditing_icdm_2016.pdf)

*   Dec 17: **Interpretability**

    Building interpretable models.

	*Readings*:

    * [Comprehensible Classification Models: A Review](http://www.kdd.org/exploration_files/V15-01-01-Freitas.pdf)
    * [Statistical Learning with Sparsity](http://web.stanford.edu/~hastie/StatLearnSparsity/)
    * [Interpretable Models for Recidivism Prediction](https://arxiv.org/pdf/1503.07810v6.pdf) (based on the [SLIM](https://arxiv.org/abs/1405.4047) model)
    * [Rule Extraction from Linear Support Vector Machines](https://pdfs.semanticscholar.org/fee0/648c150b052f4d4754151cb80fe0ea1f828d.pdf)
    * [Comprehensible Credit Scoring Models Using Rule Extraction From Support Vector Machines](https://core.ac.uk/download/pdf/6304402.pdf)
    * [Modeling the Model](http://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)
    * [EU Regulations on Right to Explanations.](https://arxiv.org/abs/1606.08813)

*   Dec 18: **Fairness, Accountability and Transparency in other areas of computer science**

    Beyond classification: unsupervised learning, representations, rankings and verification.
    *Readings*:

	* Gender bias in word embeddings: two views. 
      * [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/abs/1607.06520)
      * [Semantics derived automatically from language corpora necessarily contain human biases](http://randomwalker.info/publications/language-bias.pdf)
    * [Measuring fairness in ranked outputs.](https://arxiv.org/abs/1610.08559)
    * [Fairness as a program property.](https://arxiv.org/abs/1610.06067) 

*   Dec 19: **Belief Systems**

    Axiomatic approaches to thinking about fairness. 

    *Readings*:

    * [On the (im)possibility of fairness](https://arxiv.org/abs/1609.07236)
    * [Equality of Opportunity.](https://en.wikipedia.org/wiki/Equal_opportunity)

      ​


### Contact

Email me at [suresh@cs.utah.edu](mailto:suresh@cs.utah.edu)

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

![](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)
