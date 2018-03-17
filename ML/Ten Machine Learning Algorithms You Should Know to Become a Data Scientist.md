
Machine Learning Practitioners have different personalities. While some of them are “I am an expert in X and X can train on any type of data”, where X = some algorithm, some others are “Right tool for the right job people”. A lot of them also subscribe to “Jack of all trades. Master of one” strategy, where they have one area of deep expertise and know slightly about different fields of Machine Learning. That said, no one can deny the fact that as practicing Data Scientists, we will have to know basics of some common machine learning algorithms, which would help us engage with a new-domain problem we come across. This is a whirlwind tour of common machine learning algorithms and quick resources about them which can help you get started on them.

## 1. Principal Component Analysis(PCA)/SVD

PCA is an unsupervised method to understand global properties of a dataset consisting of vectors. Covariance Matrix of data points is analyzed here to understand what dimensions(mostly)/ data points (sometimes) are more important (ie have high variance amongst themselves, but low covariance with others). One way to think of top PCs of a matrix is to think of its eigenvectors with highest eigenvalues. SVD is essentially a way to calculate ordered components too, but you don’t need to get the covariance matrix of points to get it.

![image](https://user-images.githubusercontent.com/31998957/37551305-667370e2-29d8-11e8-8f23-68a8bfae9426.png)


This Algorithm helps one fight curse of dimensionality by getting datapoints with reduced dimensions.

Libraries:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html

http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

Introductory Tutorial:
https://arxiv.org/pdf/1404.1100.pdf


## 2a. Least Squares and Polynomial Fitting

Remember your Numerical Analysis code in college, where you used to fit lines and curves to points to get an equation. You can use them to fit curves in Machine Learning for very small datasets with low dimensions. (For large data or datasets with many dimensions, you might just end up terribly overfitting, so don’t bother). OLS has a closed form solution, so you don’t need to use complex optimization techniques.

![image](https://user-images.githubusercontent.com/31998957/37551327-c715853e-29d8-11e8-8077-67affde6f95a.png)

As is obvious, use this algorithm to fit simple curves / regression

Libraries:
https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.htmlhttps://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.polyfit.html

Introductory Tutorial:
https://lagunita.stanford.edu/c4x/HumanitiesScience/StatLearning/asset/linear_regression.pdf

## 2b. Constrained Linear Regression
Least Squares can get confused with outliers, spurious fields and noise in data. We thus need constraints to decrease the variance of the line we fit on a dataset. The right method to do it is to fit a linear regression model which will ensure that the weights do not misbehave. Models can have L1 norm (LASSO) or L2 (Ridge Regression) or both (elastic regression). Mean Squared Loss is optimized.

![image](https://user-images.githubusercontent.com/31998957/37551334-e1afd7f0-29d8-11e8-8ff8-b57d0e61a1d8.png)

Use these algorithms to fit regression lines with constraints, avoiding overfitting and masking noise dimensions from model.

Libraries:
http://scikit-learn.org/stable/modules/linear_model.html

Introductory Tutorial(s):
https://www.youtube.com/watch?v=5asL5Eq2x0A

https://www.youtube.com/watch?v=jbwSCwoT51M

## 3. K means Clustering
Everyone’s favorite unsupervised clustering algorithm. Given a set of data points in form of vectors, we can make clusters of points based on distances between them. It’s an Expectation Maximization algorithm that iteratively moves the centers of clusters and then clubs points with each cluster centers. The input the algorithm has taken is the number of clusters which are to be generated and the number of iterations in which it will try to converge clusters.

![image](https://user-images.githubusercontent.com/31998957/37551345-08bc7ab0-29d9-11e8-974e-7e8d088fd2b7.png)

As is obvious from the name, you can use this algorithm to create K clusters in dataset

Library:
http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

Introductory Tutorial(s):
https://www.youtube.com/watch?v=hDmNF9JG3lo

https://www.datascience.com/blog/k-means-clustering

## 4. Logistic Regression
Logistic Regression is constrained Linear Regression with a nonlinearity (sigmoid function is used mostly or you can use tanh too) application after weights are applied, hence restricting the outputs close to +/- classes (which is 1 and 0 in case of sigmoid). Cross-Entropy Loss functions are optimized using Gradient Descent. A note to beginners: Logistic Regression is used for classification, not regression. You can also think of Logistic regression as a one layered Neural Network. Logistic Regression is trained using optimization methods like Gradient Descent or L-BFGS. NLP people will often use it with the name of Maximum Entropy Classifier.

This is what a Sigmoid looks like:

![image](https://user-images.githubusercontent.com/31998957/37551352-1ba9f602-29d9-11e8-9193-05be27e47514.png)

Use LR to train simple, but very robust classifiers.

Library:
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

Introductory Tutorial(s):
https://www.youtube.com/watch?v=-la3q9d7AKQ

## 5. SVM (Support Vector Machines)

SVMs are linear models like Linear/ Logistic Regression, the difference is that they have different margin-based loss function (The derivation of Support Vectors is one of the most beautiful mathematical results I have seen along with eigenvalue calculation). You can optimize the loss function using optimization methods like L-BFGS or even SGD.

![image](https://user-images.githubusercontent.com/31998957/37551355-328675a8-29d9-11e8-94af-5a097495f715.png)

Another innovation in SVMs is the usage of kernels on data to feature engineer. If you have good domain insight, you can replace the good-old RBF kernel with smarter ones and profit.

One unique thing that SVMs can do is learn one class classifiers.

SVMs can used to Train a classifier (even regressors)

Library:
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

Introductory Tutorial(s):
https://www.youtube.com/watch?v=eHsErlPJWUU

Note: SGD based training of both Logistic Regression and SVMs are found in SKLearn’s http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html , which I often use as it lets me check both LR and SVM with a common interface. You can also train it on >RAM sized datasets using mini batches.

## 6. Feedforward Neural Networks
These are basically multilayered Logistic Regression classifiers. Many layers of weights separated by non-linearities (sigmoid, tanh, relu + softmax and the cool new selu). Another popular name for them is Multi-Layered Perceptrons. FFNNs can be used for classification and unsupervised feature learning as autoencoders.

![image](https://user-images.githubusercontent.com/31998957/37551365-468c8042-29d9-11e8-98c7-947244b0d42c.png)
Multi-Layered perceptron

![image](https://user-images.githubusercontent.com/31998957/37551369-531a701c-29d9-11e8-87ae-0ff2fe87b452.png)


FFNNs can be used to train a classifier or extract features as autoencoders

Libraries:
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html

https://github.com/keras-team/keras/blob/master/examples/reuters_mlp_relu_vs_selu.py

Introductory Tutorial(s):
http://www.deeplearningbook.org/contents/mlp.html

http://www.deeplearningbook.org/contents/autoencoders.html

http://www.deeplearningbook.org/contents/representation.html

## 7. Convolutional Neural Networks (Convnets)
Almost any state of the art Vision based Machine Learning result in the world today has been achieved using Convolutional Neural Networks. They can be used for Image classification, Object Detection or even segmentation of images. Invented by Yann Lecun in late 80s-early 90s, Convnets feature convolutional layers which act as hierarchical feature extractors. You can use them in text too (and even graphs).

![image](https://user-images.githubusercontent.com/31998957/37551376-6a06659c-29d9-11e8-8c5c-891fbc69115e.png)

Use convnets for state of the art image and text classification, object detection, image segmentation.

Libraries:
https://developer.nvidia.com/digits

https://github.com/kuangliu/torchcv

https://github.com/chainer/chainercv

https://keras.io/applications/

Introductory Tutorial(s):
http://cs231n.github.io/

https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/

## 8. Recurrent Neural Networks (RNNs):
RNNs model sequences by applying the same set of weights recursively on the aggregator state at a time t and input at a time t (Given a sequence has inputs at times 0..t..T, and have a hidden state at each time t which is output from t-1 step of RNN). Pure RNNs are rarely used now but its counterparts like LSTMs and GRUs are state of the art in most sequence modeling tasks.

![image](https://user-images.githubusercontent.com/31998957/37551382-8900a91c-29d9-11e8-9ef6-673da22ebdfe.png)

RNN (If here is a densely connected unit and a nonlinearity, nowadays f is generally LSTMs or GRUs ). LSTM unit which is used instead of a plain dense layer in a pure RNN.

![image](https://user-images.githubusercontent.com/31998957/37551387-970b3090-29d9-11e8-8e5c-bf6f8c115f93.png)

Use RNNs for any sequence modelling task specially text classification, machine translation, language modelling

Library:
https://github.com/tensorflow/models (Many cool NLP research papers from Google are here)

https://github.com/wabyking/TextClassificationBenchmark

http://opennmt.net/

Introductory Tutorial(s):
http://cs224d.stanford.edu/

http://www.wildml.com/category/neural-networks/recurrent-neural-networks/

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

## 9. Conditional Random Fields (CRFs)
CRFs are probably the most frequently used models from the family of Probabilitic Graphical Models (PGMs). They are used for sequence modeling like RNNs and can be used in combination with RNNs too. Before Neural Machine Translation systems came in CRFs were the state of the art and in many sequence tagging tasks with small datasets, they will still learn better than RNNs which require a larger amount of data to generalize. They can also be used in other structured prediction tasks like Image Segmentation etc. CRF models each element of the sequence (say a sentence) such that neighbors affect a label of a component in a sequence instead of all labels being independent of each other.

Use CRFs to tag sequences (in Text, Image, Time Series, DNA etc.)

Library:
https://sklearn-crfsuite.readthedocs.io/en/latest/

Introductory Tutorial(s):
http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/

7 part lecture series by Hugo Larochelle on Youtube: https://www.youtube.com/watch?v=GF3iSJkgPbA

## 10. Decision Trees
Let’s say I am given an Excel sheet with data about various fruits and I have to tell which look like Apples. What I will do is ask a question “Which fruits are red and round ?” and divide all fruits which answer yes and no to the question. Now, All Red and Round fruits might not be apples and all apples won’t be red and round. So I will ask a question “Which fruits have red or yellow color hints on them? ” on red and round fruits and will ask “Which fruits are green and round ?” on not red and round fruits. Based on these questions I can tell with considerable accuracy which are apples. This cascade of questions is what a decision tree is. However, this is a decision tree based on my intuition. Intuition cannot work on high dimensional and complex data. We have to come up with the cascade of questions automatically by looking at tagged data. That is what Machine Learning based decision trees do. Earlier versions like CART trees were once used for simple data, but with bigger and larger dataset, the bias-variance tradeoff needs to solved with better algorithms. The two common decision trees algorithms used nowadays are Random Forests (which build different classifiers on a random subset of attributes and combine them for output) and Boosting Trees (which train a cascade of trees one on top of others, correcting the mistakes of ones below them).

Decision Trees can be used to classify datapoints (and even regression)

Libraries
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

http://xgboost.readthedocs.io/en/latest/

https://catboost.yandex/

Introductory Tutorial:
http://xgboost.readthedocs.io/en/latest/model.html

https://arxiv.org/abs/1511.05741

https://arxiv.org/abs/1407.7502

http://education.parrotprediction.teachable.com/p/practical-xgboost-in-python

TD Algorithms (Good To Have)
If you are still wondering how can any of the above methods solve tasks like defeating Go world champion like DeepMind did, they cannot. All the 10 type of algorithms we talked about before this was Pattern Recognition, not strategy learners. To learn strategy to solve a multi-step problem like winning a game of chess or playing Atari console, we need to let an agent-free in the world and learn from the rewards/penalties it faces. This type of Machine Learning is called Reinforcement Learning. A lot (not all) of recent successes in the field is a result of combining perception abilities of a convent or a LSTM to a set of algorithms called Temporal Difference Learning. These include Q-Learning, SARSA and some other variants. These algorithms are a smart play on Bellman’s equations to get a loss function that can be trained with rewards an agent gets from the environment.

These algorithms are used to automatically play games mostly :D, also other applications in language generation and object detection.

Libraries:
https://github.com/keras-rl/keras-rl

https://github.com/tensorflow/minigo

Introductory Tutorial(s):
Grab the free Sutton and Barto book: https://web2.qatar.cmu.edu/~gdicaro/15381/additional/SuttonBarto-RL-5Nov17.pdf

Watch David Silver course: https://www.youtube.com/watch?v=2pWv7GOvuf0

These are the 10 machine learning algorithms which you can learn to become a data scientist.

You can also read about machine learning libraries here.

We hope you liked the article. Please Sign Up for a free ParallelDots account to start your AI journey. You can also check demo’s of our APIs here.


