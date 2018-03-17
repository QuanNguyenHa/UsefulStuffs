
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

