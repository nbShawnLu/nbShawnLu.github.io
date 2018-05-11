---
title: Introduction to Subspace Clustering
tags: ["Subspace Clustering", "Machine Learning"]
date: 2018-05-11
---

## THE SUBSPACE CLUSTERING PROBLEM

**Modeling a collection of data points with a union of subspaces.**  

Specifically,  
Given set of points `$\{x_j\in\mathbb{R}^D\}_{j=1}^N$`  
Drawn from an unknown union of linear or affine subspaces `$\{S_i\}_{i=1}^n$` of unknown dimendions `$d_i=dim(S_i)$`  
The subspaces can be described as `$S_i=\{x\in\mathbb R^D:x=\mu_i+U_iy\}$, i=1,...,n`  
`$y\in\mathbb R^{d_i}$` is a low-dimensional representation for point $x$  

**The goal of subspace clustering is to find the number of subspaces $n$, their dimensions `$\{d_i\}_{i=1}^n$`, the subspace bases `$\{U_i\}_{i=1}^n$`, the points `$\{\mu_i\}_{i=1}^n$`, and the segmentation of the points according to the subspaces.**  

### CHALLENGES

1. Strong coupling between data segmentation and model estimation.  
In practice, neither the segmentation of the data nor the subspace parameters are known, and one needs to solve both problems simultaneously.  
2. The distribution of the data inside the subspaces is generally unknown.  
3. The position and orientation of the subspaces relative to each other can be arbitrary.  
n linear subspaces are disjoint if every two subspaces intersect only at the origin.  
n linear subspaces are independent if the dimension of their sum is equal to the sum of their dimensions.  
Independent subspaces are disjoint, but the converse is not always true.  
4. The data can be corrupted by noise, missing entries, and outliers.  
5. Model selection  
The challenge is to find a model-selection criteria that favors a small number of subspaces of small dimensions.  

## ALGORITHMS

### ALGEBRAIC ALGORITHMS
#### MATRIX FACTORIZATION-BASED ALGORITHMS
**These algorithms obtain the segmentation of the data from a low-rank factorization of the data matri X. Hence, they are a natural extension of PCA from one to multiple independent linear subspaces.**  
Specifically,  
Let `$X_i\in\mathbb{R}^{D\times{N_i}}$` be the matrix containing the $N_i$ points in subspace $i$  
The columns of the data matrix can be sorted according to the n subspaces as `$\begin{bmatrix}X_1 & X_2 & \cdots & X_n\end{bmatrix} = XT$`, where $T\in\mathbb{R}^{N\times N}$is an unknown permutation matrix.  
It can be factorized as `$X_i=U_iY_i$` where `$U_i\in\mathbb{R}^{D\times d_i}$` is an orthogonal basis for subspace $i$ and `$Y_i\in\mathbb{R}^{d_i\times N_i}$` is the low-dimensional representation of the points.  
Therefore, if the subspaces are **independent**, then `$r\triangleq rank(X)=\sum_{i=1}^{n}{d_i}\le{min\{D,N\}}$` and
`$$
XT=\begin{bmatrix}U_1 & U_2 & ... & U_n\end{bmatrix}\begin{bmatrix}Y_1 & & &\\ & Y_2 & &\\& & \ddots & \\ & & & Y_n\end{bmatrix}\triangleq UY
$$`  
where $U\in\mathbb{R}^{D\times r}$ and $Y\in\mathbb{R}^{r\times N}$  
**The subspace clustering problem is then equivalent to finding a permutation matrix $T$, such that $XT$ admits a rank-r factorization into a matrix $U$ and a block diagonal matrix $Y$.**  
Specifically,  
Let $X = U\Sigma V^T$ be the rank-r SVD of the data matrix,  $U\in\mathbb{R}^{D\times r}$,  $\Sigma\in\mathbb{R}^{r\times r}$, $V\in\mathbb{R}^{N\times r}$, let $Q=VV^T\in\mathbb{R}^{N\times N}$, in the absence of noise, can be used to obtain the segmentation of the data by applying spectral clustering to the eigenvectors
of Q or by sorting and thresholding the entries of Q.  

Cons:  
1. Sensitive to noise  
2. Requires knowledge of the rank of $X$  
3. Do not provide a method for computing the number of subspaces, n

//to be continued
#### GENERALIZED PCA

#### ITERATIVE METHODS

### STATISTICAL METHODS

### SPECTRAL CLUSTERING-BASED METHODS



## REFERENCES

Parsons, Lance, Ehtesham Haque, and Huan Liu. "Subspace clustering for high dimensional data: a review." Acm Sigkdd Explorations Newsletter 6.1 (2004): 90-105.  

Vidal, René. "Subspace clustering." IEEE Signal Processing Magazine 28.2 (2011): 52-68.  


