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

### ·ALGEBRAIC ALGORITHMS

#### MATRIX FACTORIZATION-BASED ALGORITHMS
**These algorithms obtain the segmentation of the data from a low-rank factorization of the data matri X. Hence, they are a natural extension of PCA from one to multiple independent linear subspaces.**  
Specifically,  
Let `$X_i\in\mathbb{R}^{D\times{N_i}}$` be the matrix containing the `$N_i$` points in subspace $i$  
The columns of the data matrix can be sorted according to the n subspaces as `$\begin{bmatrix}X_1 & X_2 & \cdots & X_n\end{bmatrix} = XT$`, where $T\in\mathbb{R}^{N\times N}$is an unknown permutation matrix.  
It can be factorized as `$X_i=U_iY_i$` where `$U_i\in\mathbb{R}^{D\times d_i}$` is an orthogonal basis for subspace $i$ and `$Y_i\in\mathbb{R}^{d_i\times N_i}$` is the low-dimensional representation of the points.  
Therefore, if the subspaces are **independent**, then `$r\triangleq rank(X)=\sum_{i=1}^{n}{d_i}\le{min\{D,N\}}$` and
`$$
XT=\begin{bmatrix}U_1 & U_2 & \cdots & U_n\end{bmatrix}\begin{bmatrix}Y_1 & & &\\ & Y_2 & &\\& & \ddots & \\ & & & Y_n\end{bmatrix}\triangleq UY
$$`  
where $U\in\mathbb{R}^{D\times r}$ and $Y\in\mathbb{R}^{r\times N}$  
**The subspace clustering problem is then equivalent to finding a permutation matrix $T$, such that $XT$ admits a rank-r factorization into a matrix $U$ and a block diagonal matrix $Y$.**  
Specifically,  
Let $X = U\Sigma V^T$ be the rank-r SVD of the data matrix,  $U\in\mathbb{R}^{D\times r}$,  $\Sigma\in\mathbb{R}^{r\times r}$, $V\in\mathbb{R}^{N\times r}$, let $Q=VV^T\in\mathbb{R}^{N\times N}$, in the absence of noise, can be used to obtain the segmentation of the data by applying spectral clustering to the eigenvectors
of Q or by sorting and thresholding the entries of Q.  

Cons:  
1. Sensitive to noise  
2. Requires knowledge of the rank of $X$  
3. Do not provide a method for computing the number of subspaces, $n$

#### GENERALIZED PCA
**The main idea behind GPCA is that one can fit a union of n subspaces with a set of polynomials of degree n, whose derivatives at a point give a vector normal to the subspace containing that point.**  
**The first step,** is to project the data points onto a subspace of $\mathbb{R}^D$ of dimension `$r=d_{max}+1$`  
**The second step,** is to fit a homogeneous polynomial of degree $n$ to the (projected) data.  
For instance, that the data came from the union of two planes in $\mathbb{R}^3$, each one with normal vector `$b_i\in\mathbb{R}^3$`. The union of the two planes can be represented as a set of points, such that `$p(x) = (b_1^Tx)(b_2^Tx)=0$`.The equation of a conic of the form `$c_1x_1^2+c_2x_1x_2+c_3x_1x_3+c_4x_2^2+c_5x_2x_3+c_6x_3^2=0$`.  
More generally, data drawn from the union of n subspaces of $\mathbb{R}^r$ can be represented with polynomials of the form `$p(x)=(b_1^Tx)...(b_n^Tx)=0$`, where the vector `$b_i\in\mathbb{R}^r$` is orthogonal to `$S_i$`. Each polynomial is of degree $n$ in $x$ and can be written as `$c^Tv_n(x)$`, where $c$ is the vector of coefficients and `$v_n(x)$` is the vector of all monomials of degree $n$ in $x$. There are `$$M_n(r)=\binom{n+r-1}{n}$$` independent monomials; hence,`$c\in\mathbb{R}^{M_n(r)}$`.  
In the case of noiseless data, the vector of coefficients $c$ of each polynomial can be computed from `$c^T\begin{bmatrix}v_n(x_1)&v_n(x_2)&\cdots&v_n(x_N)\end{bmatrix}\triangleq c^TV_n=0^T$` and the number of polynomials is simply the dimension of the null space of `$V_n$`.  
**The last step,** is to compute the normal vectors `$b_i$` from the vector of coefficients $c$.  
This can be done by taking the derivatives of the polynomials at a data point. `$\nabla p(x)\sim b_i$` if `$x\in S_i$`.  

Pros:  
1. It is an algebraic algorithm; thus, it is computationally cheap when n and d are small.  
2. Intersections between subspaces are automatically allowed; hence, GPCA can deal with both independent and dependent subspaces.  
3. In the noiseless case, it does not require the number of subspaces or their dimensions to be known beforehand.  
Cons:  
1. Its complexity increases exponentially with $n$ and `${d_i}$`.  
2. The vector $c$ is computed using least squares; thus, the computation of $c$ is sensitive to outliers.  
3. The least-squares fit does not take into account nonlinear constraints among the entries of $c$.  

#### ITERATIVE METHODS
**Given an initial segmentation, we can fit a subspace to each group using classical PCA. Then, given a PCA model for each subspace, we can assign each data point to its closest subspace. By iterating these two steps, we can obtain a refined estimate of the subspaces and segmentation.**  
Let `$w_{ij}=1$` if point $j$ belongs to subspace $i$ and `$w_{ij}=0$` otherwise.  
We can do so by minimizing the sum of the squared distances from each data point to its own subspace
`$$
\min_{\{\mu_i\},\{U_i\},\{y_i\},\{w_i\}}\quad\sum_{i=1}^n\sum_{j=1}^Nw_{ij}||x_j-\mu_i-U_iy_j||^2\\
s.t.\quad w_{ij}\in\{0,1\}\quad and\quad \sum_{i=1}^nw_{ij}=1.
$$`
Given `$\{\mu_i\}$`,`$\{U_i\}$`, and `$\{y_i\}$`, the optimal value for `$w_{ij}$` is
`$$
w_{ij}=\begin{cases}1\quad if\quad i=arg\min_{k=1,...,n}||x_j-\mu_k-U_ky_j||^2\\
0\quad else\end{cases}
$$`
Given `$w_{ij}=1$`, the cost function decouples as the sum of $n$ cost functions, one per subspace. Since each cost function is identical to that minimized by standard PCA, the optimal values for `$\mu_i$`,`$U_i$`, and `$y_i$` are obtained by applying PCA to each group of points.  

Pros:  
1. Simplicity since it alternates between assigning points to subspaces and estimating the subspaces via PCA.  
2. It can handle both linear and affine subspaces explicitly.  
3. It converges to a local optimum in a finite number of iterations.  
Cons:  
1. Its convergence to the global optimum depends on a good initialization.  
2. K-subspaces is sensitive to outliers, partly due to the use of the l2-norm.  
3. K-subspaces requires $n$ and `$\{d_i\}_{i=1}^n$`to be known beforehand.  

### ·STATISTICAL METHODS

#### MIXTURE OF PROBABILISTIC PCA
Probabilistic PCA(PPCA) assumes that the data within a subspace S is generated as $x=\mu+Uy+\epsilon$, where $y$ and $\epsilon$ are independent zero-mean Gaussian random vectors with covariance matrices $I$ and $\sigma^2I$. Therefore, $x$ is also Gaussian with mean $\mu$ and covariance matrix $\Sigma=UU^T+\sigma^2I$.  
PPCA can be naturally extended to a generative model for a union of subspaces `$\cup_{i=1}^nS_i$` by using a mixture of PPCA (MPPCA) model.  
**MPPCA uses a mixture of Gaussians model**
`$$
p(x)=\sum_{i=1}^n\pi_iG(x;\mu_i,U_iU_i^T+\sigma_i^2I),\ \sum_{i=1}^n\pi_i=1
$$`
`$\pi_i$` represents the a priori probability of drawing a point from subspace `$S_i$`.  
**The ML estimates of the parameters of this mixture model can be found using expectation maximization(EM).**  
Pros  
1. It is a simple and intuitive method, where each iteration can be computed in closed form by using PPCA.  
2. Applicable to both linear and affine subspaces and can be extended to accommodate outliers and missing entries in the data points.  
Cons  
1. The number and dimensions of the subspaces need to be known beforehand.  
2. Not optimal when the data inside each subspace or the noise is not Gaussian.  
3. Often converges to a local maximum.  

#### AGGLOMERATIVE LOSSY COMPRESSION
**Unlike MPPCA, ALC does not aim to obtain an ML estimate of the parameters of the mixture model. Instead, it looks for the segmentation of the data that minimizes the coding length needed to fit the points with a mixture of degenerate Gaussians up to a given distortion.**  
Specifically, the number of bits needed to optimally code $N$ independent identically distributed(i.i.d.) samples from a zeromean D-dimensional Gaussian, i.e., $X\in\mathbb{R}^{D\times N}$, up to a distortion $\delta$ can be approximated as `$[(N+D)/2]log_2det(I+(D/\delta^2N)XX^T)$`. Thus, the total number of bits for coding amixture of Gaussians can be approximated as
`$$
\sum_{i=1}^n\frac{N_i+D}{2}log_2det(I+\frac{D}{\delta^2N_i}X_iX_i^T)-N_ilog_2(\frac{N_i}{N}),
$$`
where `$X_i\in\mathbb{R}^{D\times N_i}$` is the data from subspace $i$, and the last term is the number of bits needed to code (losslessly) the membership of the $N$ samples to the $n$ groups.  
**ALC deals with this issue by using an agglomerative clustering method.** Initially, each data point is considered as a separate group. At each iteration, two groups are merged if doing so results in the greatest decrease of the coding length. The algorithm terminates when the coding length cannot be further decreased.  
Pros  
1. Can naturally handle noise and outliers in the data.  
2. ALC does not need to know the number of subspaces and their dimensions.  
Cons  
1. There is no theoretical proof for the optimality of the agglomerative procedure.  

#### RANDOM SAMPLE CONSENSUS
**Random sample consensus (RANSAC) is a statistical method for fitting amodel to a cloud of points corrupted with outliers in a statistically robust way.**  
More specifically, if $d$ is the minimum number of points required to fit a model to the data, RANSAC randomly samples $d$ points from the data, fits a model to these $d$ points, computes the residual of each data point to this model, and chooses the points whose residual is below a threshold as the inliers. The procedure is then repeated for $d$ sample points, until the number of inliers is above a threshold, or enough samples have been drawn. The outputs of the algorithm are the parameters of the model and the labeling of inliers and outliers.  

Pros  
1. Ability to handle outliers explicitly.  
2. Does not require the subspaces to be independent, because it computes one subspace at a time.  
3. Does not need to know the number of subspaces beforehand.  
Cons  
1. Performance deteriorates quickly as the number of subspaces n increases.  
2. It requires the dimension of the subspaces to be known and equal.  

### ·SPECTRAL CLUSTERING-BASED METHODS
> This part will be a separate post.


## REFERENCES

Parsons, Lance, Ehtesham Haque, and Huan Liu. "Subspace clustering for high dimensional data: a review." Acm Sigkdd Explorations Newsletter 6.1 (2004): 90-105.  

Vidal, René. "Subspace clustering." IEEE Signal Processing Magazine 28.2 (2011): 52-68.  


