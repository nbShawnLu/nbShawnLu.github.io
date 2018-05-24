---
title: 谱聚类了解一下
tags: ["Spectral Clustering", "Machine Learning"]
date: 2018-05-24
---

## 引言
谱聚类是一种利用样本间的相似矩阵的特征值（谱），来对数据降维以更好的进行聚类的方法。相比传统的k-means等聚类方法，谱聚类往往能表现的更好，特别是能够保留局部的连接性，并且非常容易实现。  
谱聚类方法和图方法有很大的关联，综述[1]首先回顾了基本的图记号，之后描述了拉普拉斯矩阵的一些重要性质，最后从图分割、随机游走、摄动理论三个角度来对谱聚类做出解释。  

## 图的标记和相似图
给定一系列的数据点`$\{x_i\},i=1,..,n$`，`$x_i$`和`$x_j$`的相似性记作`$s_{ij}$`（非负），一个相似图可以被记作$G=(V,E)$ 。其中，$V$代表数据点的集合，$E$代表边的集合，对于点`$x_i$`和`$x_j$`之间的边，相似性`$s_{ij}$`。  
记无向图的带权邻接矩阵`$W=(w_{ij})_{i,j=1,...,n}$`，`$w_{ij}=0$`表示`$v_i$`和`$v_j$`不相连，由于是无向图，`$w_{ij}=w_{ji}$`。  
记`$v_i$`的度为`$d_i=\sum_{j=1}^nw_{ij}$`，定义对角矩阵`$D_{ii}=d_i$`。  
对于$V$的一个子集$A$，定义补集`$V\backslash A=\bar{A}$`。定义指示向量(indicator vector)`$\mathbb{1}_A=(f_1,...,f_n)'\in\mathbb{R}^n$`。当`$v_i\in A$`时`$f_i=1$`，否则为0。  
定义$A$的size：$|A|$为A的顶点个数。$vol(A)$为$A$中所有点的度之和。  

### 三种的相似图
1. The $\epsilon$-neighborhood graph:  
只连接距离小于$\epsilon$的顶点对。  
2. $k$-nearest neighbor graphs:  
只连接距离每个顶点最近的k个顶点。由于需要生成的是无向图，所以对于`$v_i$`在`$v_j$`邻域而`$v_j$`不在`$v_i$`邻域的点对，我们可以选择都保留或者都忽略。  
3. The fully connected graph:  
保留所有点对，通常使用高斯函数来表示相似性。  

## Laplacian矩阵及基本性质

### The unnormalized graph Laplacian
`$L=D-W$`  
**Properties of L:**  
1. 对于所有向量$f\in\mathbb{R}^n$，都有`$f'Lf=\sum_{i,j=1}^nw_{ij}(f_i-f_j)^2$`。  
2. L是个对称的半正定矩阵。  
3. L最小的特征值为0，对应的特征向量为$\mathbb{1}$。  
4. L的特征值非负。  
5. 对于一个无向图$G$，对应的连接权重$W$非负。$L$特征值$0$的度$k$为连通分量`$A_1,...,A_k$`的个数，特征值0对应的特征向量被指示向量`$\mathbb{1}_{A_1},...,\mathbb{1}_{A_k}$`张成。  

### The normalized graph Laplacians
`$L_{sym}:=D^{-1/2}LD^{-1/2}=I-D^{-1/2}WD^{-1/2}$`  
（概括的说就是$L$的每一行除以`$\sqrt{d_i}$`,然后每一列除以`$\sqrt{d_i}$`。）  
`$L_{rw}:=D^{-1}L=I-D^{-1}W$`  
（概括的说就是$L$的每一行除以`$d_i$`）  
**Properties of `$L_{sym}$` and `$L_{rw}$`**  
1. 对于所有向量$f\in\mathbb{R}^n$，都有`$f'L_{sym}f=\frac{1}{2}\sum_{i,j=1}^nw_{ij}(\frac{f_i}{\sqrt{d_i}}-\frac{f_j}{\sqrt{d_i}})^2$`。  
2. $\lambda$是`$L_{rw}$`对于特征向量$v$的特征值  IFF $\lambda$是`$L_{sym}$`对于特征向量`$w=D^{1/2}v$`的特征值。    
3. $\lambda$是`$L_{rw}$`对于特征向量$v$的特征值  IFF $Lv=\lambda Dv$  
4. $0$是`$L_{rw}$`对于特征向量$\mathbb{1}$的特征值。$0$是`$L_{sym}$`对于特征向量`$D^{1/2}\mathbb{1}$`的特征值。  
5. `$L_{rw}$`和`$L_{sym}$`都是半正定矩阵，特征值非负。  
6. 对于一个无向图$G$，对应的连接权重$W$非负。`$L_{rw}$`和`$L_{sym}$`特征值$0$的度$k$为连通分量`$A_1,...,A_k$`的个数，`$L_{rw}$`特征值0对应的特征向量被指示向量`$\mathbb{1}_{A_1},...,\mathbb{1}_{A_k}$`张成。`$L_{sym}$`特征值0对应的特征向量被`$D^{1/2}\mathbb{1}_{A_1},...,D^{1/2}\mathbb{1}_{A_k}$`张成。  

## 谱聚类算法

### Unnormalized spectral clustering
输入：相似矩阵$S\in\mathbb{R}^{n\times n}$，构建的聚类个数$k$

* 用之前提到的方法构建相似图  
* 计算Laplacian矩阵$L$  
* 计算最小的k个特征值对应的特征向量  
* 将特征向量按列拼接成$V\in\mathbb{R}^{n\times k}$  
* 令`$y_i$`等于$V$的第$i$行  
* 对`$y_i$`进行$k$-means聚类  

输出：团`$A_1,...A_k$`，其中`$A_i=\{j|y_j\in C_i\}$`  

### Normalized spectral clustering according to Shi and Malik (2000)
输入：相似矩阵$S\in\mathbb{R}^{n\times n}$，构建的聚类个数$k$

* 用之前提到的方法构建相似图  
* 计算Laplacian矩阵$L$  
* **计算`$L_{rw}$`最小的k个特征值对应的特征向量(通过广义特征问题$Lv=\lambda Dv$求解)**  
* 将特征向量按列拼接成$V\in\mathbb{R}^{n\times k}$  
* 令`$y_i$`等于$V$的第$i$行  
* 对`$y_i$`进行$k$-means聚类  

输出：团`$A_1,...A_k$`，其中`$A_i=\{j|y_j\in C_i\}$`  

### Normalized spectral clustering according to Ng, Jordan, and Weiss (2002)
输入：相似矩阵$S\in\mathbb{R}^{n\times n}$，构建的聚类个数$k$

* 用之前提到的方法构建相似图  
* 计算normalized Laplacian矩阵**`$L_{sym}$`**  
* **计算`$L_{sym}$`最小的k个特征值对应的特征向量**  
* 将特征向量按列拼接成$V\in\mathbb{R}^{n\times k}$  
* **将V按行正则化，构建$U$，`$U_{ij}=V_{ij}/(\sum_kV_{ik}^2)^{1/2}$`**  
* 令`$y_i$`等于$U$的第$i$行  
* 对`$y_i$`进行$k$-means聚类  

输出：团`$A_1,...A_k$`，其中`$A_i=\{j|y_j\in C_i\}$`  

## Graph cut point of view
直觉上，我们希望找到一种分割，使得两部分之间的权重越小，两部分内的权重越大。  
对于$V$的互斥子集$A,B$，定义`$cut(A,B)=\sum_{i\in A,j\in B}w_{ij}$`。  
给定一个相似图，将$k$；类数据分割开最直接的方式是最小化`$cut(A_1,...,A_k):=\sum_{i=1}^kcut(A_i,\bar{A_i})$`。  
但实际上，这种方法往往会只分割成一个点和剩余部分，所以需要进行一定的正则化，两种常见的优化函数为：  
`$RatioCut(A_1,...,A_k)=\sum_{i=1}^k\frac{cut(A_i,\bar{A_i})}{|A_i|}$`  
`$Ncut(A_1,...,A_k)=\sum_{i=1}^k\frac{cut(A_i,\bar{A_i})}{vol(A_i)}$`  

#### RatioCut解决2分类问题
定义向量$f\in\mathbb{R}^n$
`$$f_i=
\begin{cases}
\sqrt{|\bar A|/|A|},  &if\ v_i \in A \\
-\sqrt{|\bar A|/|A|},  &if\ v_i \in \bar A
\end{cases}$$`

可以求得

1. `$f'Lf=2|V|RatioCut(A,\bar A)$`  
2. $f\cdot\mathbb{1}=0$  
3. $\Vert{f}\Vert^2=n$  

将`$f_i$`的取值条件松弛(不再只能取特定两个值)，可以通过以下优化问题求解`$f_i$`
`$$min_{f\in\mathbb{R}^n}f'Lf\quad s.t.\quad f\bot\mathbb{1}, \Vert{f}\Vert=\sqrt{n}$$`

最简单的标记`$v_i$`类别的方法为若`$f_i>0$`则将`$v_i$`标记为$A$类，否则标记为`$\bar A$`  
更好的方式可以将`$f_i$`进行一次$k$-means聚类，然后再根据`$f_i$`的类别标记`$v_i$`  

#### RatioCut解决$k$分类问题
定义$k$个指示向量`$h_i=(h_{1,i},...,h_{n,i})'$`  
`$$h_{i,j}=
\begin{cases}
1/\sqrt{|A_i|},  &if\ i \in A_j \\
0,  &otherwise
\end{cases}$$`

可以得到 `$h_i'Lh_i=2\frac{cut(A_i,\bar{A_i})}{|A_i|}$`  
令$H$为`$h_i$`按列拼接成的矩阵，则有

1. `$h_i'Lh_i=(H'LH)_{ii}$`  
2. $H'H=I$  

所以，
`$$RatioCut(A_1,...,A_k)=\frac{1}{2}\sum_{i=1}^kh_i'Lh=\frac{1}{2}\sum_{i=1}^k(H'LH)_{ii}=\frac{1}{2}Tr(H'LH)$$`

同样的将$H$的取值范围松弛成任意实数，可以通过以下优化问题求解$H$  
`$$min_{H\in\mathbb{R}^{n\times k}}\ Tr(H'LH)\quad s.t.\quad H'H=I$$`

实际上，$H$的解其实就可以理解成$L$的前$k$个特征向量。  

#### Ncut解决2分类问题
定义向量$f\in\mathbb{R}^n$  
`$$f_i=
\begin{cases}
\sqrt{\frac{vol(\bar A)}{vol(A)}},  &if\ i \in A \\
-\sqrt{\frac{vol(A)}{vol(\bar A)}},  &if\ i \in \bar A
\end{cases}$$`

可以得到

1. $(Df)'\cdot\mathbb{1}=0$  
2. $f'Df=val(V)$  
3. $f'Lf=2vol(V)Ncut(A,\bar A)$  

同样的，可以通过以下优化问题求解$f$  
`$$min_{f\in\mathbb{R}^n}f'Lf\quad s.t.\quad Df\bot\mathbb{1}, f'Df=vol(V)$$`

#### Ncut解决$k$分类问题
定义$k$个指示向量`$h_i=(h_{1,i},...,h_{n,i})'$`  
`$$h_{i,j}=
\begin{cases}
1/\sqrt{vol(A_i)},  &if\ i \in A_j \\
0,  &otherwise
\end{cases}$$`

可以得到 `$h_i'Lh_i=2\frac{cut(A_i,\bar{A_i})}{vol(A_i)}$`  
令$H$为`$h_i$`按列拼接成的矩阵，则有

1. `$h_i'Lh_i=(H'LH)_{ii}$`  
2. $H'DH=I$  

所以，
`$$Ncut(A_1,...,A_k)=\frac{1}{2}\sum_{i=1}^kh_i'Lh=\frac{1}{2}\sum_{i=1}^k(H'LH)_{ii}=\frac{1}{2}Tr(H'LH)$$`

将$H$替换成`$U=D^{1/2}H$`，同样$U$的取值范围松弛成任意实数，可以通过以下优化问题求解$U$  
`$$min_{U\in\mathbb{R}^{n\times k}}\ Tr(U'D^{-1/2}LD^{-1/2}U)\quad s.t.\quad U'U=I$$`

实际上，$U$的解其实就可以理解成$L_{sym}$的前$k$个特征向量。 

## Random walks point of view
随机游走也是一种解释谱聚类原理的一种方法。把相似矩阵看作是游走的转移概率，我们可以寻找一系列平稳分布，而这些分布被各个连通分量的平稳分布所张成。对于连通图，我们希望得到一种分割，使得在两类之间转移的概率最小。其实可以证明这个观点得到的结果与Ncut一致。  
定义转移矩阵$P$，`$p_{ij}:=w_{ij}/d_i$`，所以`$P=D^{-1}W$`  
定义随机游走的平稳分布`$\pi=(\pi_1,...,\pi_n)'$`，可以证明对于存在平稳分布的转移矩阵，`$\pi_i=d_i/vol(G)$`  
我们希望最小化从$A$转移到$\bar A$以及$\bar A$转移到$A$的概率，首先，对于联合分布
`$$P(X_0\in A,X_1\in B)=\sum_{i\in A,j\in B}P(X_0=i,X_1=j)=\sum_{i\in A,j\in B}\pi_ip_{ij}=\sum_{i\in A,j\in B}\frac{d_i}{vol(G)}\frac{w_{ij}}{d_i}=\frac{1}{vol(G)}\sum_{i\in A,j\in B}w_{ij}$$`

可以得到条件概率
`$$P(X_1\in B|X_0\in A)=\frac{P(X_0\in A,X_1\in B)}{P(X_0\in A)}=(\frac{1}{vol(G)}\sum_{i\in A,j\in B}w_{ij})(\frac{vol(G)}{vol(A)})=\frac{\sum_{i\in A,j\in B}w_{ij}}{vol(A)}$$`

可以观察到这个形式与Ncut一致。  

## References

Von Luxburg, Ulrike. "A tutorial on spectral clustering." Statistics and computing 17.4 (2007): 395-416.  

[Spectral clustering - Wikipedia](https://en.wikipedia.org/wiki/Spectral_clustering)  