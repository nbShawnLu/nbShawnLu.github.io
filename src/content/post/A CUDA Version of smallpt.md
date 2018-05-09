---
title: A CUDA Version of smallpt
tags: ["CUDA", "smallpt", "Computer Graphics"]
date: 2018-04-22
---

作为一个机器学习方向的研究生，把第一篇技术博客献给图形学也是一个很奇妙的事情。  
这原本是一个GPU并行编程的课程设计，选择了对[smallpt](http://www.kevinbeason.com/smallpt/)光线追踪进行CUDA移植，与好友[Bingo](http://bentleyblanks.github.io/)合作完成并进行了一定的优化。代码已开源至[smallptCuda](https://github.com/BentleyBlanks/smallptCuda)  

![](/media/posts/smallpt_cuda.jpg)

> 运行结果  
> Cuda部分的入门教程较多，可以参考[CUDA C/C++ Basics](http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf)以及[An Even Easier Introduction to CUDA](https://devblogs.nvidia.com/even-easier-introduction-cuda/)  

## Benchmark
|   | GTX1080Ti | Intel Xeon E5 (6C12T) 2.80GHz |
| :-----: | :-----: | :----: |
| Resolution| 1024*768 | 1024*768 |
| SPP | 5000 | 5000 |
|  Cost Time  | 4.3s | 32.1min |

|   | GTX750 | Intel Xeon E5 (8C16T) 2.40GHz |
| :-----: | :-----: | :----: |
| Resolution| 768*768 | 768*768 |
| SPP | 2048 | 2048 |
| Cost Time | 19.0s | 7.2min |

## Usage
### Linux
    $ git clone https://github.com/BentleyBlanks/smallptCuda.git 
    $ cd smallptCuda 
    $ git checkout release3 
    $ cd src && make 
    $ ./smallpt 5000 
    $ display test.png 

## 开发与优化杂记
### GPU硬件结构
#### SP(streaming processor)
也被称为CUDA core，对应着一个线程的处理。  
每个SP包含各自的ALU和Register。  
#### SM(streaming multiprocessor)
一个SM包含多个SP，每个SM包含的SP数量依据GPU架构而不同，Fermi架构GF100是32个，GF10X是48个，Kepler架构都是192个，Maxwell都是128个。  
一个SM还包含控制单元和Shared Memory，可供该SM中的SP访问。  
#### GPU
一个GPU包含多个SM，GPU中包含Global Memory，可供所有SM和SP访问，但延迟相比Shared Memory更高。  
### CUDA程序软件结构
#### thread
一个CUDA程序包含多个threads。  
#### block
数个threads会被群组成一个block，一个block中的线程会被分配到同一个SM中处理，同一个block中的threads可以同步，也可以通过shared memory通信。  
#### grid
多个blocks则会再构成grid。 
#### warp
GPU执行程序时的调度单位，目前cuda的warp的大小为32，同在一个warp的线程，以不同数据资源执行相同的指令。不足32的会以32个线程打包运行。  
