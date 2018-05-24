---
title: PyCharm远程调用matplotlib绘制
tags: ["PyCharm", "matplotlib"]
date: 2018-05-24
---

## 背景
在机器学习开发中，我们往往使用GPU服务器来运行代码，PyCharm是一个很方便的远程开发和调试的IDE。开发过程中，我们往往需要调用matplotlib来绘制图形，而直接调用会报"no display name and no $DISPLAY environment variable"的错误。  

## 解决方案

### 所需软件

1. SSH客户端，例如[Putty](https://www.putty.org/)或[Xshell](https://www.netsarang.com/products/xsh_overview.html)  
2. X11 display server，例如[Xming](https://sourceforge.net/projects/xming/)  

### 所需配置

1. SSH客户端开启X11转发功能，例如Putty配置如下：
![](/media/posts/putty-x11.png)  
连接后输入echo $DISPLAY，记录输出，如localhost:10.0。
2. Xming安装完成即可，无需配置。  
3. PyCharm在run->edit configurations->在environment variables中添加DISPLAY = localhost:10.0。

### Reference
[PyCharm远程开发配置-Yinzm's blog](https://www.cnblogs.com/yinzm/p/8251118.html)  
[Python plotting on remote server using PyCharm
-stackoverflow](https://stackoverflow.com/questions/39803373/python-plotting-on-remote-server-using-pycharm?newreg=e4a345202f334e50b4d17be5684ad7a8)
