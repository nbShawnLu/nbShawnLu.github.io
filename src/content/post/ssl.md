---
title: 三分钟为你的博客部署SSL证书
tags: ["HTTPS", "SSL", "page"]
date: 2018-05-04
---

## 前言

这篇教程需要有站点服务器的Shell权限和Root权限。  

## SSL

SSL证书就是遵守SSL协议，由受信任的数字证书颁发机构CA，在验证服务器身份后颁发，具有服务器身份验证和数据传输加密功能。  
HTTPS即HTTP下加入SSL层，一方面可以通过证书确认网站的真实性，另一方面可以保证数据传输安全，避免网页被篡改~~，还可以使自己的博客看起来更正规有逼格~~。  
SSL证书按大类一般可分为 DV SSL, OV SSL, EV SSL证书，也叫做域名型、企业型、增强型证书。  
![](/media/posts/ssl.jpg)  


## Let's Encrypt

[Let's Encrypt](https://letsencrypt.org/)是一个叫[ISRG](https://letsencrypt.org/isrg/)(Internet Security Research Group)的组织推出的免费安全证书计划，提供了免费的DV SSL单域名证书，又于2018年推出ACME v2泛域名证书功能，有兴趣的小伙伴可以尝试一下。  

## Certbot

[Certbot](https://certbot.eff.org/)是一个SSL/TLS自动部署工具，由EFF开发，能够自动获取Let's Encrypt的证书和部署在站点服务器上。  

## Usage

[Certbot](https://certbot.eff.org/)网站给出了很无脑的部署教程，大家选择了自己的站点服务和系统发行版后会自动生成对应的教程。  
以Nginx on CentOS 7 为例，大概可以概括为：  

### 1.启用EPEL repository
    $ yum -y install yum-utils
    $ yum-config-manager --enable rhui-REGION-rhel-server-extras rhui-REGION-rhel-server-optional


### 2.安装Certbot
    $ sudo yum install certbot-nginx


### 3.获取并安装证书
    $ sudo certbot --nginx

> 期间会提示你输入邮箱，用于发送证书续期邮件。此外如果你的站点服务器如果监听了多个域名，这一步会提示你选择需要部署SSL的站点，或者输入空格表示全部。  

### 4.由于Let's Encrypt的证书只有90天有效期，你需要在失效前更新证书。首先你需要测试自动续期功能是否可以工作
    $ sudo certbot renew --dry-run

### 5.如果上一步正常的话，你可以定时调用certbot renew来检测是否需要续期并自动续期。例如你可以利用以下指令创建一个定时检测线程。
    $ 0 0,12 * * * python -c 'import random; import time; time.sleep(random.random() * 3600)' && certbot renew

好了至此你应该可以通过https打开你的站点了，如果你需要更灵活的部署方式，你可以阅读Certbot的[文档](https://certbot.eff.org/docs/)来自定义部署。  
