---
title: vutrl/SSR
date: 2019-03-07
categories: [Others]
tags: [Others]
---

## SSR配置所需命令
<!-- more -->
### 配置SSR
先ssh去到刚开的主机
```
ssh root@host_ip
```
执行这三条
```
wget --no-check-certificate https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks.sh

chmod +x shadowsocks.sh

./shadowsocks.sh 2>&1 | tee shadowsocks.log
```
输入密码，再输入端口，然后按照提示操作，等待部署完成就会出现VPS的信息。

### 更换centOS内核，使用锐速。
CentOS6 内核更换为： 2.6.32-504.3.3.el6.x86_64
```
rpm -ivh http://soft.91yun.pw/ISO/Linux/CentOS/kernel/kernel-firmware-2.6.32-504.3.3.el6.noarch.rpm

rpm -ivh http://soft.91yun.pw/ISO/Linux/CentOS/kernel/kernel-2.6.32-504.3.3.el6.x86_64.rpm --force

```
CentOS7 内核更换为： 3.10.0-229.1.2.el7.x86_64
```
rpm -ivh http://soft.91yun.pw/ISO/Linux/CentOS/kernel/kernel-3.10.0-229.1.2.el7.x86_64.rpm --force

```
查看内核是否安装成功
```
	
rpm -qa | grep kernel
```

### TCP Fast Open为了更好的连接
使用nano这个编辑器打开一个文件
```
nano /etc/rc.local
```
用方向键把光标移到最末端，粘贴下面这一行内容，然后按 Ctrl + X 退出。输入“y”并回车确认退出。
```
echo 3 > /proc/sys/net/ipv4/tcp_fastopen
```
打开sysctl.conf文件
```
nano /etc/sysctl.conf
```
最后一行添加：
```
net.ipv4.tcp_fastopen = 3
```
打开shadowsocks.json
```
nano /etc/shadowsocks.json
```
把其中 "fast_open" 一项的 false 替换成 true.
保存退出,最后，输入以下命令重启 Shadowsocks。
```
/etc/init.d/shadowsocks restart
```
### 开启锐速
重启VPS后先ssh到主机，再输入一下指令
```
wget -N --no-check-certificate https://raw.githubusercontent.com/91yun/serverspeeder/master/serverspeeder-all.sh && bash serverspeeder-all.sh
```
安装完成后，输入以下命令打开配置文件。
```
nano /serverspeeder/etc/config
```
将 advinacc 的 0 改为 1，保存并退出。(其实我没做这一步也是速度很快的了)