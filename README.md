---
# Git操作完整指南

## 0.配置全局用户信息

首次使用 Git 前需设置用户名和邮箱：
```js
git config --global user.name "whmyb6666"
git config --global user.email whmyb6@gmail.com
```

## ‌1.初始化本地仓库‌

在目标目录中执行以下命令，使该目录成为 Git 管理的仓库：
```js
git init
```

‌## 2.添加目录名称到暂存区‌

若目录为空，需先创建一个占位文件（如 .gitkeep）以保留目录结构：
```js
git add README.md
```

查看本地库状态
```js
GIT status
```


‌## 3.提交目录结构到本地仓库

```js
git commit -m "first commit"
```

‌## 4.功能‌：设置当前分支的名称为 main。
‌
说明‌：在 Git 中，main 或 master 通常用作默认分支的名称。Git 2.28 版本后，默认分支名称从 master 更改为 main。
```js
git branch -M main
```

‌## 5.‌功能‌：添加一个远程仓库，并将其命名为 origin。
‌
说明‌：远程仓库是存储在另一台计算机上的 Git 仓库，通常用于与他人共享代码。origin 是远程仓库的默认名称。
```js
git remote add origin https://github.com/whmyb6666/mypython.git   (第一次运行需要)
```

‌## 6.推送目录到远程仓库

```js
git push -u origin main
```

## 7.修改代理：

```js
git config --global http.proxy http://127.0.0.1:7890
```