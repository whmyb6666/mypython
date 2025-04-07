## ‌1. 初始化本地仓库‌
在目标目录中执行以下命令，使该目录成为 Git 管理的仓库：
# git init

‌## 2. 添加目录名称到暂存区‌
若目录为空，需先创建一个占位文件（如 .gitkeep）以保留目录结构：
# git add README.md


# GIT status

‌## 3. 提交目录结构到本地仓库

# git commit -m "first commit"


‌## 4功能‌：设置当前分支的名称为 main。
‌说明‌：在 Git 中，main 或 master 通常用作默认分支的名称。Git 2.28 版本后，默认分支名称从 master 更改为 main。
# git branch -M main

‌## 5. ‌功能‌：添加一个远程仓库，并将其命名为 origin。
‌说明‌：远程仓库是存储在另一台计算机上的 Git 仓库，通常用于与他人共享代码。origin 是远程仓库的默认名称。
# git remote add origin https://github.com/whmyb6666/mypython.git   (第一次运行需要)

‌## 6. 推送目录到远程仓库
# git push -u origin main

