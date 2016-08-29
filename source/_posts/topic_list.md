---
title: How to
date: 2016-07-04 20:15:27
tags:
---

可参考[hexo官网](https://hexo.io/zh-cn/docs/)
# 安装
首先请安装 *node.js*， *git*， *npm*
`sudo apt-get install git npm nodejs nodejs-legacy`

然后clone项目代码
`git clone git@github.com:transwarpio/teaching_ml.git`

进入*teaching_ml*目录，然后安装相应npm包
    ``` bash
    cd teaching_ml
    npm install
    ```
`npm install`会安装teaching\_ml/package.json中的包。

# 写作
`hexo new post [title]`会在`source/_posts/`下建立`[title].md`文件，然后你就可以开始写作了
markdown的具体语法可以参考：[master markdown](https://guides.github.com/features/mastering-markdown/)
  
## 图片
对于资源文件比如图片，请放到`source/images`文件夹下。然后在文章中引用时使用相对路径。比如`source/images/pic.jpg`。
引用时为`images/pic.jpg`

## 公式
可以使用latex编辑公式，或使用公式编辑器编辑后转成图片展示。
对于latex公式，使用美元符号`$`来包含inline公式，比如`$ y = x + b $`会显式为$ y = x + b $。
而双美元符号来显式整行公式，比如`$$ y = x + b $$`,显式为 
$$
y = x + b
$$ 

# 部署
当你写完后，可以使用`hexo server`命令在本地查看效果。
满意后使用`hexo generate --deploy`部署到github。

