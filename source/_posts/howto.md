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

##### 你可能需要在/usr/lib/中安装emacs, 在linux下使用`sudo apt-get install emacs`进行安装。

# 写作
`hexo new post [title]`会在`source/_posts/`下建立`[title].md`文件，然后你就可以开始写作了
markdown的具体语法可以参考：[master markdown](https://guides.github.com/features/mastering-markdown/)

## 图片
对于资源文件比如图片，在`source/_posts/`文件夹下创建与文章同名的子文件夹。比如`source/_posts/doc.md` ，则有 `source/_posts/doc`文件夹。
然后在文章中引用时使用相对路径。比如`source/_posts/doc/pic.jpg`。
引用时为`doc/pic.jpg`
 
## 公式
可以使用latex编辑公式，或使用公式编辑器编辑后转成图片展示。
对于latex公式，使用美元符号`$`来包含inline公式，比如`$ y = x + b $`会显式为$ y = x + b $。
而双美元符号来显式整行公式，比如`$$ y = x + b $$`,显式为 
$$
y = x + b
$$ 

## 错误
由于hexo中'{{'以及'}}'被reserve, `hexo server|generate`的时候如果source/_posts文档中有相关字符将会报错。
如果'{{'和'}}'不可避免, 请使用'{% raw %}'圈起有歧义的段落, 如下。
```
{% raw %}
$\large \sigma(z) = {1 \over 1 + e^{-z}}$,
{% endraw %}
```
若发生图片显示失败的问题，参考以下解决方案：

由于hexo自带的hexo-asset-image插件中调用图片链接缺失github库的部分，故piggy在/teaching_ml/node_modules/hexo-asset-image/index.js的一行代码 $(this).attr('src', '/__teaching_ml__/' + link + src)加入的黑体字部分为库的名称，blog网页中的图片方能正常显示。


# 部署
当你写完后，可以使用`hexo server`命令在本地查看效果。
满意后使用

git add .

git commit -m "备注“

git push origin master

hexo generate --deploy

部署到github。

