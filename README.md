# Ai更换表情的H5应用

## 部署

1. 安装各种环境依赖
```bash
# 安装之前记得换源，建议清华源或者中科大源
sudo pip install ninja
sudo pip install dlib
sudo pip install cmake # 安装dlib貌似需要cmake，按照报错的提示安装
sudo pip uninstall -y torch
sudo pip uninstall -y torchvision
sudo pip install torch==1.6.0
sudo pip install torchvision==0.7.0
sudo pip install treamlit
# 这里可能遇到一些cuda版本的问题
```
2. 下载代码
   ```bash
   mkdir ~/H5APP && cd ~/H5APP 
   wget https://obs-aigallery-zc.obs.cn-north-4.myhuaweicloud.com/clf/code/HFGI.zip
   sudo apt install unzip # 以ubuntu等apt包管理系为例
   unzip HFGI.zip
   git clone https://github.com/wtyqqq/AIChangeFaceH5.git
   mv AIChangeFaceH5/changeFace.py HFGI/ #移动python文件到工作目录
   mkdir HFGI/saveFiles && cd HFGI #创建临时文件夹并进入工作目录
   ```

3. 运行

   ```bash
   streamlit run AIChangeFaceH5
   # 记得开防火墙
   ```

   
