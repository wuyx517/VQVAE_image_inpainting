# VQVAE_image_inpainting

## 使用tensorflow 2.0 复现  VQVAE 图像修复

原论文地址：https://arxiv.org/abs/2103.10022

论文开源tensorflow 1.0 代码地址：https://github.com/USTC-JialunPeng/Diverse-Structure-Inpainting

### 运行环境：

- Ubunt 20.04
- tensorflow 2.3
- python 3.8

### 训练步骤

配置文件位于**config**文件夹中

修改配置文件 data_config.yaml 中的 train_flist 和 valid_flist 即可运行。

train_flist 和 valid_flist 的文件格式为：

```
/path/***.JPEG
/path/***.JPEG
/path/***.JPEG
/path/***.JPEG
```

训练依次过程如下：

```shell
python train_vqvae.py
python structure_generator.py
python texture_generator.py
```

ps: 目前 texture_generator.py 尚有一些问题未进行修复。

### 留言

由于复现时间较为匆忙，有很多细节性的东西尚未修复，而且 **Inference Pipeline** 也尚未完成（定个小目标，2021.8.15前全部完成）。

小弟恳请各位大佬可以多多指点，给予建议！若有些地方写的差的看不下去，也不要压制情绪，直接开喷吧！小弟受得住。



<img src="D:\Dorport\image inpainting\VQVAE_image_inpainting\image\image-20210809024628423.png" alt="image-20210809024628423" style="zoom:500%;" />

