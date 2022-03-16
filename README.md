# Improving Image Paragraph Captioning with Dual Relations(ICME 2022)

Source code for "Improving Image Paragraph Captioning with Dual Relations", published in ICME 2022.

In this paper, we propose a novel model (i.e., DualRel) to capture  spatial and semantic relations among objects for Image paragraph captioning.

<img src="./imgs/framework.png" alt="DualRel-framework"  />


## Tips
- Our model implementation based on [ruotianluo's](https://github.com/ruotianluo/ImageCaptioning.pytorch).
- Role of each folder and detailed uasage can be find in ruotianluo's code-base.
- Our model's main code is in file `./captioning/models/XDualRelation.py` and `./captioning/models/XRelation.py`.
- anaconda based python environment can be find in file `code.yaml`.


## Preparation
- Images features extracted from [pretrained Faster R-CNN](https://github.com/airsplay/lxmert).
- Download 'relationships_500_500_200.h5'(384.91MB) from baidu disk, and put it into folder `./data`.
```
链接(link)：https://pan.baidu.com/s/1nCUsgorvwU9Gau6zbf78tw 
提取码(code)：fnbg
```
- Others datas in folder `./data`.
- Put evaluation code in folder `./cider` and `./coco-caption`, install from `.gitmodules`.


## Usage
### Train
- Stage1: Cross Entropy training
```
conda activate python_env
python tools/train.py --cfg configs/para/x2_xe.yml
```
- Stage2: SCST training
```
conda activate python_env
python tools/train.py --cfg configs/para/x2_scst.yml
```


## Citation
If you find our paper useful, please consider cite the following paper.

```
ToDo
```