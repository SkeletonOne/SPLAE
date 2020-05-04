# SPLAE
 Segement Prostate Like An Expert by Jeffrey Liu.
 Playing with the promise12 dataset.

Dice Score 0.7825447279440513

![Loss Graph](losss.png)
![Segment Result](segres.png)

# How to use SPLAE?

Place the data in the `file_path` folder, then run
```bash
python main.py
```
Easy enough!

Note that the original implementation(for brain tumor) made a threshold=0.5 for val/test, so the val score does not change initially(because the output are totally 0 after the threshold...). But after a few epochs it will become normal.

# Todo List
- [ ] [Data Augmentation]
- [ ] [Make a detailed description]
- [ ] [Support multi-class segmentation]
- [x] [Support more networks]

# Acknowlegement
This project is mainly based on [Brain Tumor Segmentation](https://github.com/sdsubhajitdas/Brain-Tumor-Segmentation) by Subhajit Das and [Unet-Segmentation-Pytorch-Nest-of-Unets](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets) by Malav Bateriwala. Appreciate for their great work.