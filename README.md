# SPLAE
 Segement Prostate Like An Expert by Jeffrey Liu.
 Playing with the promise12 dataset.

# Results

All models were trained with BS = 8, Epochs = 60 Lr = 1e-4.

Model | Aug | Dice Score | Loss
--- |:---:|:---:|:---:
U-Net | Rot | 0.7649910148678267 | BCE
U-Net | Rot+HFlip+VFlip | 0.7820317164718084 | BCE

![U-Net Loss Graph](README_IMGS/TrainValLoss.png)
![U-Net Segment Result](README_IMGS/Predictions.png)
![U-Net Segment Result2](README_IMGS/Predictions6.png)

# How to use SPLAE?

Place the data in the `file_path` folder, then run
```bash
python main.py
```
Easy enough!

The hyperparameters are in `main.py`. Change them to whatever you want.

Note that the original implementation(for brain tumor) made a threshold=0.5 for val/test, so the val score does not change initially(because the output are totally 0 after the threshold...). But after a few epochs it will become normal.

# Todo List
- [x] [Data Augmentation]
- [x] [Make an instruction]
- [ ] [Support 5-fold validation]
- [ ] [Support multi-class segmentation]
- [x] [Support more networks]
- [ ] [Transform the output to 3D slices for submission]

# Acknowlegement
This project is mainly based on [Brain Tumor Segmentation](https://github.com/sdsubhajitdas/Brain-Tumor-Segmentation) by Subhajit Das and [Unet-Segmentation-Pytorch-Nest-of-Unets](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets) by Malav Bateriwala. Appreciate for their great work.