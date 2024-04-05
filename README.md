
# Deep learning network for shadow correction in remote sensing images

S. Luo, H. Li*, Y. Li, C. Shao, H. Shen and L. Zhang, "An Evolutionary Shadow Correction Network and a Benchmark UAV Dataset for Remote Sensing Images," IEEE Trans. on Geoscience and Remote Sensing, vol. 61, pp. 1-14, 2023, Art no. 5615414.

## Dataset and pre-trained model
Our UAV-SC dataset（including the refined version） and pre-trained model for shadow removal is available for download at Google Drive(Link: https://drive.google.com/drive/folders/1BAiUScnRZW5utx-zBZXAXVN3-sSdqIDq?usp=drive_link), in which UAV-SC is the same as the dataset we published previously (https://pan.baidu.com/s/1YiJ4SV_072Jj-L4up9CT-Q (password: vezt)) and UAV-SC_refine is the updated version.

## Requisites
- PyTorch 1.10
- Python 3.6

## Shadow Removal Network(SRNet)

You should download our dataset and put it into the folder of datasets first. 
#### Train
if you want to train the model by yourself, you need to run Combine_A_and_B.py to combine the shaded and unshaded image to get the train data.
```
python train.py --model pix2pix --name SRNet --dataset ./datasets/UAV-SC/train/train
```
#### Test
```
python test.py --model pix2pix --name SRNet --dataset ./datasets/UAV-SC/test/shaded
```


