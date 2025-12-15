# CRCME

CRCME is a pathology-enhanced CT foundation model designed to capture microscopic tumor biology from routine CT images. It provides a robust representation for colorectal cancer diagnosis, staging, and risk assessment, enabling comprehensive tumor profiling without invasive procedures.


This repository provides:

• resources to reproduce and extend CRCME for CRC or related imaging research,

• tools for developing end-to-end models for non-invasive tumor characterization,

• a pretrained CT foundation model for downstream colorectal cancer tasks.


## 1. Environmental preparation and quick start

First clone the repo and cd into the directory:

```
git clone https://github.com/Yjing07/CRCME.git
cd CRCME
```

Using requirements.txt

• Install the required Python packages using pip: `pip install -r requirements.txt`

Using environment.yml

• Install the required Python packages using pip: `conda env create -f environment.yml`

Activate the Conda environment

• Then create a conda env and install the dependencies: `conda activate my_python_env`

## 2. Preparing and loading the model

Download the [pre-trained weight](https://drive.google.com/file/d/1zT6FsCh7RubL_k0LMGQwjhRMV90sprch/view?usp=sharing) first!

First create the ```checkpoints directory``` inside the root of the repo:

```
mkdir -r checkpoints
```

## 3. Data preparation

Before inputting the model, CT images need to be preprocessed. We set all image Spaceing to ```(1.0, 1.0, 3.0)```, and all image sizes to```(32, 256, 256)```. For histopathology images, we extract patches from whole-slide images (WSIs) and resize them to `(256, 256)` pixels. These patches are normalized and used to guide CT feature learning in the pathology-enhanced representation module.

> During pretraining, the image dataset is saved as a JSON file.

> For downstream tasks, the organization of labels is as follows:
  ```
  {
    "image_name1":
        {"tnm.t": 1, "cT": 1, "tnm.n": 0, "cN": 1, "tnm.m": 0, "cM": 0, "tnm.tnm": 1, "cTNM": 2, "msi": 0, "cms":1, "os.event": 0, "os.delay": 85.15, "dfs.event": 0, "dfs.delay": 84.99}
    "image_name2":
    ...
  }
  ```
  The data splits for five-fold cross-validation are saved as:
  ```
   {
    "Fold_0":
        {
        "training":[image_name1, image_name2, ...],
        "validation":[image_name3, image_name4, ...]
        },
    "Fold_1":
        ....
   
  }
  ```

## 4. Fine-tuning in your datasets
Download the pre-trained weight from [Google Drive](https://drive.google.com/file/d/1zT6FsCh7RubL_k0LMGQwjhRMV90sprch/view?usp=sharing) and specify _weight directory_ during training.

### Fine-tuning using CT images on one task

After data preprocessing, prepare the label file according to the above structure and set the label path to start training.

```
python ./src/train_MOE.py --model_name mae --pretrained_ct checkpoints/checkpoint-ct.pth --pretrained_joint checkpoints/checkpoint-joint.pth  --epochs 200 --batch_size 8 --lr 0.0001 --shape (32,256,256) -- data_path ${IMAGE_DIR}$ --label_path $LABELS DIR$ --log_path ./logs
```

## Basic Usage: CRCME as a Vision Encoder
 ### 1. Load the CRCME
   ```
   import torch
   import os
   from lib.model import ViT

   model = ViT("CRCFound_large_patch16_256")
   pretrained_path = 'checkpoints/checkpoint-ct.pth'  # TODO
   if os.path.isfile(pretrained_path):
     print(f"Loading pretrained weights from: {pretrained_path}")
     checkpoint = torch.load(pretrained_path, map_location='cpu')
     pretrained_weights = checkpoint.get('model', checkpoint) 
     cleaned_weights = {k.replace("module.", ""): v for k, v in pretrained_weights.items()}
     compatible_weights = {k: v for k, v in cleaned_weights.items() if k in model.state_dict()}
     model.load_state_dict(
          {**model.state_dict(), **compatible_weights}
      )
     print("Pretrained weights loaded successfully.")
   else:
     raise FileNotFoundError(f"Pretrained model not found at: {pretrained_path}")
  model.to(device="cuda", dtype=torch.float16)
  model.eval()
   ```
### 2. Encode images with CRCME
   ```
   import torch
   import SimpleITK as sitk
   import numpy as np

   img_root = '' # TODO
   img = sitk.ReadImage(img_root)
   img = torch.from_numpy(sitk.GetArrayFromImage(img).astype(np.float32))
   img_tensor = img.unsqueeze(0).unsqueeze(0)
   with torch.inference_mode():
     image_embeddings = model(
          image=img_tensor.to("cuda"),
          feature=True    # return global feature token
      )[0]  # shape: [1, feature_dim]
   ```

## Optional Usage: CRCME as a Clinical Expert

After data preprocessing and fine-tuning, CRCME can be applied as a clinical expert model to predict multiple clinical endpoints from CT images.

```
python ./src/train_MOE.py --model_name mae --pretrained_ct checkpoints/checkpoint-ct.pth --pretrained_joint checkpoints/checkpoint-joint.pth  --epochs 200 --batch_size 8 --lr 0.0001 --shape (32,256,256) -- data_path ${IMAGE_DIR}$ --label_path $LABELS DIR$ --log_path ./logs
```

## References

This project builds upon and is inspired by the following works:

- [PyTorch](https://pytorch.org/) – the primary deep learning framework used for model development.
- [CLAM](https://github.com/mahmoodlab/CLAM.git) – utilized for preprocessing whole-slide histopathology images.
- [ViT-PyTorch](https://github.com/lucidrains/vit-pytorch) – implementation of the Vision Transformer (ViT) architecture.
- [MAE](https://github.com/facebookresearch/mae.git) – employed for self-supervised learning components.
