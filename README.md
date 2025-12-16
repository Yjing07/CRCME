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

### Step 1: Create the checkpoints directory
```
mkdir -r checkpoints
```
### Step 2: Download pre-trained weights
Download the following pre-trained weights and place them into the checkpoints/ directory:

• [**CT pre-trained weights**](https://drive.google.com/file/d/1hdMLMW8qA4mIGA4FjPsFg4YiwoCAIp9Q/view?usp=drive_link)

• [**Joint pre-trained weights**](https://drive.google.com/file/d/1iaKr8P0Qyi96iyac6otyh2t0lsZB7iq8/view?usp=drive_link)

Directory structure:

```
checkpoints/
├── ct_pretrained.pth
└── joint_pretrained.pth
```

## 3. Data preparation

Before feeding data into the model, CT images need to be preprocessed. Specifically, all CT images are **resampled** to a spacing of ```(1.0, 1.0, 3.0)``` and **resized** to ```(32, 256, 256)``` voxels.

For histopathology images, we extract patches from whole-slide images (WSIs) and resize them to ```(256, 256)``` pixels. These patches are **normalized** and used to guide CT feature learning within the pathology-enhanced representation module.

> During pretraining, the image dataset is saved as a JSON file. For downstream tasks, the organization of labels is as follows:
  ```
  {
    "image_name1":
        {"tnm.t": 1, "cT": 1, "tnm.n": 0, "cN": 1, "tnm.m": 0, "cM": 0, "tnm.tnm": 1, "cTNM": 2, "msi": 0, "cms":1, "os.event": 0, "os.delay": 85.15, "dfs.event": 0, "dfs.delay": 84.99}
    "image_name2":
    ...
  }
  ```
  > The data splits for five-fold cross-validation are saved as:
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

### Fine-tuning using CT images on one task

After preprocessing your data, prepare the label file following the structure described above, and set the appropriate label path to start training.

```
python ./src/train_MOE.py \
    --model_name mae \
    --pretrained_ct checkpoints/checkpoint-ct.pth \
    --pretrained_joint checkpoints/checkpoint-joint.pth \
    --epochs 200 \
    --batch_size 8 \
    --lr 0.0001 \
    --shape (32,256,256) \
    --data_path ${IMAGE_DIR} \
    --label_path ${LABELS_DIR} \
    --log_path ./logs
```
**Notes:**

• Replace ```${IMAGE_DIR}``` and ```${LABELS_DIR}``` with the paths to your CT images and label files.

• Adjust ```--epochs```, ```--batch_size```, and ```--lr``` according to your dataset size and GPU memory.

• Logs and checkpoints will be saved under ```./logs``` by default.


## Basic Usage: CRCME as a Vision Encoder

We provide a Jupyter Notebook demo that demonstrates how to extract features from CT images using CRCME. You can run it interactively to see step-by-step how the model is loaded and used.

**1. Open the Demo Notebook**

• File: [scr/CRCME_feature_extraction_demo.ipynb](src/demo.ipynb)

• This notebook includes:
  
    (1) Loading the pre-trained CT and joint/FU weights
    (2) Encoding a single CT image
    (3) Batch encoding multiple images
    (4) Optionally saving features for downstream tasks

**2. Quick Example (from the notebook)**
```
import torch
from lib.model_MOE import ViT_ct, ViT_fu, FusionModel, FusionPipeline

# Initialize models
model_a = ViT_ct("CRCFound_large_patch16_256")
model_b = ViT_fu("CRCFound_large_patch16_256")
fusion_model = FusionModel(input_dim_a=1024, input_dim_b=1024, classes=2)
pipeline = FusionPipeline(model_a, model_b, fusion_model, num_classes=2)

# Load pretrained weights
pipeline.load_pretrained(ct_path="checkpoints/checkpoint-ct.pth",
                         fu_path="checkpoints/checkpoint-joint.pth")

pipeline.to("cuda").eval()

# Encode a CT image
img_tensor = ... # preprocessed CT image tensor, shape [1,1,D,H,W]
with torch.inference_mode():
    features = pipeline(img_tensor.to("cuda"), return_features=True)

print("Feature shape:", features.shape)
```

## Optional Usage: CRCME as a Clinical Expert

After preprocessing the data and fine-tuning the model, CRCME can be used as a **clinical expert** to predict multiple clinical endpoints from CT images.

```
python ./src/eval_report_generation.py \
    --model_name mae \
    --pretrained_root ./checkpoints/downrasks \
    --img_root ${IMAGE_DIR} \
    --lr 0.0001 \
    --shape (32,256,256) \
    --Patient_ID $Patient_ID \
    --Patient_name $Patient_name \
    --Patient_age $Patient_age \
    --Patient_sex $Patient_sex \
    --log_path ./logs/report

```

## References
This project builds upon and is inspired by the following works:

- [PyTorch](https://pytorch.org/) – the primary deep learning framework used for model development.
- [CLAM](https://github.com/mahmoodlab/CLAM.git) – utilized for preprocessing whole-slide histopathology images.
- [ViT-PyTorch](https://github.com/lucidrains/vit-pytorch) – implementation of the Vision Transformer (ViT) architecture.
- [MAE](https://github.com/facebookresearch/mae.git) – employed for self-supervised learning components.
