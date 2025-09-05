# MonoCLUE: Object-aware Clustering Enhances Monocular 3D Object Detection

## Introduction
This repository provides the official implementation of [MonoCLUE: Object-aware Clustering Enhances Monocular 3D Object Detection](https://arxiv.org) based on the excellent work [MonoDGP](https://github.com/PuFanqi23/MonoDGP). In this work, we propose a DETR-based monocular 3D detection framework that strengthens visual reasoning by leveraging clustering and scene memory, enabling robust performance under occlusion and limited visibility.
<div align="center"> <img src="figures/overall_architecture.png" width="600" height="auto"/> </div> 

<div align="center"> <img src="figures/explanation.png" width="600" height="auto"/> </div>

## Main Result

Note that the randomness of training for monocular detection would cause a variance of ±1 AP<sub>3D|R40</sub> on KITTI.

The official results :
<table>
    <tr>
        <td rowspan="2",div align="center">Models</td>
        <td colspan="3",div align="center">Val, AP<sub>3D|R40</sub></td>   
        <td rowspan="2",div align="center">Logs</td>
        <td rowspan="2",div align="center">Ckpts</td>
    </tr>
    <tr>
        <td div align="center">Easy</td> 
        <td div align="center">Mod.</td> 
        <td div align="center">Hard</td> 
    </tr>
    <tr>
        <td rowspan="4",div align="center">MonoCLUE</td>
        <td div align="center">33.7426%</td> 
        <td div align="center">24.1090%</td> 
        <td div align="center">20.5883%</td> 
        <td div align="center"><a href="https://drive.google.com/file/d/1ccwmKmxjJMtiD5GAYMlB9Acz_sV2gtwJ/view?usp=sharing">log</a></td>
        <td div align="center"><a href="https://drive.google.com/file/d/1Nddzx3xDE0DPZzVluR9HEYRgH2wALU9z/view?usp=sharing">ckpt</a></td>
    </tr>  
  <tr>
        <td div align="center">31.5802%</td> 
        <td div align="center">23.5648%</td> 
        <td div align="center">20.2746%</td> 
        <td div align="center"><a href="https://drive.google.com/file/d/1mjk457aBjxs6a3Lf-biX10_YzhW2th_U/view?usp=sharing">log</a></td>
        <td div align="center"><a href="https://drive.google.com/file/d/1eCON928oVFTL2U64qZotWYhRCRopldxY/view?usp=sharing">ckpt</a></td>
    </tr>  
</table>

## Installation
1. Clone this project and create a conda environment:
    ```
    git clone https://github.com/ZrrSkywalker/MonoDETR.git
    cd MonoDETR

    conda create -n monodetr python=3.8
    conda activate monodetr
    ```
    
2. Install pytorch and torchvision matching your CUDA version:
    ```bash
    conda install pytorch torchvision cudatoolkit
    # We adopt torch 1.9.0+cu111
    ```
    
3. Install requirements and compile the deformable attention:
    ```
    pip install -r requirements.txt

    cd lib/models/monodetr/ops/
    bash make.sh
    
    cd ../../../..
    ```
    
4. Make dictionary for saving training losses:
    ```
    mkdir logs
    ```
 
5. Download [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) datasets and prepare the directory structure as:
    ```
    │MonoDETR/
    ├──...
    ├──data/KITTIDataset/
    │   ├──ImageSets/
    │   ├──training/
    │   ├──testing/
    ├──...
    ```
    You can also change the data path at "dataset/root_dir" in `configs/monodetr.yaml`.
    
## Get Started

### Train
You can modify the settings of models and training in `configs/monodetr.yaml` and indicate the GPU in `train.sh`:

    bash train.sh configs/monodetr.yaml > logs/monodetr.log
   
### Test
The best checkpoint will be evaluated as default. You can change it at "tester/checkpoint" in `configs/monodetr.yaml`:

    bash test.sh configs/monodetr.yaml


## Acknowlegment
This repo benefits from the excellent [MonoDETR](https://github.com/ZrrSkywalker/MonoDETR) / [MonoDGP](https://github.com/PuFanqi23/MonoDGPand).
