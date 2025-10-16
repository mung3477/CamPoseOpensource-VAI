<p align="center">
  <h1 align="center">Do You Know Where Your Camera Is? View-Invariant Policy Learning with Camera Conditioning </h1>
  <p align="center">
    Tianchong Jiang<sup>1</sup>,
    Jingtian Ji<sup>1</sup>,
    Xiangshan Tan<sup>1</sup>,
    Jiading Fang<sup>2,*</sup>,
    Anand Bhattad<sup>3</sup>,
    Vitor Guizilini<sup>4,†</sup>,
    Matthew R. Walter<sup>1,†</sup>
  </p>
  <p align="center">
    <sup>1</sup>TTIC &nbsp; <sup>2</sup>Waymo &nbsp; <sup>3</sup>Johns Hopkins University &nbsp; <sup>4</sup>Toyota Research Institute
  </p>
  <p align="center">
    <a href='https://arxiv.org/abs/2510.02268'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://ripl.github.io/know_your_camera' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>
</p>

## Installation
First, clone the repo and cd into it.
```
git clone https://github.com/ripl/CamPoseOpensource
cd CamPoseOpensource
```

Then, run the setup script. It will setup the conda environment and download data.

```
bash setup.sh
```

Activate the conda environment with 
```
conda activate know_your_camera
```

If you only need ManiSkill or robosuite, comment out the lines to install other one.

## How to run
You can run training in robosuite with
```
python policy_robosuite/train.py
```
or in ManiSkill with
```
python policy_maniskill/train.py
```

## Plücker Snippet
To add camera conditioning to your policy, you can use the following minimalist snippet to get Plücker raymap from intrinsics and extrinsics. (It assumes OpenCV convention i.e. image origin at top-left, +z is forward.)
```
import torch

def get_plucker_raymap(K, c2w, height, width):
    """intrinsics (3,3), cam2world (4,4), height int, width int"""
    vv, uu = torch.meshgrid(
        torch.arange(height, device=K.device, dtype=K.dtype) + 0.5,
        torch.arange(width, device=K.device, dtype=K.dtype) + 0.5,
        indexing="ij",
    )
    rays = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)
    d_world = torch.nn.functional.normalize(
        (rays @ torch.linalg.inv(K).T) @ c2w[:3, :3].T,
        dim=-1,
        eps=1e-9,
    )
    o = c2w[:3, 3].view(1, 1, 3)
    m = torch.cross(o, d_world, dim=-1)
    return torch.cat([d_world, m], dim=-1)                         
```


## BibTeX
If you find this work useful, please cite:
```
@article{jiang2025knowyourcamera,
  title     = {Do You Know Where Your Camera Is? {V}iew-Invariant Policy Learning with Camera Conditioning},
  author    = {Tianchong Jiang and Jingtian Ji and Xiangshan Tan and Jiading Fang and Anand Bhattad and Vitor Guizilini and Matthew R. Walter},
  journal   = {arXiv preprint arXiv:2510.02268},
  year      = {2025},
}
```