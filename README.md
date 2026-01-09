# Introduction

[ArXiv](https://arxiv.org/abs/2512.16449) || [Project page](https://amm.aass.oru.se/shape-completion-grasping/)

This repository contains code for the paper *Single-View Shape Completion for Robotic Grasping in Clutter* presented at [RiTA 2025](https://icrita.org/).

Paper abstract:
> In vision-based robot manipulation, a single camera view can only capture one side of objects of interest, with additional occlusions in cluttered scenes further restricting visibility. As a result, the observed geometry is incomplete, and grasp estimation algorithms perform suboptimally. To address this limitation, we leverage diffusion models to perform category-level 3D shape completion from partial depth observations obtained from a single view, reconstructing complete object geometries to provide richer context for grasp planning. Our method focuses on common household items with diverse geometries, generating full 3D shapes that serve as input to downstream grasp inference networks. Unlike prior work, which primarily considers isolated objects or minimal clutter, we evaluate shape completion and grasping in realistic clutter scenarios with household objects. In preliminary evaluations on a cluttered scene, our approach consistently results in better grasp success rates than a naive baseline without shape completion by 23\% and over a recent state of the art shape completion approach by 19\%. Our code is available at https://amm.aass.oru.se/shape-completion-grasping/.

# Setup and running inference

## Setup
Clone this repository and `cd` into it. Download checkpoint with:
```bash
wget -O shape_comp.ckpt "https://cloud.oru.se/s/KKexxqmzSyaMKE3/download" --progress=bar:force
```

Run the following commands to create a conda environment and install dependencies:
```bash
conda create -y -n shape_comp python=3.10
conda activate shape_comp
conda env update --name shape_comp --file environment.yml --prune
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

# expected output of running the line below is "True"
python -c "import torch; print(torch.cuda.is_available())"
```

(Setup was verified to work on Ubuntu 22.04, NVIDIA driver `570.195.03`, and `nvcc` version `12.3`)

## Running inference
Inference will be run with the checkpoint `shape_comp.ckpt` which has the model weights for the following object categories (refer to `Table 1` in the paper):
- `apple`
- `bottle`
- `bowl`
- `box`
- `can`
- `hammer`

NOTE: adjust `mesh_create_batch_size` in `inference.py` based on available GPU memory

### With provided example

An example partial point cloud `can_partial.ply` can be found in `example` directory along with scene it was extracted from (`scene_pcd.ply`).

Inside the conda environment, execute:
```bash
python inference.py
```

This will produce `can_completed.ply`.

### With your own data
Start with a segmented point cloud of an object and modify the following lines in `inference.py`:
1. Update `obj_name: str` to the object category whose geometry most closely resembles your segmented point cloud.
2. Update `partial_pcd_fp: str` to the filepath of the segmented point cloud (expected extension is `.ply`).

Run `inference.py` to obtain the estimated completed shape as `{obj_name}_completed.ply`.


# Reference
- [Diffusion-SDF](https://github.com/princeton-computational-imaging/Diffusion-SDF): Conditional Generative Modeling of Signed Distance Functions

# Citation
If you find this work helpful, please consider citing our paper:
```
@inproceedings{kashyap2025singleview,
  author    = {Abhishek Kashyap and Yuxuan Yang and Henrik Andreasson and Todor Stoyanov},
  title     = {Single-View Shape Completion for Robotic Grasping in Clutter},
  booktitle = {Proceedings of the 13th International Conference on Robot Intelligence Technology and Applications (RiTA 2025)},
  year      = {2025},
  series    = {Lecture Notes in Networks and Systems},
  publisher = {Springer},
  address   = {London, United Kingdom},
  month     = dec,
  url       = {https://amm.aass.oru.se/shape-completion-grasping}
}
```
