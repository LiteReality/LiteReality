<h1 align="center">
  <img src="asset/logo.png" alt="LiteReality Logo" width="50" align="absmiddle" />
  LiteReality: Graphics-Ready 3D Scene Reconstruction from RGB-D Scans
</h1>

<p align="center">
  <b>NeurIPS 2025</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2507.02861"><img src="https://img.shields.io/badge/arXiv-2507.02861-b31b1b.svg?style=flat-square" alt="arXiv"></a>
  <a href="https://litereality.github.io"><img src="https://img.shields.io/badge/Project%20Page-LiteReality-blue.svg?style=flat-square" alt="Project Page"></a>
  <a href="https://www.youtube.com/watch?v=ecK9m3LXg2c&feature=youtu.be"><img src="https://img.shields.io/badge/Video-Presentation-yellow.svg?style=flat-square" alt="Video"></a>
</p>


<p align="center">
  <b><a href="https://zheninghuang.github.io/">Zhening Huang</a></b><sup>1</sup>,
  <b><a href="https://xywu.me">Xiaoyang Wu</a></b><sup>2</sup>,
  <b><a href="https://www.cl.cam.ac.uk/~fz261/">Fangcheng Zhong</a></b><sup>1</sup>,
  <b><a href="https://hszhao.github.io">Hengshuang Zhao</a></b><sup>2</sup>,
  <b><a href="https://www.niessnerlab.org/index.html">Matthias Nießner</a></b><sup>3</sup>,
  <b><a href="https://www.eng.cam.ac.uk/profiles/jl221">Joan Lasenby</a></b><sup>1</sup>
</p>
<p align="center">
  <sup>1</sup>University of Cambridge &nbsp; <sup>2</sup>The University of Hong Kong &nbsp; <sup>3</sup>Technical University of Munich
</p>

---

## 📢 News

- **[2026-01-18]** LiteReality is out! 🔥 
- **[2025-07-03]** Our paper is available on [arXiv](https://arxiv.org/abs/2507.02861)! Check out the 🎬 [video demo](https://www.youtube.com/watch?v=ecK9m3LXg2c).
- **[Coming Soon]** We are actively preparing the full code release. ⭐ **Star this repository** to stay tuned!

---

## 🛠 Prerequisites

| Requirement | Details |
| :--- | :--- |
| **GPU** | NVIDIA RTX 3090 (24GB VRAM) or larger recommended. |
| **Storage** | ~300 GB for `LiteReality_database`. |
| **OS** | Linux (Tested on Ubuntu 20.04/22.04). |

---

## ⚙️ Installation

### Step 1: Create Conda Environment
```bash

git clone https://github.com/LiteReality/LiteReality.git
cd LiteReality

conda create -n LR_env python=3.9 -y
conda activate LR_env

# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
# Install LiteReality in editable mode
pip install -e .
```

### Step 2: Install GroundingDINO

The GroundingDINO code in this repository includes patches for compatibility with PyTorch 2.5.1+ and CUDA 12.4.

```bash
cd third_party
git clone [https://github.com/IDEA-Research/GroundingDINO.git](https://github.com/IDEA-Research/GroundingDINO.git)
cp setup_grounding_dino.py GroundingDINO/setup.py # cp this setup files to GroundingDINO for eazy installation, or you can follow GroundingDINO Repo for debugging if you see issues.

cd GroundingDINO

# Install dependencies
pip install -r requirements.txt
conda install -c conda-forge gcc=13 gxx=13 -y
pip install -e . --no-build-isolation
cd ../..
```
*Note: If issues persist, please refer to the official [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) repository.*

### Step 3: Download Pretrained Weights

This script will download weights for CLIP, DinoV2, Qwen-VL-8B-Instruct, and SAM.

```bash
python third_party/download_pretrained_weights.py 
```

### Step 4: Install Blender

```bash
mkdir -p third_party/blender_dir
wget -c [https://download.blender.org/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz](https://download.blender.org/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz)
tar -xf blender-3.6.0-linux-x64.tar.xz -C third_party/blender_dir
rm blender-3.6.0-linux-x64.tar.xz

# Install required Python packages for Blender's bundled Python
BLENDER_PY="third_party/blender_dir/blender-3.6.0-linux-x64/3.6/python/bin/python3.10"
$BLENDER_PY -m pip install --upgrade pip
$BLENDER_PY -m pip install trimesh shapely pillow numpy
```

---

## 📊 Data Preparation

### Download Material Database

This downloads and extracts the material database (~300GB) to `./LiteReality_Database/`.

```bash
python litereality/utils/litereality_database_download.py 
```

### Download Example Scans

This downloads example scans to the `./scans/` directory.

```bash
python litereality/utils/download_example_scans.py
```

---


## Test on Example Scans

After downloading the database and example scans, run the full test suite:

```bash
python example_scans_test.sh
```

## Test on Your Own Scans

1. **Prepare Data:** Organize your scan data folder. You can see an example recording process below:

<video width="640" height="360" controls>
  <source src="asset/scan_your_own_room.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


2. **Run:** Once your scan data folder is placed inside `/scans`, run:
```bash
bash script.sh scans/{your_scan_name} {scene_name}
```

**Example:**
```bash
bash script.sh scans/2025_01_20_08_44_07 BordRoom_CUED
```


## Output Structure


### 🔧 **output/mat_painting_stage/**
Contains material painting results for each processed scene:
- `{scene_name}/` - Per-object material assignments and textures
- `{scene_name}_output_gltf/` - GLTF exports with applied PBR materials

### 📦 **output/object_stage/**
Contains intermediate object-level processing results:
- `{scene_name}/` - Individual reconstructed objects before material painting

### 🎨 **output/whole_scene_model/**
Final integrated scene models ready for rendering:
- `blender/` - Native Blender project files (`.blend`) for the reconstructed scene
- `glb/` - 3D scene files (`.glb`) with full PBR materials for the reconstructed scene

### 🎬 **output/whole_scene_render/**
Rendered visualizations and videos of the complete scenes:
- `videos/` - Side-by-side comparison with the original RGB-D inputs
- `rendered_rgbd/` - Rendered images from reconstructed scene 


---

## Acknowledgments

The following works are very helpful and inspirational for the creation of LiteReality:

- **[Make-it-Real](https://sunzey.github.io/Make-it-Real/)**: Unleashing Large Multimodal Model for Painting 3D Objects with Realistic Materials
- **[MatSynth](https://huggingface.co/datasets/gvecchio/MatSynth)**: A Modern PBR Materials Dataset
- **[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)**: Alibaba's Vision-Language Model
- **[Phone2Proc](https://arxiv.org/abs/2212.04618)**: Bringing Robust Robots Into Our Chaotic World
- **[3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future)**: 3D Furniture Shape with TextURE
- **[AI2-THOR](https://ai2thor.allenai.org/)**: An Interactive 3D Environment for Visual AI
- **[Apple RoomPlan](https://developer.apple.com/documentation/roomplan)**: ARKit 6 framework for 3D floor plans

We gratefully acknowledge the contributions of these projects to advancing 3D scene reconstruction and graphics-ready content generation.

---

## 📝 Citation

If you find this project useful for your research, please cite:

```bibtex
@inproceedings{huang2025litereality,
  title={LiteReality: Graphics-Ready 3D Scene Reconstruction from RGB-D Scans},
  author={Zhening Huang and Xiaoyang Wu and Fangcheng Zhong and Hengshuang Zhao and Matthias Nießner and Joan Lasenby},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}

```
