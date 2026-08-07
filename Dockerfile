FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Core system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl xz-utils \
    && rm -rf /var/lib/apt/lists/*

# OpenCV / headless rendering dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgomp1 libglu1-mesa libxi6 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

# Open3D / Blender dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libusb-1.0-0 libegl1 libxxf86vm1 \
    libxfixes3 libxrandr2 libxinerama1 libxcursor1 \
    libfreetype6 libfontconfig1 libtbb2 \
    && rm -rf /var/lib/apt/lists/*

# ffmpeg for video export
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Install Miniforge (conda-forge default, no Anaconda TOS required)
# ---------------------------------------------------------------------------
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Create conda env with Python 3.10
RUN conda create -n litereality python=3.10 -y

# Activate env for all subsequent RUN commands
SHELL ["conda", "run", "-n", "litereality", "/bin/bash", "-c"]

# ---------------------------------------------------------------------------
# Install PyTorch (CUDA 12.4)
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ---------------------------------------------------------------------------
# Install Python dependencies (split for better layer caching)
# ---------------------------------------------------------------------------
WORKDIR /app

# Core scientific computing
RUN pip install --no-cache-dir "numpy<2.0"

# Computer vision and image processing
RUN pip install --no-cache-dir opencv-python scikit-image pillow scikit-learn shapely accelerate

# 3D processing
RUN pip install --no-cache-dir open3d trimesh rtree

# ML and transformers
RUN pip install --no-cache-dir transformers segment-anything huggingface_hub

# Utilities
RUN pip install --no-cache-dir tqdm requests gdown python-dotenv

# GroundingDINO extra deps
RUN pip install --no-cache-dir addict yapf timm "supervision>=0.22.0" pycocotools

# ---------------------------------------------------------------------------
# Install GroundingDINO (skip CUDA extension, use PyTorch-native fallback)
# All Python deps (torch, torchvision, transformers, addict, yapf, timm,
# supervision, pycocotools) are already installed above.
# Package is importable via PYTHONPATH="/app/third_party/GroundingDINO".
# ---------------------------------------------------------------------------
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git /app/third_party/GroundingDINO

# Apply compatibility patches for Python 3.10+ and newer transformers
COPY litereality/utils/setup_grounding_dino.py /tmp/setup_grounding_dino.py
COPY litereality/utils/grounding_dino_patches/ /tmp/grounding_dino_patches/
RUN python /tmp/setup_grounding_dino.py \
    --target-dir /app/third_party/GroundingDINO \
    --patches-dir /tmp/grounding_dino_patches \
    --skip-clone --skip-verify

# ---------------------------------------------------------------------------
# Install Blender 3.6.0
# ---------------------------------------------------------------------------
RUN mkdir -p /app/third_party/blender_dir

RUN wget -q https://download.blender.org/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz \
    && tar -xf blender-3.6.0-linux-x64.tar.xz -C /app/third_party/blender_dir \
    && rm blender-3.6.0-linux-x64.tar.xz

# Install Python packages into Blender's bundled Python
RUN /app/third_party/blender_dir/blender-3.6.0-linux-x64/3.6/python/bin/python3.10 \
    -m pip install --no-cache-dir --upgrade pip

RUN /app/third_party/blender_dir/blender-3.6.0-linux-x64/3.6/python/bin/python3.10 \
    -m pip install --no-cache-dir trimesh shapely pillow numpy

# ---------------------------------------------------------------------------
# Copy project source code and install
# ---------------------------------------------------------------------------
COPY . /app/

RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Download pretrained weights (GroundingDINO + SAM)
# ---------------------------------------------------------------------------
RUN mkdir -p /app/third_party/pre-trained \
    /app/third_party/hf_cache \
    /app/third_party/GroundingDINO/weights

# GroundingDINO weights (~700MB)
RUN wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
    -P /app/third_party/GroundingDINO/weights/

# SAM weights (~2.5GB)
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -P /app/third_party/pre-trained/

# Uncomment to bake HuggingFace models into the image (adds ~20GB+):
# RUN HF_HOME=/app/third_party/hf_cache python /app/litereality/utils/download_pretrained_weights.py

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
ENV PATH="/app/third_party/blender_dir/blender-3.6.0-linux-x64:${PATH}"
ENV PYTHONPATH="/app:/app/third_party/GroundingDINO"
ENV HF_HOME="/app/third_party/hf_cache"

# Create directories for runtime data
RUN mkdir -p /app/scans /app/output /app/cache /app/input /app/litereality_database

# ---------------------------------------------------------------------------
# Entrypoint - activate conda env by default
# ---------------------------------------------------------------------------
WORKDIR /app

# Switch back to default shell for conda init
SHELL ["/bin/bash", "-c"]

# Initialize conda for bash and auto-activate litereality env
RUN /opt/conda/bin/conda init bash \
    && echo "conda activate litereality" >> ~/.bashrc

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "litereality"]
CMD ["/bin/bash"]
