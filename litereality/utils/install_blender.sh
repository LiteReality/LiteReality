mkdir -p third_party/blender_dir
wget -c https://download.blender.org/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz
tar -xf blender-3.6.0-linux-x64.tar.xz -C third_party/blender_dir
rm blender-3.6.0-linux-x64.tar.xz

BLENDER_PY="third_party/blender_dir/blender-3.6.0-linux-x64/3.6/python/bin/python3.10"
$BLENDER_PY -m pip install --upgrade pip
$BLENDER_PY -m pip install trimesh shapely pillow numpy