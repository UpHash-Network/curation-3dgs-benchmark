#!/usr/bin/env bash
# One-time setup of the WSL-side training venv (~/fx-venv) for gsplat simple_trainer.
set -e
cd ~
if [ ! -d fx-venv ]; then
    python3 -m venv fx-venv
fi
source fx-venv/bin/activate
pip install -q --upgrade pip
# torch/torchvision matching the prebuilt gsplat wheel (pt24cu124)
pip install -q torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install -q gsplat==1.5.3 --index-url https://docs.gsplat.studio/whl/pt24cu124 --no-deps
# gsplat examples deps (per examples/requirements.txt; viewer/fused-cuda extras omitted --
# viewer is disabled and fused_ssim is replaced by an equivalent plain-torch SSIM)
pip install -q "numpy<2.0.0" "opencv-python-headless==4.10.0.84" tyro "imageio[ffmpeg]" \
    "torchmetrics[image]" tqdm pyyaml tensorboard splines scikit-learn tensorly \
    typing_extensions Pillow \
    "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e"
python - <<'EOF'
import torch, gsplat, torchmetrics, numpy, cv2, tyro, imageio, tensorly, splines
from pycolmap import SceneManager
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gsplat", gsplat.__version__, "numpy", numpy.__version__)
print("WSL_ENV_OK")
EOF
