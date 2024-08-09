# CarPlateRecognition
 yolo.deepstream

## Installation

### Prepare YOLOv8 Environment

1. Build ByteTrack:
   ```sh
   cd ByteTrack
   python setup.py build_ext --inplace
   ```

2. Add CUDA Path to Environment Variables:
   ```sh
   sudo vim ~/.bashrc
   ```

   Add the following lines:
   ```sh
   export PATH=/usr/local/cuda-11.4/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   export CUDA_HOME=/usr/local/cuda-11.4
   ```

   Source the updated `.bashrc`:
   ```sh
   source ~/.bashrc
   ```

### Install Miniconda
1. Download Miniconda:
   [Miniconda Installation Guide](https://docs.anaconda.com/free/miniconda/)

   ```sh
   mkdir -p ~/miniconda3
   cd miniconda3
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
   bash Miniconda3-latest-Linux-aarch64.sh -b -u -p ~/miniconda3
   ```

   Remove the installation script:
   ```sh
   rm -rf ~/miniconda3/miniconda.sh
   ```

   Initialize Conda:
   ```sh
   ~/miniconda3/bin/conda init bash
   ~/miniconda3/bin/conda init zsh
   ```

### Install YOLOv8
1. Clone YOLOv8 repository:
   ```sh
   git clone https://github.com/ultralytics/ultralytics.git
   ```

2. Install YOLOv8:
   ```sh
   pip3 install ultralytics==8.1.18
   ```

3. Update system packages:
   ```sh
   sudo apt update
   sudo apt install -y python3-pip
   pip3 install --upgrade pip
   cd ultralytics
   ```

### Install PyTorch and Dependencies
1. Install additional dependencies:
   ```sh
   sudo apt install -y libfreetype6-dev python3-pip libopenblas-base libopenmpi-dev libomp-dev
   ```

2. Set and install PyTorch:
   ```sh
   export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
   pip3 install --no-cache $TORCH_INSTALL
   ```

3. Install additional libraries:
   ```sh
   sudo apt-get install libjpeg-dev zliblg-dev libpython3-dev libavcodec-dev libavformat-dev
   ```

### Install TorchVision
1. Clone TorchVision repository:
   ```sh
   cd ..
   git clone --branch v0.13.0 https://github.com/pytorch/vision torchvision
   cd torchvision
   ```

2. Set environment variables and install:
   ```sh
   export BUILD_VERSION=0.13.0
   export CUDA_HOME=/usr/local/cuda
   python3 setup.py install --user
   ```

### Verify Installation
1. Check Torch and TorchVision:
   ```python
   >>> import torch
   >>> print(torch.__version__)
   >>> print('CUDA available: ' + str(torch.cuda.is_available()))
   >>> print('cuDNN version: ' + str(torch.backends.cudnn.version()))

   >>> a = torch.cuda.FloatTensor(2).zero_()
   >>> print('Tensor a = ' + str(a))
   >>> b = torch.randn(2).cuda()
   >>> print('Tensor b = ' + str(b))
   >>> c = a + b
   >>> print('Tensor c = ' + str(c))

   >>> import torchvision
   >>> print(torchvision.__version__)
   ```

### Run Inference with YOLOv8
1. Predict using YOLOv8:
   ```sh
   cd ../ultralytics/
   yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
   ```
