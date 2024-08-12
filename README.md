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


----------------------------------------------------------------------------------------------

# ByteTrack 安裝建議

依各系統和配置可能會有不同的安裝方式。

## 系統要求

Linux
- Ubuntu 20.04 或更高版本
- Python 3.8 或更高版本

## 安裝步驟

### 1. ByteTrack 

```bash
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
```

### 2. 安裝 HDF5

```bash
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev
```

### 3. 修改 requirements.txt

打開 requirements.txt 文件，註解掉以下兩行：

```text
torch>=1.7
torchvision>=0.10.0
```

### 4. 安裝 Python

```bash
pip3 install cmake==3.22
pip3 install -r requirements.txt
pip3 install cython
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
pip3 install numpy==1.23.4
```

### 5. 設置 ByteTrack

```bash
python3 setup.py develop
```

### 6. 安裝 Flask 及相關依賴

```bash
pip3 install flask
pip3 install flask_socketio
pip install av
```

### 7. 下載 ONNX 和 ONNX Runtime

```bash
pip3 install onnx==1.12.0
pip3 install onnxsim==0.4.33
```

從 [Jetson Zoo](https://elinux.org/Jetson_Zoo#ONNX_Runtime) 下載 `onnxruntime_gpu-1.15.1`，然後安裝：

```bash
pip3 install ../Downloads/onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl
```

### 8. 安裝 TensorRT

```bash
sudo apt-get install tensorrt nvidia-tensorrt-dev python3-libnvinfer-dev
```

將 TensorRT 複製到虛擬環境中：

```bash
cp -r /usr/lib/python3.8/dist-packages/tensorrt* miniconda3/envs/yolov8/lib/python3.8/site-packages
```

### 9. 將 PyTorch 模型轉換為 TensorRT

確保已經安裝了 `ultralytics` 庫，然後使用以下腳本將模型導出為 TensorRT 格式：

```python
from ultralytics import YOLO

# 載入 YOLOv8 模型
model = YOLO('yolov8n.pt')

# 將模型導出為 TensorRT 格式
model.export(
    format='engine',    # 導出格式
    imgsz=640,          # 圖像尺寸 (高, 寬) 或標量
    keras=False,        # 是否使用 Keras 導出 TF SavedModel
    optimize=False,     # 是否優化 TorchScript 模型
    half=False,         # 是否使用 FP16 量化
    int8=False,         # 是否使用 INT8 量化
    dynamic=True,       # 是否使用動態軸
    simplify=True,      # 是否簡化模型
    opset=12,           # ONNX: opset 版本
    workspace=4,        # TensorRT: 工作區域大小 (GB)
    nms=False,          # CoreML: 是否添加 NMS
)
```

## 注意事項

- 請確保所有步驟都已正確完成，以避免依賴庫衝突或安裝問題。
- 根據您的系統和需求調整 TensorRT 安裝和模型導出設置。

```

