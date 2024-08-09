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
###安裝torch&torchvision的另一種方式

下載 torch
sudo apt install -y libfreetype6-dev
sudo apt install python3-pip libopenblas-base libopenmpi-dev libomp-dev

https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html#overview__section_z4r_vjd_v2b
export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl

pip3 install --no-cache $TORCH_INSTALL

sudo apt-get install libjpeg-dev zliblg-dev libpython3-dev libavcodec-dev libavformat-dev

下載 torchvision
cd ..
sudo git clone --branch v0.13.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.13.0
export CUDA_HOME=/usr/local/cuda
python3 setup.py install --user

檢查 torch 與 torchvision
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
![image](https://github.com/user-attachments/assets/ff35cc84-72bd-4723-ab45-39058d64efe3)

----------------------------------------------------------------------------------------------

下載 ByteTrack
cd ..
git clone https://github.com/ifzhang/ByteTrack.git

cd ByteTrack

sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev

將 requirements.txt  中 torch>=1.7 torchvision>=0.10.0 註解掉
pip3 install cmake==3.22
pip3 install -r requirements.txt

python3 setup.py develop
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
pip3 install numpy==1.23.4

下載 Flask
pip3 install flask
pip3 install flask_socketio
pip install av

Export pytorch to tensorRT
下載 Onnx
pip3 install onnx==1.12.0
pip3 install onnxsim==0.4.33

下載 onnxruntime 1.15.1 https://elinux.org/Jetson_Zoo#ONNX_Runtime
pip3 install ../Downloads/onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl 
pip3 install numpy==1.23.4

下載 TensorRT
sudo apt-get install tensorrt nvidia-tensorrt-dev python3-libnvinfer-dev
# 將 tensorRT 複製到虛擬環境中
cp -r /usr/lib/python3.8/dist-packages/tensorrt* miniconda3/envs/yolov8/lib/python3.8/site-packages

python3 pt2trt.py 
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Export the model to TensorRT format
# model.export(format='onnx')  # creates 'yolov8n.engine'

model.export(format='engine', # format to export to
               imgsz = 640, # image size as scalar or (h, w) list, i.e. (640, 480)
               keras = False, # use Keras for TF SavedModel export
               optimize = False, # TorchScript: optimize for mobile
               half = False, # FP16 quantization
               int8 = False, # INT8 quantization
               dynamic = True, # ONNX/TensorRT: dynamic axes
               simplify = True, # ONNX/TensorRT: simplify model
               opset = 12, # ONNX: opset version (optional, defaults to latest)
               workspace = 4, # TensorRT: workspace size (GB)
               nms = False, # CoreML: add NMS
               # batch = 10
               )
![image](https://github.com/user-attachments/assets/c54a0b2c-266c-47de-a359-94e00642d390)
