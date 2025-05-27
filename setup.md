Ubuntu VM Setup
========

VM Requirements
---------------

*   Secure Boot off
*   Cuda compatible GPUs

Required Dependencies and Drivers
---------------------------------

### NVIDIA-SMI 570

[Following this NVIDIA documentation](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html#ubuntu-installation) we get a series of commands. 

For an Ubuntu 24.04 VM with Trusted Launch (secure boot) disabled, "distro" is "ubuntu2404," and "arch" is "x86_64." The table detailing these values is found at the top of the NVIDIA documentation. An example is for network installations, the following command:
`wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.1-1_all.deb`
is changed to:
`wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb`

1. `wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb`
2. `sudo dpkg -i cuda-keyring_1.1-1_all.deb`
3. `sudo apt install nvidia-open`
4. `sudo apt install cuda-drivers`

### Cuda and Cuda Toolkit

This can be the latest version, the current installation is 12.8 

[https://developer.nvidia.com/cuda-downloads?target\_os=Linux&target\_arch=x86\_64&Distribution=Ubuntu&target\_version=22.04&target\_type=deb\_local](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

We are 24.04 and have chosen the network route

1. `wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb`
2. `sudo dpkg -i cuda-keyring_1.1-1_all.deb`
3. `sudo apt-get update`
4. `sudo apt-get -y install cuda-toolkit-12-8`

We do not need to install the drivvers listed in the instructions as those have already beeing installed. 

### Conda

*   Install miniconda for ubuntu
*   Run the following command to download the latest Miniconda (a minimal Conda distribution):  
    `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
*   Run the installer:  
    `bash Miniconda3-latest-Linux-x86_64.sh`
*   Follow the on-screen instructions:
    *   Press Enter to continue.
    *   Scroll through the license (Space key) and type yes to accept.
    *   Choose an installation directory (default is ~/miniconda3 or ~/anaconda3 which will be on an Azure VM /home/azureuser/miniconda3).


### Initialize Conda

*   Once installed, run:
    *   `source ~/.bashrc`
    *   `conda init`
*   Then restart your terminal or run:
    *   `source ~/.bashrc`

### Conda Issues & Verify Installation

*   Check if Conda is installed:
    *   `conda --version`
*   You should see output like:
    *   `conda 23.1.0`
*   If there are issues with installation, try this
    *   `echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc`
    *   `source ~/.bashrc`

### Open3D with Cuda
[https://www.open3d.org/docs/latest/compilation.html](https://www.open3d.org/docs/latest/compilation.html)

If g++ is not installed, running the following command should install it:
1. `sudo apt install -y build-essential gcc g++`
2. `g++ --version`

Check if cmake is installed, and if not, install it:
1. `cmake --version`
2. `sudo apt install -y cmake`

We must install the dependencies inside Open3d
1. `git clone https://github.com/isl-org/Open3D`
2. `cd Open3D`
3. `util/install_deps_ubuntu.sh`
4. `mkdir build`
5. `cd build`

* We now activate the virtualenv
1. `conda init`
2. `source ~/.bashrc`

*   When you make the packages, use the following flags  
    - `cmake -DBUILD_CUDA_MODULE=ON -DBUILD_PYTHON_MODULE=ON -DCUDA_ARCH=86 ..`
    - If CPU is not found, try this command: `cmake -DBUILD_CUDA_MODULE=ON -DBUILD_PYTHON_MODULE=ON -DBUILD_SHARED_LIBS=ON -DBUILD_CXX11=ON ..`
    - `make -j$(nproc)`
    - `sudo make install`

Install the python library
1. `sudo make install-pip-package`

*   Navigate to the pip packages and pip install them
*   You made need to pip install these again later in the correct conda environment

There may be errors with the CUDA Compiler: "No CMAKE_CUDA_COMPILER could be found."

### Fixing CUDA Errors

CUDA may be installed but not set in the paths correctly:
1. `export CUDA_HOME=/usr/local/cuda-12.8`
2. `export PATH=$CUDA_HOME/bin:$PATH`
3. `export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH`
4. `export CUDACXX=$CUDA_HOME/bin/nvcc`

Let's test if after those commands, results are returned:
1. `which nvcc`
2. `nvcc --version`

If values are returned, we are able to set these permanently:
1. `echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc`
2. `echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc`
3. `echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc`
4. `echo 'export CUDACXX=$CUDA_HOME/bin/nvcc' >> ~/.bashrc`
5. `source ~/.bashrc`

### SLAM Repo Download

*   Git clone this repo
    *   [https://github.com/VladimirYugay/Gaussian-SLAM/tree/main?tab=readme-ov-file](https://github.com/VladimirYugay/Gaussian-SLAM/tree/main?tab=readme-ov-file)
*   `git clone https://github.com/VladimirYugay/Gaussian-SLAM`

### Create conda environment

*   `conda env create -f environment.yml`
*   `conda activate gslam`

### Create conda environment replication
* `conda create -n gslam python=3.10 -y`
* `conda activate gslam`
* `conda install -c conda-forge faiss-gpu=1.8.0 -y`
* `conda install -c nvidia cuda-toolkit=12.1 -y`
* `conda install pip -y`
```
pip install open3d==0.18.0 wandb trimesh pytorch_msssim torchmetrics tqdm imageio opencv-python plyfile \
  git+https://github.com/eriksandstroem/evaluate_3d_reconstruction_lib.git@9b3cc08be5440db9c375cc21e3bd65bb4a337db7 \
  git+https://github.com/VladimirYugay/simple-knn.git@c7e51a06a4cd84c25e769fee29ab391fe5d5ff8d \
  git+https://github.com/VladimirYugay/gaussian_rasterizer.git@9c40173fcc8d9b16778a1a8040295bc2f9ebf129
```

### More Dependencies

*   Likely, the environment.yml or the command above will not properly install everything
*   There will be some repositories that need some edits
*   Install each dependency before the last two git repositories
*   For the last two, you need to git clone them, attempt to build them using setup.py (pip install .)
*   This build will fail because some of the files are missing necessary dependencies on ubuntu

### Pip install
* `pip install wandb`
* `pip install opencv-python`
* `pip install trimesh pytorch_msssim torchmetrics tqdm plyfile`
* `pip install open3d==0.18.0 git+https://github.com/eriksandstroem/evaluate_3d_reconstruction_lib.git@9b3cc08be5440db9c375cc21e3bd65bb4a337db7`

### Gaussian Rasterizer

*   Git clone the repository:
    *   `git clone https://github.com/VladimirYugay/gaussian_rasterizer.git`
    *   `cd gaussian_rasterizer`
    *   `git checkout 9c40173fcc8d9b16778a1a8040295bc2f9ebf129`
*   Likely it will give this error:
    *   Include `<cstdint>` for uint32\_t / uint64\_t
    *   error: identifier "uint32\_t" is undefined
*   This means that uint32\_t and uint64\_t are not defined. Try modifying rasterizer\_impl.h by adding:
    *   `#include <cstdint>`
*   at the top of the file. This ensures that fixed-width integer types like uint32\_t and uint64\_t are recognized.

### Simpleknn

*   Git clone the repository:
    *   `git clone https://github.com/VladimirYugay/simple-knn.git`
    *   `cd simple-knn`
    *   `git checkout c7e51a06a4cd84c25e769fee29ab391fe5d5ff8d`
*   Likely it will give the error
    *   FLT\_MAX is Undefined
    *   float best\[3\] = { FLT\_MAX, FLT\_MAX, FLT\_MAX };
*   FLT\_MAX is defined in `<cfloat>` (C++ standard header for float limits).
*   Try adding:  
    `#include <cfloat>`
*   at the top of the .cu file this error happens in (simple_knn.cu)
*   Once you have created the pip package for these two, pip install them both


### Git-LFS

*   `sudo apt update sudo apt install git-lfs`

### Downloading Data

*   Run the script “scripts/download\_replica.sh” in the SLAM repo
*   There may be issues with permissions opening the files in Linux
*   Use chmod to fix this
*   You may also need to run
    *   `git lfs install`
    *   `git lfs pull`