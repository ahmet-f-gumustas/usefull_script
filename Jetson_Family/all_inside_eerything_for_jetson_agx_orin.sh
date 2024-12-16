#!/bin/bash

# Jetson AGX Orin Developer Kit Setup Script
# This script automates the setup process for the Jetson AGX Orin Developer Kit.
# It includes flashing JetPack SDK, building the kernel, installing libraries,
# setting up AI frameworks, configuring Docker with CUDA support, and more.
# 
# **Warning:** This script performs system-level changes and compiles software from source.
# Ensure you understand each step before running. It's recommended to run this on a fresh installation.

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Pipeline returns the exit status of the last command in the pipe that failed

# Variables
USER_HOME=$(eval echo ~${SUDO_USER})
PROJECTS_DIR="$USER_HOME/Projects"
JETSON_LINUX_URL="https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v3.0/release/jetson_linux_r36.3.0_aarch64.tbz2"
ROOT_FS_URL="https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v3.0/release/tegra_linux_sample-root-filesystem_r36.3.0_aarch64.tbz2"
KERNEL_URL="https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.8.9.tar.xz"
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb"

# Ensure the script is run with sudo
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root using sudo."
  exit
fi

# Create Projects directory
mkdir -p "$PROJECTS_DIR"
cd "$PROJECTS_DIR"

echo "=== Jetson AGX Orin Developer Kit Setup ==="

########################################
# 1. Flashing JetPack SDK with SDK Manager
########################################

echo "=== Step 1: Flashing JetPack SDK with SDK Manager ==="

echo "Please ensure you have downloaded and installed the NVIDIA SDK Manager."
echo "Flashing JetPack typically requires a GUI and cannot be fully automated via script."
echo "Proceeding to open SDK Manager. Please follow the on-screen instructions to flash JetPack to NVME."

# Launch SDK Manager (Assuming it's installed and available in PATH)
sdkmanager &

echo "After flashing, press Enter to continue..."
read -r

########################################
# 2. Building the Kernel
########################################

echo "=== Step 2: Building the Kernel ==="

# Install Dependencies
echo "Installing kernel build dependencies..."
apt-get update
apt-get install -y git fakeroot build-essential ncurses-dev xz-utils libssl-dev bc flex libelf-dev bison

# Download and Extract Sources
echo "Downloading Jetson Linux and Root Filesystem..."
wget -O jetson_linux.tbz2 "$JETSON_LINUX_URL"
wget -O root_fs.tbz2 "$ROOT_FS_URL"

echo "Extracting Jetson Linux..."
tar -xvjf jetson_linux.tbz2

echo "Extracting Root Filesystem..."
tar -xvjf root_fs.tbz2

echo "Downloading Linux Kernel..."
wget -O linux-6.8.9.tar.xz "$KERNEL_URL"

echo "Extracting Linux Kernel..."
xz -d linux-6.8.9.tar.xz
tar -xf linux-6.8.9.tar
mv linux-6.8.9 kernel-jammy-src

echo "Syncing sources with NVIDIA..."
cd Linux_for_Tegra/source/
./source_sync.sh -k -t jetson_36.3

echo "Replacing kernel source..."
cd kernel
rm -rf kernel-jammy-src
cd "$PROJECTS_DIR"
mv kernel-jammy-src Linux_for_Tegra/source/kernel/

# Modify defconfig
echo "Modifying kernel configuration..."
cd Linux_for_Tegra/source/kernel/kernel-jammy-src/configs/aarch
make defconfig
scripts/config --file .config --enable ARM64_PMEM
scripts/config --file .config --enable PCIE_TEGRA194
scripts/config --file .config --enable PCIE_TEGRA194_HOST
scripts/config --file .config --enable BLK_DEV_NVME
scripts/config --file .config --enable NVME_CORE
scripts/config --file .config --enable FB_SIMPLE

# Build the Kernel
echo "Building the kernel..."
cd "$PROJECTS_DIR/Linux_for_Tegra/source/"
make -j "$(nproc)" -C kernel

# Set installation path
export INSTALL_MOD_PATH="$PROJECTS_DIR/Linux_for_Tegra/rootfs"

# Install the kernel
echo "Installing the kernel..."
sudo -E make install -C kernel

# Copy the kernel image
cp kernel/kernel-jammy-src/arch/arm64/boot/Image "$PROJECTS_DIR/Linux_for_Tegra/kernel/Image"

# Build and Install Modules
echo "Building and installing kernel modules..."
export KERNEL_HEADERS="$PWD/kernel/kernel-jammy-src"
export INSTALL_MOD_PATH="$PROJECTS_DIR/Linux_for_Tegra/rootfs"
make modules
sudo -E make modules_install

# Edit Boot Configuration
echo "Modifying boot configuration..."
BOOT_CONF="/boot/extlinux/extlinux.conf"

# Backup existing configuration
cp "$BOOT_CONF" "${BOOT_CONF}.bak"

# Insert the new kernel path
sed -i 's|kernel/Image|'"$PROJECTS_DIR"'/Linux_for_Tegra/kernel/Image|' "$BOOT_CONF"

# Build Device Trees
echo "Building device trees..."
make dtbs
sudo cp nvidia-oot/device-tree/platform/generic-dts/dtbs/* /boot/dtb

########################################
# 3. Installing Required Libraries
########################################

echo "=== Step 3: Installing Required Libraries ==="

# Update package lists
apt-get update

# Install Mandatory Tools and Libraries
echo "Installing mandatory tools and libraries..."
apt-get install -f -y --no-install-recommends \
    ninja-build \
    libopenblas-dev \
    libopenmpi-dev \
    openmpi-bin \
    openmpi-common \
    libomp-dev \
    autoconf \
    bc \
    build-essential \
    cmake \
    ffmpeg \
    g++ \
    gcc \
    gettext-base \
    git \
    gfortran \
    hdf5-tools \
    iputils-ping \
    libatlas-base-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libblas-dev \
    libbz2-dev \
    libc++-dev \
    libcgal-dev \
    libeigen3-dev \
    libffi-dev \
    libfreeimage-dev \
    libfreetype6-dev \
    libglew-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtk-3-dev \
    libgtk2.0-dev \
    libhdf5-dev \
    libjpeg-dev \
    libjpeg-turbo8-dev \
    libjpeg8-dev \
    liblapack-dev \
    liblapacke-dev \
    liblzma-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libomp-dev \
    libopenblas-dev \
    libopenblas-base \
    libopenexr-dev \
    libopenjp2-7 \
    libopenjp2-7-dev \
    libopenmpi-dev \
    libpng-dev \
    libprotobuf-dev \
    libreadline-dev \
    libsndfile1 \
    libsqlite3-dev \
    libssl-dev \
    libswresample-dev \
    libswscale-dev \
    libtbb-dev \
    libtbb2 \
    libtesseract-dev \
    libtiff-dev \
    libv4l-dev \
    libx264-dev \
    libxine2-dev \
    libxslt1-dev \
    libxvidcore-dev \
    libxml2-dev \
    locales \
    moreutils \
    openssl \
    pkg-config \
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-matplotlib \
    qv4l2 \
    rsync \
    scons \
    v4l-utils \
    zlib1g-dev \
    zip \
    nvidia-l4t-gstreamer \
    ubuntu-restricted-extras \
    libsoup2.4-dev \
    libjson-glib-dev

# Install GStreamer
echo "Installing GStreamer..."
apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools \
    gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl \
    gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio \
    gstreamer1.0-rtsp libgstrtspserver-1.0-dev

# Install and Configure ccache
echo "Installing and configuring ccache..."
apt-get install -y ccache
/usr/sbin/update-ccache-symlinks
echo 'export PATH="/usr/lib/ccache:$PATH"' >> "$USER_HOME/.bashrc"
source "$USER_HOME/.bashrc"

########################################
# 4. Installing CUDA, cuDNN, TensorRT
########################################

echo "=== Step 4: Installing CUDA, cuDNN, TensorRT ==="

# Download and Install CUDA Toolkit
echo "Downloading and installing CUDA Toolkit..."
wget -O cuda-keyring.deb "$CUDA_KEYRING_URL"
dpkg -i cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit-12-6 cuda-compat-12-6

# Install cuDNN and TensorRT
echo "Installing cuDNN and TensorRT..."
apt-get install -y cudnn python3-libnvinfer python3-libnvinfer-dev tensorrt

########################################
# 5. Installing CMake
########################################

echo "=== Step 5: Installing CMake ==="

# Remove Existing CMake
echo "Removing existing CMake..."
apt-get remove -y cmake

# Download and Build CMake
echo "Downloading and building CMake..."
CMAKE_VERSION="3.30.3"
wget "https://cmake.org/files/v3.30/cmake-${CMAKE_VERSION}.tar.gz"
tar xf "cmake-${CMAKE_VERSION}.tar.gz"
cd "cmake-${CMAKE_VERSION}"
./configure
make -j "$(nproc)"
make install
cd "$PROJECTS_DIR"

# Verify CMake Installation
cmake --version

########################################
# 6. Installing Clang 18
########################################

echo "=== Step 6: Installing Clang 18 ==="

# Add LLVM Repository and Install Clang 18
echo "Adding LLVM repository and installing Clang 18..."
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
add-apt-repository "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main"
apt-get update
apt-get install -y clang-18 lldb-18 lld-18

# Update Alternatives for Clang
echo "Configuring alternatives for Clang..."
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100

########################################
# 7. Building OpenBLAS
########################################

echo "=== Step 7: Building OpenBLAS ==="

git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS/
export OMP_NUM_THREADS=12
make TARGET=ARMV8 USE_OPENMP=1
make PREFIX=/usr/local install
ldconfig
grep OPENBLAS_VERSION /usr/local/include/openblas_config.h
cd "$PROJECTS_DIR"

########################################
# 8. Installing PyTorch and TensorFlow
########################################

echo "=== Step 8: Installing PyTorch and TensorFlow ==="

# Install PyTorch Native
echo "Installing PyTorch..."
TORCH_INSTALL_URL="https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/torch-2.2.0a0+81ea7a4.nv24.01-cp310-cp310-linux_aarch64.whl"
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade --no-cache "$TORCH_INSTALL_URL"

# Install TensorFlow Native
echo "Installing TensorFlow..."
apt-get -y update
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60dp/tensorflow/tensorflow-2.14.0+nv24.01-cp310-cp310-linux_aarch64.whl

########################################
# 9. Installing Miniconda and Setting Up Python Environment
########################################

echo "=== Step 9: Installing Miniconda and Setting Up Python Environment ==="

# Download and Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh
chmod +x miniconda.sh
bash miniconda.sh -b -p "$USER_HOME/miniconda3"

# Initialize Conda
eval "$("$USER_HOME/miniconda3/bin/conda" shell.bash hook)"
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> "$USER_HOME/.bashrc"
source "$USER_HOME/.bashrc"

# Create Python 3.12 Environment
echo "Creating Python 3.12 environment..."
conda create -y -n py312 python=3.12

########################################
# 10. Building from Source
########################################

echo "=== Step 10: Building from Source ==="

# PyTorch 2.4
echo "=== Building PyTorch 2.4 ==="

git clone --recursive --branch v2.4.1 https://github.com/pytorch/pytorch.git
cd pytorch

export PYTORCH_BUILD_NUMBER=1
export TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
export USE_NCCL=0
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0
export USE_NATIVE_ARCH=1
export USE_DISTRIBUTED=1
export USE_TENSORRT=0
export TORCH_CUDA_ARCH_LIST="8.7"

export PYTORCH_BUILD_VERSION=2.4.1
export MAKEFLAGS="-j$(nproc)"

pip3 install --no-cache-dir -r requirements.txt
pip3 install --no-cache-dir scikit-build ninja
conda install -y -c conda-forge libstdcxx-ng=12
apt-get remove -y cmake
pip install --upgrade cmake
python setup.py bdist_wheel

cd "$PROJECTS_DIR"

# TorchVision
echo "=== Building TorchVision ==="

git clone --branch v0.19.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.19.1
export TORCH_CUDA_ARCH_LIST="8.7"
python setup.py bdist_wheel
cd "$PROJECTS_DIR"

# TorchAudio
echo "=== Building TorchAudio ==="

git clone --branch v2.4.1 https://github.com/pytorch/audio torchaudio
cd torchaudio
export BUILD_VERSION=2.4.1
export TORCH_CUDA_ARCH_LIST="8.7"
python setup.py bdist_wheel
cd "$PROJECTS_DIR"

# TorchText
echo "=== Building TorchText ==="

git clone --branch v0.19.1 https://github.com/pytorch/text.git torchtext
cd torchtext
export BUILD_VERSION=0.19.1
export TORCH_CUDA_ARCH_LIST="8.7"
python setup.py build_ext -j "$(nproc)" bdist_wheel
cd "$PROJECTS_DIR"

########################################
# 11. Building TensorRT
########################################

echo "=== Step 11: Building TensorRT ==="

# Function to build TensorRT for a specific Python version
build_tensorrt() {
    PYTHON_MAJOR=$1
    PYTHON_MINOR=$2
    PYTHON_VERSION="$PYTHON_MAJOR.$PYTHON_MINOR"
    
    echo "=== Building TensorRT for Python $PYTHON_VERSION ==="
    
    export EXT_PATH="$USER_HOME/external"
    export TRT_OSSPATH="$EXT_PATH/TensorRT"
    
    mkdir -p "$EXT_PATH" && cd "$EXT_PATH"
    
    git clone https://github.com/pybind/pybind11.git
    
    # Download Python source and headers
    if [ "$PYTHON_VERSION" == "3.12" ]; then
        PYTHON_TGZ="Python-3.12.2.tgz"
        wget "https://www.python.org/ftp/python/3.12.2/${PYTHON_TGZ}"
        tar -xvf "$PYTHON_TGZ"
        mkdir -p "$EXT_PATH/python3.12/include"
        cp -r Python-3.12.2/Include/* "$EXT_PATH/python3.12/include"
        
        LIBPYTHON_DEB="libpython3.12-dev_3.12.2-5_arm64.deb"
        wget "http://http.us.debian.org/debian/pool/main/p/python3.12/${LIBPYTHON_DEB}"
        ar x "$LIBPYTHON_DEB"
        mkdir -p debian && tar -xf data.tar.zst -C debian
        cp debian/usr/include/aarch64-linux-gnu/python3.12/pyconfig.h "$EXT_PATH/python3.12/include/"
        
    elif [ "$PYTHON_VERSION" == "3.11" ]; then
        PYTHON_TGZ="Python-3.11.9.tgz"
        wget "https://www.python.org/ftp/python/3.11.9/${PYTHON_TGZ}"
        tar -xvf "$PYTHON_TGZ"
        mkdir -p "$EXT_PATH/python3.11/include"
        cp -r Python-3.11.9/Include/* "$EXT_PATH/python3.11/include"
        
        LIBPYTHON_DEB="libpython3.11-dev_3.11.9-1_arm64.deb"
        wget "http://http.us.debian.org/debian/pool/main/p/python3.11/${LIBPYTHON_DEB}"
        ar x "$LIBPYTHON_DEB"
        mkdir -p debian && tar -xf data.tar.xz -C debian
        cp debian/usr/include/aarch64-linux-gnu/python3.11/pyconfig.h "$EXT_PATH/python3.11/include/"
        
    elif [ "$PYTHON_VERSION" == "3.10" ]; then
        PYTHON_TGZ="Python-3.10.11.tgz"
        wget "https://www.python.org/ftp/python/3.10.11/${PYTHON_TGZ}"
        tar -xvf "$PYTHON_TGZ"
        mkdir -p "$EXT_PATH/python3.10/include"
        cp -r Python-3.10.11/Include/* "$EXT_PATH/python3.10/include"
        
        LIBPYTHON_DEB="libpython3.10-dev_3.10.12-1_amd64.deb"
        wget "http://http.us.debian.org/debian/pool/main/p/python3.10/${LIBPYTHON_DEB}"
        ar x "$LIBPYTHON_DEB"
        mkdir -p debian && tar -xf data.tar.xz -C debian
        cp debian/usr/include/aarch64-linux-gnu/python3.10/pyconfig.h "$EXT_PATH/python3.10/include/"
        
    else
        echo "Unsupported Python version: $PYTHON_VERSION"
        return
    fi
    
    # Clone and build TensorRT
    git clone --branch release/10.4 --recursive https://github.com/NVIDIA/TensorRT.git
    cd TensorRT
    mkdir -p build && cd build
    export TRT_LIBPATH=/usr/lib/aarch64-linux-gnu/
    cmake .. -DTRT_LIB_DIR="$TRT_LIBPATH" -DTRT_OUT_DIR="$PWD/out" \
        -DTRT_PLATFORM_ID=aarch64 -DCUDA_VERSION=12.6 \
        -DCUDNN_VERSION=9.4 -DGPU_ARCHS="87"
    CC=/usr/bin/gcc make -j "$(nproc)"
    
    # Build Python Bindings
    cd ../python
    TENSORRT_MODULE=tensorrt PYTHON_MAJOR_VERSION="$PYTHON_MAJOR" PYTHON_MINOR_VERSION="$PYTHON_MINOR" \
    TARGET_ARCHITECTURE=aarch64 TRT_OSSPATH="$TRT_OSSPATH" ./build.sh
    
    # Install the wheel
    pip install "./build/bindings_wheel/dist/tensorrt-*.whl"
    
    # Optional: Install specific wheel
    if [ "$PYTHON_VERSION" == "3.11" ]; then
        pip3 install -U tensorrt-10.4-cp311-none-linux_aarch64.whl
    elif [ "$PYTHON_VERSION" == "3.10" ]; then
        pip3 install -U tensorrt-10.4-cp310-none-linux_aarch64.whl
    fi
    
    cd "$PROJECTS_DIR"
}

# Build TensorRT for Python 3.12
build_tensorrt 3 12

# Build TensorRT for Python 3.11
build_tensorrt 3 11

# Build TensorRT for Python 3.10
build_tensorrt 3 10

########################################
# 12. Building OpenCV with CUDA
########################################

echo "=== Step 12: Building OpenCV with CUDA ==="

# Create and Activate Conda Environment
echo "Creating and activating conda environment 'py311'..."
conda create -y -n py311 python=3.11
conda activate py311
conda install -y cmake numpy

# Download OpenCV and OpenCV Contrib
echo "Downloading OpenCV and OpenCV Contrib..."
cd "$PROJECTS_DIR"
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip

# Build OpenCV
echo "Building OpenCV with CUDA support..."
cd opencv-4.x
mkdir -p build && cd build
export ENABLE_CONTRIB=1
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
cmake -DCMAKE_BUILD_TYPE=RELEASE \
      -DCMAKE_PREFIX_PATH="/usr/lib/aarch64-linux-gnu;/usr/include" \
      -DCPACK_BINARY_DEB=ON \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_opencv_python2=OFF \
      -DBUILD_opencv_python3=ON \
      -DBUILD_opencv_java=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCUDA_ARCH_BIN=8.7 \
      -DCUDA_ARCH_PTX= \
      -DCUDA_FAST_MATH=ON \
      -DCUDNN_INCLUDE_DIR=/usr/include/ \
      -DEIGEN_INCLUDE_PATH=/usr/include/eigen3 \
      -DWITH_EIGEN=ON \
      -DENABLE_NEON=ON \
      -DOPENCV_DNN_CUDA=ON \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
      -DOPENCV_ENABLE_NONFREE=ON \
      -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.x/modules \
      -DOPENCV_GENERATE_PKGCONFIG=ON \
      -DOpenGL_GL_PREFERENCE=GLVND \
      -DWITH_CUBLAS=ON \
      -DWITH_CUDA=ON \
      -DWITH_CUDNN=ON \
      -DWITH_GSTREAMER=ON \
      -DWITH_LIBV4L=ON \
      -DWITH_GTK=ON \
      -DWITH_OPENGL=OFF \
      -DWITH_OPENCL=OFF \
      -DWITH_IPP=OFF \
      -DWITH_TBB=ON \
      -DBUILD_TIFF=ON \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_NEW_PYTHON_SUPPORT=ON \
      -DBUILD_opencv_python3=ON \
      -DHAVE_opencv_python3=ON \
      ../

# Build and Install OpenCV
make -j "$(nproc)"
make install
ldconfig
cd "$PROJECTS_DIR"

# Build OpenCV Python
echo "Building OpenCV Python bindings..."
git clone --recursive https://github.com/opencv/opencv-python.git
cd opencv-python
wget https://raw.githubusercontent.com/dusty-nv/jetson-containers/master/packages/opencv/patches.diff
git apply patches.diff || echo "Failed to apply git patches"

# Apply additional patches
sed -i 's|weight != 1.0|(float)weight != 1.0f|' opencv/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp
sed -i 's|nms_iou_threshold > 0|(float)nms_iou_threshold > 0.0f|' opencv/modules/dnn/src/cuda4dnn/primitives/region.hpp

# Verify patches
grep 'weight' opencv/modules/dnn/src/cuda4dnn/primitives/normalize_bbox.hpp
grep 'nms_iou_threshold' opencv/modules/dnn/src/cuda4dnn/primitives/region.hpp

# Build OpenCV Python Wheel
export ENABLE_CONTRIB=1
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export CMAKE_ARGS="\
       -DCPACK_BINARY_DEB=ON \
       -DBUILD_EXAMPLES=OFF \
       -DBUILD_opencv_python2=OFF \
       -DBUILD_opencv_python3=ON \
       -DBUILD_opencv_java=OFF \
       -DCMAKE_BUILD_TYPE=RELEASE \
       -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DCUDA_ARCH_BIN=8.7 \
       -DCUDA_ARCH_PTX= \
       -DCUDA_FAST_MATH=ON \
       -DCUDNN_INCLUDE_DIR=/usr/include/ \
       -DEIGEN_INCLUDE_PATH=/usr/include/eigen3 \
       -DWITH_EIGEN=ON \
       -DENABLE_NEON=ON \
       -DOPENCV_DNN_CUDA=ON \
       -DOPENCV_ENABLE_NONFREE=ON \
       -DOPENCV_EXTRA_MODULES_PATH=$(pwd)/opencv_contrib/modules \
       -DOPENCV_GENERATE_PKGCONFIG=ON \
       -DOpenGL_GL_PREFERENCE=GLVND \
       -DWITH_CUBLAS=ON \
       -DWITH_CUDA=ON \
       -DWITH_CUDNN=ON \
       -DWITH_GSTREAMER=ON \
       -DWITH_LIBV4L=ON \
       -DWITH_GTK=ON \
       -DWITH_OPENGL=OFF \
       -DWITH_OPENCL=OFF \
       -DWITH_IPP=OFF \
       -DWITH_TBB=ON \
       -DBUILD_TIFF=ON \
       -DBUILD_PERF_TESTS=OFF \
       -DBUILD_TESTS=OFF"

pip3 wheel --verbose .

# Install OpenCV Python Wheel
pip install -U opencv_contrib_python-4.9.0.80-cp312-cp312-linux_aarch64.whl
cd "$PROJECTS_DIR"

########################################
# 13. Building OnnxRuntime-GPU
########################################

echo "=== Step 13: Building OnnxRuntime-GPU ==="

git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# Set Environment Variables
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDACXX="/usr/local/cuda/bin/nvcc"

# Install Packaging Tools
pip3 install -U packaging

# Build OnnxRuntime
./build.sh --config Release --update --parallel --build --build_wheel --build_shared_lib --skip_tests \
--use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu \
--tensorrt_home /usr/lib/aarch64-linux-gnu --cmake_extra_defines CMAKE_CXX_FLAGS="-Wno-unused-variable -I/usr/local/cuda/include" \
--cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="87"

cd "$PROJECTS_DIR"

########################################
# 14. Installing torch2trt and trt-Pose
########################################

echo "=== Step 14: Installing torch2trt and trt-Pose ==="

# Install torch2trt
git clone --recursive https://github.com/NVIDIA-AI-IOT/torch2trt.git
cd torch2trt
python setup.py install

# Build and Install Contributions
cd scripts
bash build_contrib.sh
cd "$PROJECTS_DIR"

# Install trt-Pose
git clone --recursive https://github.com/NVIDIA-AI-IOT/trt_pose.git
cd trt_pose
python3 setup.py develop --user
cd "$PROJECTS_DIR"

########################################
# 15. Example Projects
########################################

echo "=== Step 15: Setting Up Example Projects ==="

# nanoSAM
echo "Setting up nanoSAM..."
git clone https://github.com/NVIDIA-AI-IOT/nanosam.git
cd nanosam

# Download Models
echo "Please download the 'data/models' folder and unzip it in the 'nanosam' directory."
echo "Press Enter after downloading and unzipping the models..."
read -r

# Install Python Dependencies
pip install -U pillow transformers

# Run Demo
echo "Running nanoSAM demo..."
python3 examples/demo_click_segment_track.py

# Fix External Camera Issues (If Applicable)
echo "Fixing external camera issues..."
apt-get update
apt-get upgrade -y
apt-get install -y libffi-dev libglib2.0-0 v4l-utils
echo 'export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libffi.so.7' >> "$USER_HOME/.bashrc"
source "$USER_HOME/.bashrc"

# Verify Camera Functionality
v4l2-ctl --list-devices
apt-get install -y guvcview

# Update demo_click_segment_track.py
echo "Updating demo_click_segment_track.py..."
SCRIPT_PATH="examples/demo_click_segment_track.py"
if [ -f "$SCRIPT_PATH" ]; then
    sed -i 's|cap = cv2.VideoCapture(0)|camera_id = "/dev/video0"\ncap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)|' "$SCRIPT_PATH"
    echo "Updated $SCRIPT_PATH successfully."
else
    echo "Script $SCRIPT_PATH not found. Skipping update."
fi

cd "$PROJECTS_DIR"

# llamaspeak
echo "Setting up llamaspeak..."
git clone --recursive https://github.com/dusty-nv/jetson-containers.git
cd jetson-containers

# Configure Docker Daemon
echo "Configuring Docker daemon for NVIDIA runtime..."
DOCKER_DAEMON_JSON="/etc/docker/daemon.json"

# Backup existing daemon.json
cp "$DOCKER_DAEMON_JSON" "${DOCKER_DAEMON_JSON}.bak"

# Add default-runtime if not present
if grep -q '"default-runtime": "nvidia"' "$DOCKER_DAEMON_JSON"; then
    echo '"default-runtime" already set to "nvidia".'
else
    jq '. + {"default-runtime": "nvidia", "runtimes": {"nvidia": {"path": "nvidia-container-runtime", "runtimeArgs": []}}}' "$DOCKER_DAEMON_JSON" | tee "$DOCKER_DAEMON_JSON"
fi

# Restart Docker
systemctl restart docker

# Initialize Riva
echo "Initializing Riva..."
bash riva_init.sh
bash riva_start.sh

# Clone jetson-containers (already cloned)
cd "$PROJECTS_DIR/jetson-containers"

# Download Models for llamaspeak
echo "Downloading models for llamaspeak..."
mkdir -p data/models/text-generation-webui && cd data/models/text-generation-webui
wget https://huggingface.co/TheBloke/Llama-2-13B-GGUF/resolve/main/llama-2-13b.Q4_0.gguf
cd "$PROJECTS_DIR/jetson-containers"

# Run Text-Web-UI
echo "Running Text-Web-UI..."
./run.sh --workdir /opt/text-generation-webui "$(./autotag text-generation-webui)" \
   python3 server.py --listen --verbose --api \
    --model-dir=/data/models/text-generation-webui \
    --model=llama-2-13b.Q4_0.gguf \
    --loader=llamacpp \
    --n-gpu-layers=128 \
    --n_ctx=4096 \
    --n_batch=4096 \
    --threads=$(( $(nproc) - 2 ))

# Generate SSL Certificates
echo "Generating SSL certificates for llamaspeak..."
cd data
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes -subj '/CN=localhost'
cd "$PROJECTS_DIR/jetson-containers"

# Run llamaspeak
echo "Running llamaspeak..."
./run.sh --workdir=/opt/llamaspeak \
  --env SSL_CERT=/data/cert.pem \
  --env SSL_KEY=/data/key.pem \
  "$(./autotag llamaspeak)" \
  python3 chat.py --verbose

# Open Navigator (Assuming it's a GUI application)
echo "Please open your web browser and navigate to the llamaspeak interface."

cd "$PROJECTS_DIR"

########################################
# 16. Installing Docker with CUDA Support
########################################

echo "=== Step 16: Installing Docker with CUDA Support ==="

# Install Docker
echo "Installing Docker..."
apt-get install -y curl
curl https://get.docker.com | sh
systemctl --now enable docker

# Add NVIDIA Container Toolkit Repository
echo "Adding NVIDIA Container Toolkit repository..."
distribution=$(source /etc/os-release && echo "$ID$VERSION_ID")
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor | tee /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg > /dev/null
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update

# Install NVIDIA Container Toolkit
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Configure other runtimes (containerd and crio)
nvidia-ctk runtime configure --runtime=containerd
systemctl restart containerd

nvidia-ctk runtime configure --runtime=crio
systemctl restart crio

# Add User to Docker Group
echo "Adding user to Docker group..."
groupadd docker || echo "Docker group already exists."
usermod -aG docker "$SUDO_USER"
chmod 666 /var/run/docker.sock
newgrp docker <<EONG
echo "User added to Docker group."
EONG

# Ensure Docker daemon.json has default-runtime set
echo "Ensuring Docker daemon.json has default-runtime set to NVIDIA..."
if grep -q '"default-runtime": "nvidia"' "$DOCKER_DAEMON_JSON"; then
    echo '"default-runtime" is already set to "nvidia".'
else
    jq '. + {"default-runtime": "nvidia", "runtimes": {"nvidia": {"path": "nvidia-container-runtime", "runtimeArgs": []}}}' "$DOCKER_DAEMON_JSON" | tee "$DOCKER_DAEMON_JSON"
    systemctl restart docker
fi

########################################
# 17. Building FFMPEG
########################################

echo "=== Step 17: Building FFMPEG ==="

cd "$PROJECTS_DIR"
wget https://www.ffmpeg.org/releases/ffmpeg-7.0.2.tar.gz
tar -xvzf ffmpeg-7.0.2.tar.gz
cd ffmpeg-7.0.2

# Build and Install libaom
echo "Building and installing libaom..."
mkdir -p ./libaom && cd ./libaom
git clone https://aomedia.googlesource.com/aom
mkdir -p aom_build && cd aom_build
PATH="$USER_HOME/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$USER_HOME/ffmpeg_build" -DENABLE_TESTS=OFF -DENABLE_NASM=on ../aom
PATH="$USER_HOME/bin:$PATH" make
sudo make install
cd "$PROJECTS_DIR/ffmpeg-7.0.2"

# Build and Install SVT-AV1
echo "Building and installing SVT-AV1..."
git -C SVT-AV1 pull 2>/dev/null || git clone https://gitlab.com/AOMediaCodec/SVT-AV1.git
mkdir -p SVT-AV1/build && cd SVT-AV1/build
PATH="$USER_HOME/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$USER_HOME/ffmpeg_build" -DCMAKE_BUILD_TYPE=Release -DBUILD_DEC=OFF -DBUILD_SHARED_LIBS=OFF ..
PATH="$USER_HOME/bin:$PATH" make
sudo make install
cd "$PROJECTS_DIR/ffmpeg-7.0.2"

# Install libdav1d
echo "Installing libdav1d..."
apt-get update
apt-get install -y --no-install-recommends libdav1d-dev

# Configure and Build FFMPEG
echo "Configuring and building FFMPEG..."
PATH="$USER_HOME/bin:$PATH" PKG_CONFIG_PATH="$USER_HOME/ffmpeg_build/lib/pkgconfig" ./configure \
    --prefix="$USER_HOME/ffmpeg_build" \
    --extra-cflags="-I$USER_HOME/ffmpeg_build/include" \
    --extra-cflags="-I/usr/src/jetson_multimedia_api/include/" \
    --extra-ldflags="-L$USER_HOME/ffmpeg_build/lib -L/usr/lib/aarch64-linux-gnu/tegra" \
    --extra-libs="-lpthread -lm -lnvbufsurface -lnvbufsurftransform" \
    --ld="g++" \
    --bindir="$USER_HOME/bin" \
    --enable-shared --disable-doc \
    --enable-libaom --enable-libsvtav1 --enable-libdav1d \
    --enable-nvv4l2dec --enable-libv4l
make -j "$(nproc)"
make install
ldconfig

# Verify FFMPEG Installation
echo "Verifying FFMPEG installation..."
ffmpeg -version
ffmpeg -decoders | grep av1 || echo "AV1 decoder not found."
ffmpeg -decoders | grep h264_nvv4l2dec || echo "H264 nvv4l2dec decoder not found."

echo "=== Setup Complete ==="
echo "Please reboot your system to ensure all changes take effect."
