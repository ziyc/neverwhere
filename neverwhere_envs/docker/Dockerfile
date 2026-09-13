# Neverwhere Scene-Making Docker Image
# Includes: COLMAP, OpenMVS, Python environment with gsplat
#
# Build from repo root:
#   docker build -t neverwhere-scene-maker -f neverwhere_envs/docker/Dockerfile .
#
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y \
    git cmake build-essential wget \
    python3-dev python3-pip python3.10-dev \
    libboost-iostreams-dev libboost-program-options-dev \
    libboost-system-dev libboost-serialization-dev \
    libboost-filesystem-dev libboost-graph-dev libboost-test-dev \
    libopencv-dev libcgal-dev libcgal-qt5-dev \
    libpng-dev libjpeg-dev libtiff-dev \
    libglu1-mesa-dev freeglut3-dev libglew-dev libglfw3-dev \
    libatlas-base-dev libsuitesparse-dev libabsl-dev \
    libeigen3-dev libfreeimage-dev \
    libgoogle-glog-dev libgflags-dev libsqlite3-dev libmetis-dev \
    qtbase5-dev libqt5opengl5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone https://gitlab.com/libeigen/eigen.git --branch 3.4 && \
    mkdir eigen_build && cd eigen_build && \
    cmake ../eigen && make && make install && \
    cd .. && rm -rf eigen eigen_build

RUN git clone https://github.com/cdcseacave/VCG.git vcglib

RUN git clone https://ceres-solver.googlesource.com/ceres-solver && \
    git clone --branch 20240116.2 \
        https://github.com/abseil/abseil-cpp.git \
        ceres-solver/third_party/abseil-cpp && \
    mkdir ceres_build && cd ceres_build && \
    cmake ../ceres-solver \
          -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF && \
    make -j"$(nproc)" && make install && \
    cd .. && rm -rf ceres-solver ceres_build

RUN git clone https://github.com/cdcseacave/openMVS.git && \
    cd openMVS && git checkout v2.0.1 && \
    mkdir make && cd make && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DVCG_ROOT="/workspace/vcglib" \
             -DEIGEN3_INCLUDE_DIR="/usr/local/include/eigen3" && \
    make -j"$(nproc)" && make install && \
    cd ../.. && rm -rf openMVS

RUN git clone https://github.com/colmap/colmap.git && \
    cd colmap && mkdir build && cd build && \
    cmake .. -DCMAKE_CUDA_ARCHITECTURES=native && \
    make -j"$(nproc)" && make install && \
    cd ../.. && rm -rf colmap

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

RUN conda create -n nw python=3.10 -y
SHELL ["conda", "run", "--no-capture-output", "-n", "nw", "/bin/bash", "-c"]

RUN pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

RUN pip install "setuptools==65.5.0" "wheel==0.38.4" "numpy<2"

COPY pyproject.toml README.md LICENSE /workspace/
COPY neverwhere/ /workspace/neverwhere/
COPY neverwhere_envs/ /workspace/neverwhere_envs/
RUN pip install -e "/workspace[scene-making]"

RUN pip install --no-build-isolation \
        "git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157" \
        "git+https://github.com/nerfstudio-project/gsplat.git@ec3e715f5733df90d804843c7246e725582df10c"

ENV PYTHONPATH=/workspace
ENV COLMAP_BIN=colmap
ENV MUJOCO_GL=egl

WORKDIR /workspace
CMD ["conda", "run", "--no-capture-output", "-n", "nw", "/bin/bash"]
