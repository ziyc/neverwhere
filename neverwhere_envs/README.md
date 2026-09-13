# Neverwhere Toolchain

## Install Dependencies

### Install Python Environment
```bash
conda create -n neverwhere python=3.10 -y
conda activate neverwhere

# PyTorch first (match your CUDA version)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U pip "setuptools==65.5.0" "wheel==0.38.4"

# simulator + scene-making Python dependencies
pip install -e ".[scene-making]"

# CUDA extensions (compiled for your GPU, need nvcc)
pip install --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157"
pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@ec3e715f5733df90d804843c7246e725582df10c"
```

> **Note:** If your system gcc is too new for nvcc (e.g. gcc-13 with CUDA 12.2), install gcc-12 via conda:
> ```bash
> conda install -c conda-forge gcc_linux-64=12 gxx_linux-64=12
> export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
> export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
> export CUDAHOSTCXX=$CXX
> ```
> Then re-run the `pip install --no-build-isolation` commands above.

Alternatively build the Docker image in `neverwhere_envs/docker/`, which installs COLMAP, OpenMVS and all Python dependencies.

### Setting Up COLMAP Environment

We use [COLMAP](https://github.com/colmap/colmap) to generate camera poses and point clouds.

<details>
<summary>COLMAP Installation Guide</summary>

```bash
# Install dependencies
sudo apt-get update -qq && sudo apt-get install -qq
sudo apt-get install \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev

# Clone COLMAP
git clone https://github.com/colmap/colmap.git

# Build COLMAP
cd colmap
mkdir build
cd build
cmake ..
make -j
sudo make install
```
</details>

### Setting Up OpenMVS Environment

We use [OpenMVS](https://github.com/cdcseacave/openMVS) to generate high-quality meshes for:
- Initializing 3D Gaussians for high-quality 3DGS Reconstruction
- Creating accurate collision geometry for the environment

Follow the installation instructions in the [OpenMVS Building Guide](https://github.com/cdcseacave/openMVS/wiki/Building).

<details>
<summary>Click here for a modified installation guide</summary>

#### Prepare and empty machine for building:
```bash
sudo apt-get update -qq && sudo apt-get install -qq
sudo apt-get -y install git cmake libpng-dev libjpeg-dev libtiff-dev libglu1-mesa-dev
main_path=$(pwd)
```

#### Eigen (Required)
```bash
git clone https://gitlab.com/libeigen/eigen.git --branch 3.4
mkdir eigen_build && cd eigen_build
cmake ../eigen
make && sudo make install
cd ..
```

#### Boost (Required)
```bash
sudo apt-get -y install libboost-iostreams-dev libboost-program-options-dev libboost-system-dev libboost-serialization-dev
```

#### OpenCV (Required)
```bash
sudo apt-get -y install libopencv-dev
```

#### CGAL (Required)
```bash
sudo apt-get -y install libcgal-dev libcgal-qt5-dev
```

#### VCGLib (Required)
```bash
git clone https://github.com/cdcseacave/VCG.git vcglib
```

#### Python (Required)
```bash
sudo apt-get install -y python3-dev python3-pip python3.10-dev
```

#### Ceres (Optional)
```bash
sudo apt-get -y install libatlas-base-dev libsuitesparse-dev
git clone https://ceres-solver.googlesource.com/ceres-solver ceres-solver
mkdir ceres_build && cd ceres_build
cmake ../ceres-solver/ -DMINIGLOG=ON -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
make -j2 && sudo make install
cd ..
```

#### GLFW3 (Optional)
```bash
sudo apt-get -y install freeglut3-dev libglew-dev libglfw3-dev
```

#### OpenMVS
```bash
git clone https://github.com/cdcseacave/openMVS.git
cd openMVS
git checkout develop
mkdir make && cd make
cmake .. -DCMAKE_BUILD_TYPE=Release -DVCG_ROOT="$main_path/vcglib"
```

#### Install OpenMVS library:
```bash
make -j4
sudo make install
```

</details>

## Create Your Own Data

Follow these simple steps to add a new environment to the Neverwhere project:

### Step 1: Obtaining Polycam Scans

1. Use the Polycam app to capture a 3D scan of your desired environment.
2. Export the raw data from Polycam.

### Step 2: Setting Up the Environment Structure

1. Create a new directory for your environment under `projectpath/neverwhere_envs/datasets`:
   ```
   mkdir projectpath/neverwhere_envs/datasets/your_environment_name
   ```

2. Place the Polycam raw data in a `polycam` folder within your new environment directory:
   ```
   mkdir projectpath/neverwhere_envs/datasets/your_environment_name/polycam
   ```
   Copy your Polycam raw data into this `polycam` folder.

3. Your new environment structure should look like this:
   ```
   projectpath/neverwhere_envs/datasets/
   └── your_environment_name/
       └── polycam/
           └── [Polycam raw data files]
   ```

4. run the all-in-one script to process the polycam raw data to desired format:
   ```bash
       python neverwhere_envs/launch.py \
        --scene-name your_scene_name \
        --dataset-dir /path/to/your/dataset \
        --gpu-index your_gpu_index \
        --downsample your_downsample_rate \ # downsample rate for image number
        --downsample-threshold your_downsample_threshold \ # downsample threshold for image number, if the image number is less than the threshold, the image will not be downsampled
        --depth-keys depth confidence \
        --gs-type 3dgs \
        --refine-mesh
   ```

    after running the script, you will get the following file structure:

    ```
    /path/to/your/dataset/
        ├── scene000_name/
        │   ├── 3dgs/ # the 3dgs model
        │   ├── colmap/ # the colmap result
        │   ├── geo2d/ # the geometry data of the scene from openmvs, used for 3dgs regularization
        │   ├── geometry/ # the mesh and point clouds of the scene
        │   ├── images/ # the images of the scene, could be downsampled if downsample is set, used for making the scene
        │   ├── openmvs/ # the openmvs result
        │   ├── polycam/ # the raw data from polycam
        │   └── scene000_name.xml # the scene description file, generated after step 3 is done
        └── ...
    ```

### Step 3: Manually label the scene

```bash
export NEVERWHERE_DATASET_ROOT=/path/to/your/dataset
export NEVERWHERE_SCENE_NAME=your_scene_name
python neverwhere_envs/tools/label.py
```
The tool opens a Vuer web viewer (default port 9072). Follow the on-screen steps: orient the mesh, place the two alignment markers, scale/crop, add waypoints, save. It writes `geometry/collision_tf.json` and `nw-go1-<scene>.xml` / `nw-go2-<scene>.xml` into the scene directory and into `neverwhere/tasks/`. Add a line for the scene to `neverwhere/scenes.txt` to register it with the simulator. `tools/vis_xml.py` previews a labeled XML and `tools/modify_waypoints.py` edits waypoints of an existing scene.

