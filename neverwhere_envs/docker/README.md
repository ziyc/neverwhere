# Neverwhere Scene-Making Docker

## Build

From the repository root:

```bash
docker build -t neverwhere-scene-maker -f neverwhere_envs/docker/Dockerfile .
```

## Run

```bash
# Interactive shell
docker run --gpus all -it -v /path/to/datasets:/data neverwhere-scene-maker

# Inside the container, activate the env and run the pipeline:
conda activate nw
python neverwhere_envs/launch.py \
    --scene-name your_scene \
    --dataset-dir /data \
    --gpu-index 0 \
    --gs-type 3dgs
```
