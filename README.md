# MDEC
Mutli-modality Deep Clustering based on AE

## paper: https://iopscience.iop.org/article/10.1088/1742-6596/1684/1/012059/pdf

## Get Started

Make sure you have installed the [Nvidia docker](https://github.com/NVIDIA/nvidia-docker/).

```
docker pull louisdong/mdec
```

```
docker run --gpus all -it --rm -v <yourfilepath>:/workspace --name=mdec 7484808b6558
```

```
cd workspace
sh test.sh
```

