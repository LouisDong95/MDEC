# Multi-View Deep Clustering base on AutoEncoder
## Abstract
In recent years, with the development of deep learning, replacing traditional clustering methods with subspaces extracted by deep neural networks will help better clustering performance. However, due to the instability of unsupervised learning, the features extracted each time are different even if the same data is processed. In order to improve the stability and performance of clustering, we propose a novel unsupervised deep embedding clustering multi-view method, which treats multiple different subspaces as different views through some data expansion methods for the same data. Specifically, our method uses a variety of different deep autoencoders to learn the latent representation of the original data and constrain them to learn different features. Our experimental evaluations on several natural image datasets show that this method has a significant improvement compared to existing methods.

## Get Started

Make sure you have installed the [Nvidia docker](https://github.com/NVIDIA/nvidia-docker/).

```
docker pull louisdong/mdec
docker run --gpus all -it --rm -v <yourfilepath>:/workspace --name=mdec 7484808b6558
cd workspace
sh test.sh
```
## Cite
If you find the code useful in your research, please consider citing our paper:
```
@inproceedings{dong2020multi,
  title={Multi-view deep clustering based on autoencoder},
  author={Dong, Shihao and Xu, Huiying and Zhu, Xinzhong and Guo, XiFeng and Liu, Xinwang and Wang, Xia},
  booktitle={Journal of physics: conference series},
  volume={1684},
  number={1},
  pages={012059},
  year={2020},
  organization={IOP Publishing}
}
```
