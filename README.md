# Learning to Solve Multiple-TSP with Time Window and Rejection via Deep Reinforcement Learning

## Requirement
Ubuntu 20.04 LTS 

[Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

python 3.7.x

## Manual Installation


cuda 11.0 + pytorch1.7.1

```commandline
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -f https://download.pytorch.org/whl/cu110/torch_stable.html
```

Install other dependencies: pyg, matplotlib, ortools
```commandline
pip install --upgrade pip
pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-geometric==1.6.3
pip install matplotlib==3.4.3
pip install ortools==9.0.9048
```
## Docker Setup (preferred)
Clone this repo and within the repo folder run the following command.

Create image `manager-worker-mtsptwr_image`:
```commandline
sudo docker build -t manager-worker-mtsptwr-image .
```

Create container `manager-worker-mtsptwr_container` from `manager-worker-mtsptwr_image`, and activate it:
```commandline
sudo docker run --gpus all --name manager-worker-mtsptwr-container -it manager-worker-mtsptwr-image
```