# Learning to Solve Multiple-TSP with Time Window and Rejection via Deep Reinforcement Learning

## Installation
python 3.7 (tested on 3.7.7 and 3.7.10, recommend 3.7.10)

cuda 10.1 (you can create a docker container for this if your system has higher/lower version of cuda)

Install [pytorch1.7.1](https://pytorch.org/get-started/previous-versions/) with your CUDA version or CPU.

For example, to install `pytorch 1.7.1` with `CUDA 10.1` using `pip`, just run:
```commandline
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -f https://download.pytorch.org/whl/cu101/torch_stable.html
```

Install [torch-geometric](https://github.com/rusty1s/pytorch_geometric) 1.6.3:
```commandline
pip install --upgrade pip
pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
pip install torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
pip install torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu101.html
pip install torch-geometric==1.6.3
```
### Docker Setup
Clone this repo and within the repo folder run the following command.

Create image `manager-worker-mtsptwr_image`:
```commandline
sudo docker build -t manager-worker-mtsptwr_image .
```

Create container `manager-worker-mtsptwr_container` from `manager-worker-mtsptwr_image`, and activate it:
```commandline
sudo docker run --gpus all --name manager-worker-mtsptwr_container -it manager-worker-mtsptwr_image
```