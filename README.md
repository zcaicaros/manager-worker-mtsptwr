# Learning to Solve Multiple-TSP with Time Window and Rejection via Deep Reinforcement Learning


This repository is the official PyTorch implementation of the algorithms in the following paper: 

Rongkai Zhang, Cong Zhang, Zhiguang Cao, Wen Song, Puay Siew Tan, Jie Zhang, Bihan Wen, and Justin Dauwels. Learning to Solve Multiple-TSP with Time Window and Rejection via Deep Reinforcement Learning. IEEE Transactions on Intelligent Transportation Systems, 2022. [\[PDF\]](http://arxiv.org/abs/2209.06094)


Paper reference:
```
@ARTICLE{9901466,
  author={Zhang, Rongkai and Zhang, Cong and Cao, Zhiguang and Song, Wen and Tan, Puay Siew and Zhang, Jie and Wen, Bihan and Dauwels, Justin},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Learning to Solve Multiple-TSP With Time Window and Rejections via Deep Reinforcement Learning}, 
  year={2022},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TITS.2022.3207011}}
```

If you have any issues running the code, please let me know.

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

## Training the model
Change parameters type in ```./manager/parameters.py``` file and run:
```
cd ./manager/
python3 train.py
```

In this repo, we have provided the trained worker with different sizes, stored in ```./trained/workers```

If you want to train the worker with different sizes, please refer to [code](pending) and [paper](https://ieeexplore.ieee.org/abstract/document/9207026/).

## Testing the model
run:
```
cd ./manager/
python3 test.py
```

The parameters for testing is in ```./manager/test.py```. You can adjust accordingly.

