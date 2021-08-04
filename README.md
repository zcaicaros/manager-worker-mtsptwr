# Learning to Solve Multiple-TSP with Time Window and Rejection via Deep Reinforcement Learning

## Installation
python 3.7 (tested on 3.7.7 and 3.7.10, recommend 3.7.10)


Install [pytorch1.7.1](https://pytorch.org/get-started/previous-versions/) with your CUDA version or CPU.

For example, to install `pytorch 1.7.1` with `CUDA 10.1`, just run:
````
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
````

Install [torch-geometric](https://github.com/rusty1s/pytorch_geometric) 1.6.3:
````
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.1+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+${CUDA}.html
pip install torch-geometric==1.6.3
````
where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101`, `cu102`, or `cu110` depending on your PyTorch installation.

