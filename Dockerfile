FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

# pay attention ARG "cuda_ver" should match base image above
ARG cuda_ver=cu110
# python 3.7.9
ARG miniconda_ver=Miniconda3-py37_4.9.2-Linux-x86_64.sh
ARG project=manager-worker-mtsptwr
ARG username=czhang
ARG password=czhang
ARG torch_ver=1.7.1
ARG torchvision_ver=0.8.2
ARG torchaudio_ver=0.7.2
ARG torch_scatter_ver=2.0.5
ARG torch_sparse_ver=0.6.8
ARG torch_torch_spline_conv_ver=1.2.0
ARG torch_cluster_ver=1.5.8
ARG pyg_ver=1.6.3
ARG matplotlib_ver=3.4.3
ARG ortools_ver=9.0.9048

# Install some basic utilities and create users
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/* \
    # Create a user with home dir /home/${project}
    && useradd -md /home/${project} ${username} \
    # user owns the home dir
    && chown -R ${username} /home/${project} \
    # set user password
    && echo ${username}:${password} | chpasswd \
    # add user to sudoers
    && echo ${username}" ALL=(ALL:ALL) ALL" > /etc/sudoers.d/90-user
# switch to user
USER ${username}
# to home dir
WORKDIR /home/${project}

# download conda installer and save as "~/miniconda.sh"
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/${miniconda_ver} \
    # user owns installer
    && chmod +x ~/miniconda.sh \
    # install conda with name ~/${project}-miniconda-environment;
    # "-p" = path of installed conda env;
    # sublime open https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh to check meaning of -b -p.
    && bash ~/miniconda.sh -b -p ~/${project}-miniconda-environment \
    && rm ~/miniconda.sh
ENV CONDA_AUTO_UPDATE_CONDA=false \
    # add conda to env variables
    PATH=~/${project}-miniconda-environment/bin:$PATH
RUN ~/${project}-miniconda-environment/bin/pip install \
    # install pytorch
    torch==${torch_ver} torchvision==${torchvision_ver} torchaudio==${torchaudio_ver} -f https://download.pytorch.org/whl/${cuda_ver}/torch_stable.html \
    # install pyg and its dependencies
    && ~/${project}-miniconda-environment/bin/pip install --upgrade pip \
    && ~/${project}-miniconda-environment/bin/pip install torch-scatter==${torch_scatter_ver} -f https://pytorch-geometric.com/whl/torch-${torch_ver}+${cuda_ver}.html \
    && ~/${project}-miniconda-environment/bin/pip install torch-sparse==${torch_sparse_ver} -f https://pytorch-geometric.com/whl/torch-${torch_ver}+${cuda_ver}.html \
    && ~/${project}-miniconda-environment/bin/pip install torch-cluster==${torch_cluster_ver} -f https://pytorch-geometric.com/whl/torch-${torch_ver}+${cuda_ver}.html \
    && ~/${project}-miniconda-environment/bin/pip install torch-spline-conv==${torch_torch_spline_conv_ver} -f https://pytorch-geometric.com/whl/torch-${torch_ver}+${cuda_ver}.html \
    && ~/${project}-miniconda-environment/bin/pip install torch-geometric==${pyg_ver} \
    # install matplotlib
    && ~/${project}-miniconda-environment/bin/pip install matplotlib==${matplotlib_ver} \
    # install ortools using pip
    && ~/${project}-miniconda-environment/bin/pip install ortools==${ortools_ver}


