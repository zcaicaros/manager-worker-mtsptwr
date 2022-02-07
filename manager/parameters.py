import argparse
parser = argparse.ArgumentParser(description='mtsptwr-manager-training-parameters')

# env parameters
parser.add_argument('--n_agent', type=int, default=5)
parser.add_argument('--n_nodes', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--iteration', type=int, default=10000)
parser.add_argument('--sh_or_mh', type=str, default='MH')  # 'MH'-MultiHead, 'SH'-SingleHead
parser.add_argument('--node_embedding_type', type=str, default='gin')  # 'mlp', 'gin'
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--beta', type=int, default=10)
args = parser.parse_args()