import torch
import os
import json
import torch.nn.functional as F


def load_problem(name):
    from problems import TSP
    problem = {
        'tsp': TSP
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args


def load_model(path, dev):
    from nets.attention_model import AttentionModel
    model_filename = path
    path = os.path.dirname(model_filename)
    args = load_args(os.path.join(path, 'args.json'))
    problem = load_problem(args['problem'])

    model = AttentionModel(
        args['embedding_dim'],
        args['hidden_dim'],
        problem,
        n_encode_layers=args['n_encode_layers'],
        mask_inner=True,
        mask_logits=True,
        normalization=args['normalization'],
        tanh_clipping=args['tanh_clipping'],
        checkpoint_encoder=args.get('checkpoint_encoder', False),
        shrink_size=args.get('shrink_size', None)
    )
    # Overwrite model parameters by parameters to load
    if dev == 'cpu':
        model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')).get('model', {}))
    else:
        model.load_state_dict(torch.load(model_filename).get('model', {}))
        model.to(dev)
    # model.load_state_dict(
    #     {**model.state_dict(), **torch.load(model_filename, map_location=torch.device(dev)).get('model', {})}
    # )
    # model.load_state_dict(torch.load(model_filename, map_location=torch.device(dev)).get('model', {}))

    model.eval()
    return model


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(inner_func,helper,get_cost_func, input, batch_rep=1, iter_rep=1):
    """
    :param input: (batch_size, graph_size, node_dim) input node features
    :return:
    """
    input = do_batch_rep(input, batch_rep)

    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)
        
        pi_,c,t = helper(input,pi)
        # pi.view(-1, batch_rep, pi.size(-1))
        cost, mask = get_cost_func(input, pi_, c,t)
        
        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi_.view(batch_rep, -1, pi_.size(-1)).transpose(0, 1))

    max_length = max(pi_.size(-1) for pi_ in pis)
    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat(
        [F.pad(pi_, (0, max_length - pi_.size(-1))) for pi_ in pis],
        1
    )  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]

    return minpis, mincosts
