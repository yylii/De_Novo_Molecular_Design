import os
import json
import numpy as np
import torch
import dgl
import random
from multiprocessing import Pool


class CacheAndPool:
    def __init__(self, func, processes=1):
        self.results = dict()
        self.func = func
        self.processes = processes

    def __call__(self, args):
        unseen_args = list(set(args).difference(self.results))
        if self.processes <= 1:
            values = map(self.func, unseen_args)
        else:
            with Pool(processes=self.processes) as pool:
                values = pool.map(self.func, unseen_args)
        new_results = dict(zip(unseen_args, values))
        self.results = {**self.results, **new_results}
        return [self.results[arg] for arg in args]

# class CacheAndPool:
#     def __init__(self, func, weight=1, processes=1):
#         self.results = dict()
#         self.func = func
#         self.processes = processes
#         self.weight = weight  # Used to weight the reward part

#     def update_weight(self, new_weight):
#         """Update the weight used in reward calculation."""
#         self.weight = new_weight

#     def __call__(self, args):
#         # Determine which arguments still need processing
#         unseen_args = list(set(args).difference(self.results))

#         # Process unseen arguments
#         if self.processes <= 1:
#             values = map(self.func, unseen_args)
#         else:
#             with Pool(processes=self.processes) as pool:
#                 values = pool.map(self.func, unseen_args)

#         # Cache the new results
#         new_results = dict(zip(unseen_args, values))
#         self.results.update(new_results)

#         print(f"Weight used in cache: {self.weight}")
        
#         # Now apply the weight only to the reward part of each result
#         output = []
#         for arg in args:
#             result = self.results[arg]
#             if isinstance(result, tuple) and len(result) == 2:
#                 reward, prop = result
#                 weighted_reward = self.weight * reward
#                 print(f"Weight used in cache: {self.weight}")

#                 output.append((weighted_reward, prop))
#             else:
#                 # Fallback to old behavior if not a tuple
#                 weighted = (
#                     self.weight * result if isinstance(result, (int, float)) 
#                     else [x * self.weight for x in result]
#                 )
#                 output.append(weighted)

#         return output


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    dgl.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def read_json(path):
    with open(path, 'rt') as f:
        vals = json.load(f)
    return vals


def read_mols(args, epoch):
    suffix = int2str(epoch)
    path = os.path.join(args['mols_dir'], f'sample_{suffix}.json')
    mols = read_json(path)
    return mols


def makedirs(args):
    os.makedirs(args.mols_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)


def dump2json(obj, path):
    with open(path, 'wt') as f:
        json.dump(obj, f, indent=4)


def lmap(f, l):
    return list(map(f, l))


def dmap(f, d):
    return {k: f(v) for k, v in d.items()}


def lzip(*args):
    return list(zip(*args))


def dsuf(s, d):
    return {f'{k}{s}': v for k, v in d.items()}


def int2str(number, length=3):
    assert isinstance(number, int) and number < 10 ** length
    return str(number).zfill(length)
