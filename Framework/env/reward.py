from functools import partial
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import torch

# class Reward:
#     def __init__(self, property, reward, weight=1.0, preprocess=None):
#         self.property = property
#         self.reward = reward
#         self.weight = weight 
#         self.preprocess = preprocess
    
#     # def update_weight(self, new_weight):
#     #     print(f"new_weight: {self.weight}")
#     #     self.weight = new_weight

#     def __call__(self, input):
#         if self.preprocess:
#             input = self.preprocess(input)
#         property = self.property(input)
#         print(f"Weight used in reward: {self.weight}")
#         reward = self.weight * self.reward(property)
#         #print(reward, property, self.reward(property))
#         return reward, property

class Reward:
    def __init__(self, property_func, reward_func, weight=1.0, preprocess=None):
        self.property = property_func         # e.g., MolFromSmiles or similar
        self.reward_func = reward_func        # raw reward function
        self.weight = weight
        self.preprocess = preprocess

    def update_weight(self, new_weight):
        self.weight = new_weight

    def __call__(self, input):
        if self.preprocess:
            if isinstance(input, list):
                # Apply preprocessing to each individual molecule
                input = [self.preprocess(i) for i in input]
            else:
                input = self.preprocess(input)

        prop = self.property(input)[0]

        print(f"prop = {prop}")

        if prop is None:
            return 0.0, 0.0  # Handle invalid molecules

        raw_reward = self.reward_func(prop)
        print(raw_reward)
        weighted_reward = self.weight * raw_reward

        print(f"[Reward] weight={self.weight}, raw={raw_reward}, final={weighted_reward}")

        return weighted_reward, prop

def identity(x):
    return x

def docking(x):
    if x < 0:
        x = -x
    else:
        x = 0
    return x

def fraction(x):
    return 10*(1/x)

def ReLU(x):
    return max(x, 0)


def HSF(x):
    return float(x > 0)


class OutOfRange:
    def __init__(self, lower=None, upper=None, hard=True):
        self.lower = lower
        self.upper = upper
        self.func = HSF if hard else ReLU

    def __call__(self, x):
        y, u, l, f = 0, self.upper, self.lower, self.func
        if u is not None:
            y += f(x - u)
        if l is not None:
            y += f(l - x)
        return y


class PatternFilter:
    def __init__(self, patterns):
        self.structures = list(filter(None, map(Chem.MolFromSmarts, patterns)))

    def __call__(self, molecule):
        return int(any(molecule.HasSubstructMatch(struct) for struct in self.structures))


def MolLogP(m):
    return rdMolDescriptors.CalcCrippenDescriptors(m)[0]

def MolSAS(m):
    return sascorer.calculateScore(m)

