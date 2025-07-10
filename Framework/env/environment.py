from rdkit import Chem
from operator import methodcaller, itemgetter
from functools import partial

import numpy as np

from ffreed.env.state import State
from ffreed.utils import dmap, lmap, dsuf
from ffreed.env.utils import connect_mols


class Environment(object):
    def __init__(self, *, atom_vocab, bond_vocab, frag_vocab, timelimit=4,
                 rewards, starting_smile='c1([*:1])c([*:2])ccc([*:3])c1', 
                 action_size=40, fragmentation='crem', **kwargs):
        self.atom_vocab = atom_vocab
        self.frag_vocab = frag_vocab
        self.bond_vocab = bond_vocab

        assert fragmentation in ['crem', 'brics']
        self.fragmentation = fragmentation
        if fragmentation == 'crem':
            attach_vocab = ['*']
        elif fragmentation == 'brics':
            attach_vocab = [f"[{i}*]" for i in range(1, 17)]
            attach_vocab.remove("[2*]")
 
        self.attach_vocab = attach_vocab
        self.num_att_types = len(attach_vocab)
        # atom representation have 18 meta features
        self.atom_dim = len(atom_vocab) + len(attach_vocab) + 18
        self.bond_dim = len(self.bond_vocab)
        self.action_size = action_size
        self.starting_smile = starting_smile
        self.state_args = {
            'fragmentation': fragmentation,
            'atom_dim': self.atom_dim,
            'bond_dim': self.bond_dim,
            'atom_vocab': atom_vocab,
            'bond_vocab': bond_vocab,
            'attach_vocab': attach_vocab
        }
        self.num_steps = 0
        self.state = State(starting_smile, self.num_steps, **self.state_args)
        self.rewards = rewards
        self.timelimit = timelimit
        self.fragments = [State(frag, 0, **self.state_args) for frag in self.frag_vocab]
        num_att = [len(frag.attachments) for frag in self.fragments]
        S, T = len(self.state.attachments), timelimit
        N, M = len(frag_vocab), max(num_att)
        self.actions_dim = (S + T * (M - 1), N, M)
    
    def reward_batch(self, smiles, w1):

        print(f"\nProcessing SMILES: {smiles}")
        # Set weights dynamically
        w2 = 1.0 - w1

        # Update weights for each objective
        self.rewards['DockingScore'].update_weight(w1)
        self.rewards['SAS'].update_weight(w2)

        # Debug: print the current weights
        print(f"Updated Weights:")
        for name, reward_obj in self.rewards.items():
            print(f"  {name}: weight = {reward_obj.weight}")

        # Initialize output containers
        total_reward = []
        reward_components = {name: [] for name in self.rewards}
        property_components = {name: [] for name in self.rewards}

        # Process each SMILES individually
        for smile in smiles:
            smile_total_reward = 0.0

            print(f"\nProcessing SMILES: {smile}")

            print(f"\nReward item: {self.rewards.items()}")

            for name, reward_obj in self.rewards.items():

                print(f"\nname: {name}")
                print(f"\nreward_obj: {reward_obj}")
                print(f"\nreward_obj_smile: {reward_obj([smile])}")

                reward, prop = reward_obj([smile])

                # Log reward details
                print(f"  {name}: Reward = {reward:.4f}, Property = {prop}")

                reward_components[name].append(reward)
                property_components[name].append(prop)

                smile_total_reward += reward

            total_reward.append(smile_total_reward)
            print(f"  Total Combined Reward: {smile_total_reward:.4f}")

        # Create final rewards and properties dicts
        rewards = {
            'Reward': total_reward,
            **reward_components
        }
        properties = property_components

        return {'Reward': total_reward, **reward_components, **property_components}


    # def reward_batch(self, smiles, w1=0.5):
    #     w2 = 1.0 - w1
    #     # Update the weights for the Reward class instances
    #     for reward_obj in self.rewards:
    #         if reward_obj.property.__name__ == "DockingScore":
    #             reward_obj.update_weight(w1)  # Update the weight for the DockingScore
    #         elif reward_obj.property.__name__ == "SAS":
    #             reward_obj.update_weight(w2)  # Update the weight for the SAS

    #     objectives = dmap(methodcaller('__call__', smiles), self.rewards)
    #     rewards = dsuf('Reward', dmap(partial(lmap, itemgetter(0)), objectives))
    #     properties = dsuf('Property', dmap(partial(lmap, itemgetter(1)), objectives))
    #     rewards['Reward'] = np.sum(list(rewards.values()), axis=0).tolist()

    #     return {**rewards, **properties}

    def step(self, action):
        self.attach_fragment(action)
        self.num_steps += 1
        terminated = not self.state.attachments
        truncated = self.num_steps >= self.timelimit
        # reward calculated only for terminal states in "reward_batch" call
        reward = 0.
        state = self.state
        info = dict()
        return state, reward, terminated, truncated, info

    def reset(self, starting_smile=None):
        self.num_steps = 0
        self.state = State(self.starting_smile, self.num_steps, **self.state_args)
        return self.state

    def attach_fragment(self, action):
        a1, a2, a3 = action
        mol = self.state.molecule
        frag_state = self.fragments[a2]
        frag = frag_state.molecule
        mol_attachments = self.state.attachment_ids
        mol_attachment = mol.GetAtomWithIdx(mol_attachments[a1])
        frag_attachments = frag_state.attachment_ids
        frag_attachment = frag.GetAtomWithIdx(frag_attachments[a3])
        new_mol = connect_mols(mol, frag, mol_attachment, frag_attachment)
        self.state = State(Chem.MolToSmiles(new_mol), self.num_steps + 1, **self.state_args)
