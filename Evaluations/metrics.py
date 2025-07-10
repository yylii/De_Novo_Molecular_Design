import os
import json
from multiprocessing import Pool
import numpy as np
from moses.utils import mapper
from moses.metrics import fraction_passes_filters, internal_diversity, fraction_valid, remove_invalid, fraction_unique 
from moses.metrics.utils import get_mol
from rdkit import Chem

args = {}
args['mols_dir'] = ''

epoch = 200
pool = 1
metrics = {}

def read_json(path):
    with open(path, 'rt') as f:
        vals = json.load(f)
    return vals


def read_mols(args, epoch):
    suffix = int2str(epoch)
    path = os.path.join(args['mols_dir'], f'sample_{suffix}.json')
    mols = read_json(path)
    return mols


def dump2json(obj, path):
    with open(path, 'wt') as f:
        json.dump(obj, f, indent=4)


def int2str(number, length=3):
    assert isinstance(number, int) and number < 10 ** length
    return str(number).zfill(length)

mols = read_mols(args, epoch)

# Define the validation function
def is_valid_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Filter the valid molecules
valid_mols = [smiles for smiles in mols if is_valid_molecule(smiles)]

# Check if there are any invalid molecules
if len(valid_mols) != len(mols):
    print(f"Found {len(mols) - len(valid_mols)} invalid molecules!")


metrics['valid'] = fraction_valid(mols, n_jobs=pool)
gen = remove_invalid(mols, canonize=True)

metrics['unique@{}'.format(1000)] = fraction_unique(valid_mols, 1000, pool)

mols = mapper(pool)(get_mol, mols)
metrics['IntDiv'] = internal_diversity(mols, pool)
metrics['IntDiv2'] = internal_diversity(mols, pool, p=2)
metrics['Filters'] = fraction_passes_filters(mols, pool)

print(metrics)
