import argparse
import json
import logging
import os
from random import shuffle

import numpy as np

MAX_LEN = 120
PADDING = 'right'

def load_test_data(test_path, n_chars, max_len, char_list, limit=None):
    '''
    structure to string.
    '''
    with open(test_path, 'r') as f:
        smiles = f.readlines()
    smiles = [s.strip() for s in smiles]
    if limit is not None:
        smiles = smiles[:limit]
    print('Training set size is', len(smiles))
    smiles = [smile_convert(i) for i in smiles if smile_convert(i)]
    print('Training set size is {}, after filtering to max length of {}'.format(len(smiles), max_len))
    shuffle(smiles)

    print(('total chars:', n_chars))

    cleaned_data = np.zeros((len(smiles), max_len, n_chars), dtype=np.float32)

    char_lookup = dict((c, i) for i, c in enumerate(char_list))

    for i, smile in enumerate(smiles):
        for t, char in enumerate(smile):
            cleaned_data[i, t, char_lookup[char]] = 1

    return cleaned_data,smiles
    
def smile_convert(string):
    if len(string) < MAX_LEN:
        if PADDING == 'right':
            return string + " " * (MAX_LEN - len(string))
        elif PADDING == 'left':
            return " " * (MAX_LEN - len(string)) + string
        elif PADDING == 'none':
            return string


def generate_samples(test_data,num_sample=100):
    '''
    generate a batch of data as input.
    '''
    shuffle(test_data)
    gen_data = test_data[:num_sample]
    return gen_data
    


