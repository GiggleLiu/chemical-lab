import numpy as np
from numpy.testing import dec, assert_, assert_raises, assert_almost_equal, assert_allclose
import json

from interface import load_test_data, generate_samples

MAX_LEN = 120
limit = 5000

def test_input1():
    num_sample = 100
    char_list = json.load(open("../data/zinc_char_list.json"))
    n_char = len(char_list)
    test_data,smiles = load_test_data("../data/250k_rndm_zinc_drugs_clean.smi",n_char,MAX_LEN,char_list,limit)
    samples = generate_samples(test_data,num_sample)
    assert_(len(samples) == 100)
    assert_(all([isinstance(s,np.ndarray) and s!='' for s in samples]))
    

    
if __name__ == '__main__':
    test_input1()