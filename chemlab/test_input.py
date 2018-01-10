import numpy as np
from numpy.testing import dec, assert_, assert_raises, assert_almost_equal, assert_allclose

from .interface import chemical_string, generate_samples

def test_input1():
    num_sample = 100
    samples = generate_samples(num_sample)
    assert_(len(samples) == 100)
    assert_(all([isinstance(s, str) and s!='' for s in samples]))
