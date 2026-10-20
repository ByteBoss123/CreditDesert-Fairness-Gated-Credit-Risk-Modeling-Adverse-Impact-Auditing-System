import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
import pytest
from monitoring import psi
from training import split_indices

def test_disjoint_complete_stratified_splits():
    y = pd.Series([0, 1] * 100)
    parts = split_indices(y)
    assert [len(p) for p in parts] == [120,40,40]
    assert len(set(np.concatenate(parts))) == len(y)
    assert all(y.iloc[p].mean() == .5 for p in parts)

def test_psi_detects_out_of_range_shift():
    assert psi(np.arange(100), np.arange(100)) == 0
    assert psi(np.arange(100), np.arange(100)+1000) > 1
    assert psi([1]*20, [2]*20) > 1

@pytest.mark.parametrize('a,b', [([], [1]), ([1], []), ([np.nan], [1])])
def test_psi_rejects_empty_finite_samples(a,b):
    with pytest.raises(ValueError): psi(a,b)
