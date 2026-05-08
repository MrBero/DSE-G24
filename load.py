import numpy as np
import pandas as pd

def load_one_cfd(csv_path):
    """
    Load one CFD output file with columns x, y, u, v.
    
    Returns four 1D NumPy arrays: x, y, u, v
    """
    # TODO: read the CSV
    df = pd.read_csv(csv_path, sep=",", skipinitialspace=True)
    # TODO: sort by (x, y) for consistent cell ordering
    df = df.sort_values(by=["x", "y"]).reset_index(drop=True)
    # TODO: extract columns
    x = df["x"].values
    y = df["y"].values
    u = df["u"].values
    v = df["v"].values
    # TODO: return
    return x, y, u, v

def load_all_cfds(path_lst, U_inlet_lst, alpha_lst):
    assert len(path_lst) == U_inlet_lst == alpha_lst, "there should be a same amount of cfds as velocity inlets and angles"

    cfd1_tupel = load_one_cfd(path_lst[0])
    cfd1_shape = cfd1_tupel.shape()

    X_ensemble = np.zeros(2 * cfd1_shape[0] +2, len(path_lst)) # two cfd variables provide useful information, and two differing initial conditions
    X_ensemble[:,0] = np.vstack(cfd1_tupel[2], cfd1_tupel[3])
    for i in range(1, len(path_lst)):
        