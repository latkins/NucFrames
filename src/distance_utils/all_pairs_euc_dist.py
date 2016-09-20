import numpy as np
from numba import cuda
import math
import h5py
import logging
import time
import glob
import itertools
import os
from tqdm import tqdm


########################################################################
def nuc_dist_pairs(nuc_file, chrm_a, chrm_b):
    f = h5py.File(nuc_file, 'r')
    struct = f["structures"]["0"]

    coords_set = struct["coords"]
    particles_set = struct["particles"]

    chrms_arr = []
    coords_arr = []
    particles_arr = []

    for chrm in (chrm_a, chrm_b):
        coords = coords_set[chrm]
        particles = particles_set[chrm]["positions"]
        if coords.shape[1] != particles.shape[0]:
            raise TypeError

    a_coords = coords_set[chrm_a]
    b_coords = coords_set[chrm_b]

    coords_arr = np.hstack([a_coords, b_coords])

    now = time.time()
    out_arr = cuda_all_euc_dists(coords_arr)
    done = time.time()
    logging.info("chrm {} vs chrm {} done in {}s".format(chrm_a, chrm_b,
                                                         done - now))

    a_len = a_coords.shape[1]

    a_to_b = out_arr[0:a_len, a_len:, :]
    b_to_a = out_arr[a_len:, :a_len, :]

    return (a_to_b, b_to_a)


########################################################################
def cuda_all_euc_dists(coords_arr):
    """
    Pass an array of shape (models, side, dimensions), return all the
    euclidean distances between each coord in each model.
    """
    num_threads = cuda.get_current_device().WARP_SIZE
    models, side, _ = coords_arr.shape

    out_arr = np.zeros((side, side, models))

    # What block dims?
    tpb = (32, 16, 2)

    # Given threads per block, what should blocks per grid be?
    bpg = _grid_dim(out_arr, tpb)

    cuda_all_euc_dists_inner[bpg, tpb](coords_arr, out_arr)
    return (out_arr)


def _grid_dim(coords_arr, tpb):
    size = (coords_arr.shape)
    bpg = np.ceil(np.array(size, dtype=np.float) / tpb).astype(np.int).tolist()

    return (tuple(bpg))


########################################################################
@cuda.jit()
def cuda_all_euc_dists_inner(coords_arr, out_arr):
    x, y, m = cuda.grid(3)

    num_models, num_beads, num_dims = coords_arr.shape
    if x < num_beads and y < num_beads and m < num_models:
        acc = 0.0
        for d in range(num_dims):
            acc += (coords_arr[m, x, d] - coords_arr[m, y, d]) ** 2
        out_arr[x, y, m] = math.sqrt(acc)
