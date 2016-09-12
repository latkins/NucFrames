import numpy as np
import os
import h5py
from utils import bp_to_idx

class PopNucFrame(object):

  def __init__(self, path, chrms=None, bin_size=None, chrm_limit_dict=None):
    if bin_size is None:
      bin_size = 100e3

    self.bin_size = bin_size

    if chrms is None:
      chrms = [str(x) for x in range(1, 20)] + ["X"]

    if not chrm_limit_dict:
      chrm_limit_dict = {chrm: (None, None) for chrm in chrms}

    self.store = h5py.File(path, 'r')

    self.trans = PopTransGroup(self.store, chrms, bin_size, chrm_limit_dict)
    self.cis = PopChromosomeGroup(self.store, chrms, bin_size, chrm_limit_dict)


class PopChromosomeGroup(object):
  def __init__(self, store, chromosomes, bin_size, chrm_limit_dict):
    self.bin_size = bin_size
    self.store = store
    self.chrms = chromosomes
    self.chrm_limit_dict = chrm_limit_dict

  def __getitem__(self, chrm):
    return (PopChromosome(self.store, chrm, self.bin_size, self.chrm_limit_dict))

  def __iter__(self):
    for chrm in self.chrms:
      yield (PopChromosome(self.store, chrm, self.bin_size, self.chrm_limit_dict))


class PopTransGroup(object):
  def __init__(self, store, chromosomes, bin_size, chrm_limit_dict):
    self.bin_size = bin_size
    self.store = store
    self.chrms = chromosomes
    self.chrm_limit_dict = chrm_limit_dict

  def __getitem__(self, chrm_tuple):
    chrm_a, chrm_b = chrm_tuple
    return (PopTrans(self.store, chrm_a, chrm_b, self.bin_size, self.chrm_limit_dict))

  def __iter__(self):
    for (chrm_a, chrm_b) in product(self.chrms, repeat=2):
      yield (PopTrans(self.store, chrm_a, chrm_b, self.bin_size, self.chrm_limit_dict))


class PopChromosome(object):

  def __init__(self, store, chrm, bin_size, chrm_limit_dict):
    self.store = store
    self.chrm = chrm
    self.bin_size = bin_size

    lower_bp_lim, upper_bp_lim = chrm_limit_dict[chrm]

    cell_name = list(self.store["contacts"]["original"].keys())[0]
    self.contacts_store = self.store["contacts"]["original"][cell_name][chrm][chrm]
    contact_data = self.contacts_store[:].T

    if lower_bp_lim is None:
      min_bp = min(np.min(contact_data[:, 0]), np.min(contact_data[:, 1]))
      lower_bp_lim = int(self.bin_size * np.floor_divide(min_bp, self.bin_size))

    if upper_bp_lim is None:
      max_bp = max(np.max(contact_data[:, 0]), np.max(contact_data[:, 1]))
      upper_bp_lim = int(self.bin_size * np.ceil(max_bp / self.bin_size))

    self.lower_bp_lim = lower_bp_lim
    self.upper_bp_lim = upper_bp_lim

    self.contact_mat, self.positions = self.make_contact_matrix()

  def make_contact_matrix(self):
    size = (self.upper_bp_lim - self.lower_bp_lim) / self.bin_size

    positions = np.arange(self.lower_bp_lim,
                          self.upper_bp_lim,
                          self.bin_size,
                          dtype=np.int32)

    contact_data = self.contacts_store[:].T

    bps_a = contact_data[:, 0].astype(np.int32)
    bps_b = contact_data[:, 1].astype(np.int32)
    counts = contact_data[:, 2]

    idxs_a, valid_a = bp_to_idx(bps_a, positions, self.bin_size)
    idxs_b, valid_b = bp_to_idx(bps_b, positions, self.bin_size)

    valid = np.logical_and(valid_a, valid_b)

    idxs_a = idxs_a[valid]
    idxs_b = idxs_b[valid]
    counts = counts[valid]

    mat = np.zeros((size, size), dtype=np.int64)

    mat[idxs_a, idxs_b] = counts

    mat[idxs_b, idxs_a] = counts

    return(mat, positions)


class PopTrans(object):

  def __init__(self, store, chrm_a, chrm_b, bin_size, chrm_limit_dict):
    self.store = store
    self.bin_size = bin_size

    cell_name = list(self.store["contacts"]["original"].keys())[0]
    try:
      self.contacts_store = self.store["contacts"]["original"][cell_name][chrm_a][chrm_b]
      self.contact_data = self.contacts_store[:].T
    except KeyError:
      self.contacts_store = self.store["contacts"]["original"][cell_name][chrm_b][chrm_a]
      contact_data = self.contacts_store[:].T
      a = contact_data[:, 1]
      b = contact_data[:, 0]
      count = contact_data[:, 2]

      self.contact_data = np.hstack([a[:, None], b[:, None], count[:, None]])

    pop_chrm_a = PopChromosome(store, chrm_a, self.bin_size, chrm_limit_dict)

    self.lower_bp_lim_a = pop_chrm_a.lower_bp_lim
    self.upper_bp_lim_a = pop_chrm_a.upper_bp_lim

    pop_chrm_b = PopChromosome(store, chrm_b, self.bin_size, chrm_limit_dict)

    self.lower_bp_lim_b = pop_chrm_b.lower_bp_lim
    self.upper_bp_lim_b = pop_chrm_b.upper_bp_lim

    self.chrm_a = pop_chrm_a
    self.chrm_b = pop_chrm_b

    self.contact_mat = self.make_contact_matrix()

  def make_contact_matrix(self):
    size_a = (self.upper_bp_lim_a - self.lower_bp_lim_a) / self.bin_size
    size_b = (self.upper_bp_lim_b - self.lower_bp_lim_b) / self.bin_size

    positions_a = np.arange(self.lower_bp_lim_a,
                            self.upper_bp_lim_a,
                            self.bin_size,
                            dtype=np.int32)

    positions_b = np.arange(self.lower_bp_lim_b,
                            self.upper_bp_lim_b,
                            self.bin_size,
                            dtype=np.int32)


    bps_a = self.contact_data[:, 0].astype(np.int32)
    bps_b = self.contact_data[:, 1].astype(np.int32)
    counts = self.contact_data[:, 2]

    idxs_a, valid_a = bp_to_idx(bps_a, positions_a, self.bin_size)
    idxs_b, valid_b = bp_to_idx(bps_b, positions_b, self.bin_size)

    valid = np.logical_and(valid_a, valid_b)

    idxs_a = idxs_a[valid]
    idxs_b = idxs_b[valid]
    counts = counts[valid]

    mat = np.zeros((size_a, size_b), dtype=np.int64)

    mat[idxs_a, idxs_b] = counts
    if self.chrm_a.chrm == self.chrm_b.chrm:
      mat[idxs_b, idxs_a] = counts

    return(mat)


if __name__ == "__main__":
  store = h5py.File("/home/lpa24/dev/cam/data/edl_chromo/mm10/population_nuc/SLX-7671_hapsort_pop_mm10.nuc", 'r')
  chrm = "12"
  bin_size = 100e3

  pnf = PopNucFrame("/mnt/SSD/population_nuc/SLX-7671_hapsort_pop_mm10.nuc")

  print(pnf.cis["12"].contact_mat.shape)
  print(pnf.trans["1", "2"].contact_mat.shape)

  #p = PopChromosome(store, chrm, bin_size)
  #p = PopTrans(store, "12", "1", bin_size)
