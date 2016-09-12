import numpy as np
import os
import h5py
from Chromosome import Chromosome
from utils import bp_to_idx


class Trans(object):
  """
  Class to represent the relationship between two chromosomes of a single cell.
  Should work with both true trans, and cis.
  """
  def __init__(self, store, chrm_a, chrm_b, chrm_limit_dict):
    """
    Follows the HDF5 layout of NucFrame
    Contains cis-distances, coordinates, depths.
    Each property obeys the chromosome limits it has been passed, meaning
    indexes are directly comparable.
    """
    self.store = store
    self.chrm_limit_dict = chrm_limit_dict

    lower_a, upper_a = self.chrm_limit_dict[chrm_a]
    self.chrm_a = Chromosome(self.store, chrm_a, lower_a, upper_a)
    lower_b, upper_b = self.chrm_limit_dict[chrm_b]
    self.chrm_b = Chromosome(self.store, chrm_b, lower_b, upper_b)

  @property
  def cell(self):
    return(self.store["name"][()])

  @property
  def bp_pos(self):
    return(self.chrm_a.bp_pos, self.chrm_b.bp_pos)

  @property
  def positions(self):
    return(self.chrm_a.positions, self.chrm_b.positions)

  @property
  def expr_contacts(self):

    try:
      contact_data = self.store["expr_contacts"][self.chrm_a.chrm][self.chrm_b.chrm]
      bps_a = contact_data[:, 0].astype(np.int32)
      bps_b = contact_data[:, 1].astype(np.int32)
      counts = contact_data[:, 2]
    except KeyError:
      # Trans contacts may be empty.
      try:
        contact_data = self.store["expr_contacts"][self.chrm_b.chrm][self.chrm_a.chrm]
        bps_a = contact_data[:, 1].astype(np.int32)
        bps_b = contact_data[:, 0].astype(np.int32)
        counts = contact_data[:, 2]
      except KeyError:
        bps_a = np.array([], dtype=np.int32)
        bps_b = np.array([], dtype=np.int32)
        counts = np.array([])


    size_a = self.chrm_a.bp_pos.shape[0]
    size_b = self.chrm_b.bp_pos.shape[0]

    positions_a = self.chrm_a.bp_pos
    positions_b = self.chrm_b.bp_pos

    idxs_a, valid_a = bp_to_idx(bps_a, positions_a, self.chrm_a.bin_size)
    idxs_b, valid_b = bp_to_idx(bps_b, positions_b, self.chrm_b.bin_size)


    valid = np.logical_and(valid_a, valid_b)

    idxs_a = idxs_a[valid]
    idxs_b = idxs_b[valid]
    counts = counts[valid]

    mat = np.zeros((size_a, size_b), dtype=np.int64)

    mat[idxs_a, idxs_b] = counts
    if self.chrm_a.chrm == self.chrm_b.chrm:
      mat[idxs_b, idxs_a] = counts

    return(mat)

  @property
  def dists(self):
    try:
      return(self.store["dists"][self.chrm_a.chrm][self.chrm_b.chrm][:][self.chrm_a.valid, :][:, self.chrm_b.valid])
    except KeyError:
      return(self.store["dists"][self.chrm_b.chrm][self.chrm_a.chrm][:][self.chrm_b.valid, :][:, self.chrm_a.valid].T)
