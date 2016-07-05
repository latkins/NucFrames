import numpy as np
import os
import h5py

class Chromosome(object):
  """
  Class to represent a single chromosome of a single cell.
  """
  def __init__(self, store, chrm, lower_bp_lim=None, upper_bp_lim=None):
    """
    Follows the HDF5 layout of NucFrame
    Contains cis-distances, coordinates, depths.
    """
    self.store = store
    self.chrm = chrm
    self.bin_size = store["bin_size"]
    self.valid = self.valid_idxs(lower_bp_lim, upper_bp_lim)

  def valid_idxs(self, lower, upper):
    # Assumes position is in sorted order .
    all_positions = self.store["bp_pos"][self.chrm][:]
    if lower and upper:
      return(np.logical_and(all_positions >= lower, all_positions < upper))
    elif lower and not upper:
      return(all_positions >= lower)
    elif not lower and upper:
      return(all_positions < upper)
    else:
      return(~np.isnan(all_positions))

  @property
  def bp_pos(self):
    return(self.store["bp_pos"][self.chrm][:][self.valid])

  @property
  def expr_contacts(self):
    print(self.store["expr_contacts"][self.chrm][self.chrm][:].T.shape)
    return(self.store["expr_contacts"][self.chrm][self.chrm][:][self.valid, self.valid])

  @property
  def dists(self):
    return(self.store["dists"][self.chrm][self.chrm][:][self.valid, self.valid])

  @property
  def positions(self):
    return(self.store["position"][self.chrm][:][:, self.valid, :])
