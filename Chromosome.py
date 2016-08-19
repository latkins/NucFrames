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
    Each property obeys the chromosome limits it has been passed, meaning
    indexes are directly comparable.
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

  def depths(self, alpha=None, i=0, tag=None):
    i = str(i)

    if alpha is None and tag is None:
      raise TypeError("Neither alpha nor tag passed to function")

    dset = self.store["surface_dist"]

    if tag:
      for k in dset.keys():
        try:
          t = dset[k]["tag"][()]
        except KeyError:
          pass
        else:
          if t == tag:
            return(dset[k][i][self.chrm][self.valid])

      raise ValueError("No entry found for tag: {}".format(tag))

    elif alpha:
      return(dset[str(alpha)][self.chrm][str(i)][:][self.valid])


  @property
  def cell(self):
    return(self.store["name"][()])

  @property
  def bp_pos(self):
    return(self.store["bp_pos"][self.chrm][:][self.valid].astype(np.int32))

  @property
  def expr_contacts(self):
    return(self.store["expr_contacts"][self.chrm][self.chrm][:][self.valid, self.valid])

  @property
  def dists(self):
    return(self.store["dists"][self.chrm][self.chrm][:][self.valid, self.valid])

  @property
  def positions(self):
    return(self.store["position"][self.chrm][:][:, self.valid, :])
