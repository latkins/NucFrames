import numpy as np
import os
import h5py
from Chromosome import Chromosome


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
      return(self.store["expr_contacts"][self.chrm_a.chrm][self.chrm_b.chrm][:][self.chrm_a.valid,:][:, self.chrm_b.valid])
    except KeyError:
      return(self.store["expr_contacts"][self.chrm_b.chrm][self.chrm_a.chrm][:][self.chrm_b.valid,:][:, self.chrm_a.valid].T)

  @property
  def dists(self):
    try:
      return(self.store["dists"][self.chrm_a.chrm][self.chrm_b.chrm][:][self.chrm_a.valid, :][:, self.chrm_b.valid])
    except KeyError:
      return(self.store["dists"][self.chrm_b.chrm][self.chrm_a.chrm][:][self.chrm_b.valid, :][:, self.chrm_a.valid].T)
