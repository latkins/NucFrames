import numpy as np
import os
import h5py
from itertools import combinations_with_replacement
import logging
from tqdm import tqdm

from distance_utils.all_pairs_euc_dist import nuc_dist_pairs
from depth_utils.alpha_shape import AlphaShape
from Chromosome import Chromosome


class NucFrame(object):
  """
  Class to represent a Nuc file, in a consistent fashion.

  -- Used by a NucFrame
  -- Calculates the maximum information possible (e.g. all distances)
  -- Somehow allow initialisation with allowed basepair limits.
  -- All getters / setters should then use these limits to return the appropriate stuff.

  """

  @classmethod
  def from_nuc(cls, nuc_files, nuc_slice_file):
    """
    Factory to produce a NucFrame (and the associated file) from a
    .nuc file.
    """


    try:
      os.remove(nuc_slice_file)
    except OSError:
      # File didn't actually exist.
      pass

    store = h5py.File(nuc_slice_file, 'a', libvar="latest")

    store["chrms"] = cls._extract_chrms(nuc_file)
    chrms = [x.decode("utf8") for x in store["chrms"]]
    store["bin_size"] = cls._get_bin_size(nuc_file, chrms)
    store["name"] = cls._get_name(nuc_file)
    cls._store_bp_positions(nuc_file, store, chrms)
    cls._store_expr_contacts(nuc_file, store, chrms)
    cls._store_dists(nuc_file, store, chrms)
    cls._store_positions(nuc_file, store, chrms)

  @staticmethod
  def _extract_chrms(nuc_file):
    # TODO: Extract from nuc_file, rather than just setting.
    chrms = ["X"] + [str(x) for x in range(1, 20)]
    return (list(map(lambda x: x.encode("utf-8"), chrms)))

  @staticmethod
  def _get_bin_size(nuc_file, chrms):
    sizes = set()
    particle = h5py.File(nuc_file, 'r')["structures"]["0"]["particles"]
    for chrm in chrms:
      positions = particle[chrm]["positions"][:]
      chrm_sizes = np.diff(positions)
      sizes = sizes.union(set(chrm_sizes))

    if len(sizes) != 1:
      raise ValueError("Inconsistent bin sizes: {}".format(len(sizes)))
    else:
      return (list(sizes)[0])

  @staticmethod
  def _get_name(nuc_file):
    raw_name, _ = os.path.splitext(os.path.basename(nuc_file))
    raw_name = raw_name.replace("-", "_")
    name = "_".join(raw_name.split("_")[:3])
    return (name)

  @staticmethod
  def _store_bp_positions(nuc_file, store, chrms):
    """
    Create the datastore for each chromosome that stores the basepair
    positions of each structural particle.
    """
    nuc = h5py.File(nuc_file, 'r')
    chrm_parts = nuc["structures"]["0"]["particles"]
    for chrm in chrms:
      positions = chrm_parts[chrm]["positions"][:]

      if np.all(np.sort(positions) != positions):
        raise ValueError("Positions not in sorted order.")

      store.create_dataset(os.path.join("bp_pos", chrm), data=positions)
      logging.info("Stored basepair positions for chrm {} in {}".format(
          chrm, store["name"]))

  @staticmethod
  def _store_expr_contacts(nuc_file, store, chrms):
    """Store experimental contacts from .nuc file.
    """
    f = h5py.File(nuc_file, 'r')
    k = list(f["contacts"]["working"].keys())[0]
    contact_chrm_as = f["contacts"]["working"][k]
    for chrm_a in contact_chrm_as.keys():
      if chrm_a not in chrms:
        continue
      contact_chrm_bs = contact_chrm_as[chrm_a]
      for chrm_b in contact_chrm_bs.keys():
        if chrm_b not in chrms:
          continue

        contact_vals = contact_chrm_bs[chrm_b][:].T
        chrm_path = os.path.join("expr_contacts", chrm_a, chrm_b)
        dset = store.create_dataset(chrm_path, data=contact_vals)
        logging.info("Created {}".format(chrm_path))

  @staticmethod
  def _store_dists(nuc_file, store, chrms):
    """Store the average distance between particles in a chromosome.
    """

    # A:B == B:A.T, so just add one, and add an accessor method
    chrm_pairs = list(combinations_with_replacement(chrms, 2))
    for (chrm_a, chrm_b) in tqdm(chrm_pairs):
      a_to_b, b_to_a = nuc_dist_pairs(nuc_file, chrm_a, chrm_b)
      dists_path = os.path.join("dists", chrm_a, chrm_b)

      dists = np.median(a_to_b, axis=2)

      try:
        store.create_dataset(dists_path, data=dists)
      except RuntimeError as e:
        logging.info("{}".format(e))
      else:
        logging.info("Created {}".format(dists_path))

  @staticmethod
  def _store_positions(nuc_file, store, chrms):
    """Store 3d positions of each particle in each model.
    """
    f = h5py.File(nuc_file, 'r')["structures"]["0"]["coords"]
    for chrm in chrms:
      position_path = os.path.join("position", chrm)
      store.create_dataset(position_path, data=f[chrm])
      logging.info("Created positions for chrm {} in {}".format(chrm, store[
          "name"]))

  def __init__(self, nuc_slice_file, chrm_limit_dict=None):
    """HDF5 hierarchy:
    name :: String -- the name of the NucFrame
    bin_size :: Int -- the common bin_size of the nuc files.
    chrms :: ["X", "1", ..] -- all of the chromosomes that are present.
    bp_pos/chrm :: [Int] -- The start bp index of each particle in each chrm.
    position/chrm :: [[[Float]]] -- (model, bead_idx, xyz)
    expr_contacts/chrm/chrm :: [[Int]] -- (bead_idx, bead_idx), raw contact count.
    dists/chrm/chrm :: [[Float]] -- (bead_idx, bead_idx), distanes between beads.
    """
    self.store = h5py.File(nuc_slice_file, 'r', libvar="latest")
    chromosomes = [x.decode("utf-8") for x in self.store["chrms"]]
    if not chrm_limit_dict:
      chrm_limit_dict = {chrm: (None, None) for chrm in chromosomes}

    self.chrms = Chromosomes(self.store, chromosomes, chrm_limit_dict)


class Chromosomes(object):
  def __init__(self, store, chromosomes, chrm_limit_dict):
    self.store = store
    self.chrms = chromosomes
    self.chrm_limit_dict = chrm_limit_dict

  def __getitem__(self, chrm):
    lower, upper = self.chrm_limit_dict[chrm]
    return(Chromosome(self.store, chrm, lower, upper))

  def __iter__(self):
    for chrm in self.chrms:
      lower, upper = self.chrm_limit_dict[chrm]
      yield(Chromosome(self.store, chrm, lower, upper))

if __name__=="__main__":
  import glob

  nuc_file = "/home/lpa24/dev/cam/data/edl_chromo/mm10/single_cell_nuc_100k/Q5_ambig_10x_100kb.nuc"
  slice_file = "/mnt/SSD/LayeredNuc/Q5_ambig_100kb.hdf5"
  # nf = NucFrame.from_nuc(nuc_file, slice_file)
  nf = NucFrame(slice_file)

  coords = []
  for c in nf.chrms:
    coords.append(c.positions[0, :, :])

  coords = np.vstack(coords)

  alpha_shape = AlphaShape(coords)
  print(alpha_shape)

  print(alpha_shape.get_verts(1.5))
