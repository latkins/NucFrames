import numpy as np
import os
import h5py
from itertools import combinations_with_replacement, combinations
import logging
from tqdm import tqdm
import networkx as nx
import math

from distance_utils.all_pairs_euc_dist import nuc_dist_pairs
from depth_utils.alpha_shape import AlphaShape
from depth_utils.point_surface_dist import points_tris_dists
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
    cls._store_alpha_depths(nuc_file, store, chrms)
    return(cls(nuc_slice_file))

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

    # TODO: I need some messed up files to work.
    sizes = {math.floor(x  / 1000) * 1000 for x in sizes}
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

      positions = [math.floor(x  / 1000) * 1000 for x in positions]

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

  @staticmethod
  def _store_alpha_depths(nuc_file, store, chrms, alpha=1.6, perc=0.005):
    """Store the absolute distance of each particle from each surface.
    It is likely that there will be multiple disconnected surfaces found. The
    outer surface will be valid, as will an inner surface if present.

    Use a value of alpha to define the surface, and a percentage to decide how
    big stored subgraphs should be.
    """
    all_positions = []

    nuc = h5py.File(nuc_file, 'r')
    chrm_parts = nuc["structures"]["0"]["coords"]

    for chrm in chrms:
      positions = chrm_parts[chrm][:]
      all_positions.append(positions[0, :, :])

    all_positions = np.vstack(all_positions)
    alpha_shape = AlphaShape(all_positions)

    facets = list(alpha_shape.get_facets(alpha))

    # Construct the graph
    edges = {frozenset(x) for y in facets for x in combinations(y, 2)}
    nodes = {x for y in edges for x in y}
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    # Iterate over subgraphs, ordered by size.
    for i, sg in enumerate(sorted(
        nx.connected_component_subgraphs(g),
        key=lambda x: len(x),
        reverse=True)):

      valid_nodes = set(sg.nodes())
      # Filter facets
      valid_facets = np.array(
          [alpha_shape.coords[list(x)] for x in facets if x <= valid_nodes],
          dtype=np.float32)

      base_surface_dists_path = os.path.join("depths", str(i))

      # Store distances for points in each chromosome vs subgraph surface.
      if len(valid_nodes) / all_positions.shape[0] >= perc:
        store.create_dataset(os.path.join(base_surface_dists_path, "alpha"), data=alpha)
        for chrm in chrms:
          positions = chrm_parts[chrm][0,:,:].astype(np.float32)
          surface_dists = np.min(points_tris_dists(valid_facets, positions), axis=1)
          store.create_dataset(os.path.join(base_surface_dists_path, chrm), data=surface_dists)
          logging.info("Inserted distances from chrm {} particles to surface {}".format(chrm, i))

      # Store surfaces, even those we didn't measure distance for.
      surf_path = os.path.join("surface", str(i))
      facet_indices = np.array([list(x) for x in facets if x <= valid_nodes])
      store.create_dataset(surf_path, data=facet_indices)
      logging.info("Inserted surface verticies for {}".format(i))


  def __init__(self, nuc_slice_file, chrm_limit_dict=None):
    """HDF5 hierarchy:
    name :: String -- the name of the NucFrame
    bin_size :: Int -- the common bin_size of the nuc files.
    chrms :: ["X", "1", ..] -- all of the chromosomes that are present.
    bp_pos/chrm :: [Int] -- The start bp index of each particle in each chrm.
    position/chrm :: [[[Float]]] -- (model, bead_idx, xyz)
    expr_contacts/chrm/chrm :: [[Int]] -- (bead_idx, bead_idx), raw contact count.
    dists/chrm/chrm :: [[Float]] -- (bead_idx, bead_idx), distanes between beads.
    depths/i/alpha :: Float -- alpha value used to calculate depths.
    depths/i/chrm/ :: [Float] -- (bead_idx, ), depth of point from surface i.
    surface/i :: [[Int]] -- vertices of triangles that form surface i.
                            index is relative to positions vstacked by chromosome.
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
    return (Chromosome(self.store, chrm, lower, upper))

  def __iter__(self):
    for chrm in self.chrms:
      lower, upper = self.chrm_limit_dict[chrm]
      yield (Chromosome(self.store, chrm, lower, upper))


if __name__ == "__main__":
  import glob

  logging.basicConfig(level=logging.INFO)
  slice_path = "/mnt/SSD/LayeredNuc/frames/"
  # for nuc_file in glob.glob("/home/lpa24/dev/cam/data/edl_chromo/mm10/single_cell_nuc_100k/ambig/*"):
  #   print(nuc_file)
  #   slice_file = os.path.join(slice_path, os.path.splitext(os.path.basename(nuc_file))[0] + ".hdf5")
  #   nf = NucFrame.from_nuc(nuc_file, slice_file)
    #nf = NucFrame(slice_file)

  old_file_path = "/home/lpa24/dev/cam/data/edl_chromo/mm10/old_single_cell_nuc_100k/"
  old_files = ["S1028_GTGAAA_06_10x_100kb_mm10.nuc",
                "S1112_NXT-46_06_10x_100kb_mm10.nuc",
                "S1112_NXT-55_18_10x_100kb_mm10.nuc",
                "S1112_NXT-65_11_10x_100kb_mm10.nuc",
                "S1227_NXT-34_02_10x_100kb_mm10.nuc",
                "S1028_TGACCA_07_10x_100kb_mm10.nuc",
                "S1112_NXT-48_08_10x_100kb_mm10.nuc",
                "S1112_NXT-57_20_10x_100kb_mm10.nuc",
                "S1112_NXT-67_13_10x_100kb_mm10.nuc",
                "S1227_NXT-36_04_10x_100kb_mm10.nuc",
                "S1028_ACAGTG_01_10x_100kb_mm10.nuc",
                "S1112_NXT-44_04_10x_100kb_mm10.nuc",
                "S1112_NXT-52_15_10x_100kb_mm10.nuc",
                "S1112_NXT-58_21_10x_100kb_mm10.nuc",
                "S1112_NXT-68_14_10x_100kb_mm10.nuc",
                "S1227_NXT-41_06_10x_100kb_mm10.nuc",
                "S1028_CCGTCC_03_10x_100kb_mm10.nuc",
                "S1112_NXT-45_05_10x_100kb_mm10.nuc",
                "S1112_NXT-53_16_10x_100kb_mm10.nuc",
                "S1112_NXT-63_09_10x_100kb_mm10.nuc",
                "S1225_NXT-32_01_10x_100kb_mm10.nuc"]

  old_paths = [os.path.join(old_file_path, x) for x in old_files]
  for nuc_file in old_paths:
    print(nuc_file)
    try:
      slice_file = os.path.join(slice_path, os.path.splitext(os.path.basename(nuc_file))[0] + ".hdf5")
      nf = NucFrame.from_nuc(nuc_file, slice_file)
    except ValueError as e:
      print(e)


