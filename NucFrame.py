import numpy as np
import os
import h5py
from itertools import combinations_with_replacement, combinations
import logging
from tqdm import tqdm
import networkx as nx
import math
from collections import defaultdict, deque, Counter

from distance_utils.all_pairs_euc_dist import nuc_dist_pairs
from depth_utils.alpha_shape import AlphaShape, circular_subgroup
from depth_utils.point_surface_dist import points_tris_dists
from Chromosome import Chromosome

def surf_norm(tri):
  a = tri[1] - tri[0]
  b = tri[2] - tri[0]
  return(np.cross(a, b))


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
    cls._store_alpha_shape(store, chrms)
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
  def _store_alpha_shape(store, chrms):
    "Calculates and stores an AlphaShape."

    all_positions = []
    for chrm in chrms:
      all_positions.append(store["position"][chrm][0,:,:])

    all_positions = np.vstack(all_positions)

    # Store alpha_shape.interval_dict
    alpha_shape = AlphaShape.from_points(all_positions)
    try:
      del(store["alpha_shape"])
    except KeyError as e:
      pass

    for k in { len(x) for x in alpha_shape.interval_dict.keys() }:
      simplices = []
      ab_values = []
      for simplex, (a, b) in alpha_shape.interval_dict.items():
        if len(simplex) == k:
          simplices.append(simplex)
          ab_values.append([a,b])

      path = os.path.join("alpha_shape", str(k))
      store.create_dataset(os.path.join(path, "simplices"), data=simplices)
      store.create_dataset(os.path.join(path, "ab"), data=ab_values)
    logging.info("Created AlphaShape dataset")

  def _load_alpha_shape(self):
    all_positions = self.all_pos
    interval_dict = {}
    for k in self.store["alpha_shape"].keys():
      simplices = self.store["alpha_shape"][k]["simplices"][:]
      ab_values = self.store["alpha_shape"][k]["ab"][:]
      for simplex, ab in zip(simplices, ab_values):
        interval_dict[tuple(simplex)] = ab.tolist()

    self.alpha_shape = AlphaShape(interval_dict, self.all_pos)

  def alpha_surface(self, alpha=1.6):
    """For a given value of alpha, return all surfaces, ordered by size.
    """
    all_pos = self.all_pos
    all_facets = list(self.alpha_shape.get_facets(alpha))
    # Construct the graph
    edges = {x for y in all_facets for x in circular_subgroup(y, 2)}
    nodes = {x for y in edges for x in y}
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    points = self.alpha_shape.coords.astype(np.float32)

    surfaces = []

    # Iterate over subgraphs, ordered by size.
    for sg in (sorted(nx.connected_component_subgraphs(g),
        key=lambda x: len(x),
        reverse=True)):

      valid_nodes = set(sg.nodes())

      # Filter facets
      facet_vert_idxs = np.array([x for x in all_facets if all_in(x, valid_nodes)])
      facet_vert_coords = np.array([all_pos[x] for x in facet_vert_idxs], dtype=np.float32)

      flip_order = [1, 0, 2]
      flip_facet_vert_coords = facet_vert_coords[:, flip_order, :]
      # Precompute norms
      facet_norms = np.cross(facet_vert_coords[:,0,:] - facet_vert_coords[:,1,:],
                              facet_vert_coords[:,1,:] - facet_vert_coords[:,2,:])
      flip_facet_norms = np.cross(flip_facet_vert_coords[:,0,:] - flip_facet_vert_coords[:,1,:],
                                  flip_facet_vert_coords[:,1,:] - flip_facet_vert_coords[:,2,:])

      # Ensure consistent vertex ordering
      # Check that the normal of each facet is in the same direction as its neighbour.

      vert_idx_facet_idx_lu = defaultdict(set)
      for facet_idx, facet in enumerate(facet_vert_idxs):
        for vert_idx in facet:
          vert_idx_facet_idx_lu[vert_idx].add(facet_idx)

      facet_neighbor_lu = defaultdict(set)
      for facet_idx, facet in enumerate(facet_vert_idxs):
        c = Counter()
        for vert_idx in facet:
          c.update(vert_idx_facet_idx_lu[vert_idx] - set([facet_idx]))
        facet_neighbor_lu[facet_idx] = {x for x, n in c.items() if n >= 2 }

      processed_facets = set([0])
      d = deque()
      d.append(0)
      while True:
        try:
          facet_idx = d.popleft()
        except IndexError:
          break

        facet_n = facet_norms[facet_idx]

        # Neighboring facets
        neighbor_idxs = facet_neighbor_lu[facet_idx] - processed_facets

        for neighbor_idx in neighbor_idxs:
          neighbor_n = facet_norms[neighbor_idx]
          proj = np.dot(facet_n, neighbor_n)

          if proj < 0:
            t = facet_vert_coords[neighbor_idx]
            t_ = facet_norms[neighbor_idx]

            facet_vert_coords[neighbor_idx] = flip_facet_vert_coords[neighbor_idx]
            facet_norms[neighbor_idx] = flip_facet_norms[neighbor_idx]

            flip_facet_vert_coords[neighbor_idx] = t
            flip_facet_norms[neighbor_idx] = t_


          if proj != 0:
            d.append(neighbor_idx)
            processed_facets.add(neighbor_idx)

      surfaces.append(facet_vert_coords)
    return(surfaces)


  def store_surface_dists_tag(self, alpha, tag):
    """
    Since there are so many surfaces, sometimes we want
    to add a tag. E.g. EXTERNAL for distances to exterior.
    This is because different cells might have different alpha values
    for the external surface.
    """
    path = "surface_dist"

    # Check tag isn't already present
    for alpha in self.store[path].keys():
      try:
        t = self.store[path][alpha]["tag"][()]
      except KeyError:
        pass
      else:
        assert t != tag

    path = os.path.join("surface_dist", str(alpha), "tag")
    self.store.create_dataset(path, data=tag)

  def store_surface_dists(self, alpha=1.6):
    """Store the absolute distance of each particle from each surface.
    It is likely that there will be multiple disconnected surfaces found. The
    outer surface will be valid, as will an inner surface if present.

    Use a value of alpha to define the surface, and a percentage to decide how
    big stored subgraphs should be.
    """
    surfaces = self.alpha_surface(alpha)

    try:
      del self.store["surface_dist"][str(alpha)]
    except KeyError:
      pass

    for i, facets in enumerate(surfaces):
      facets = facets.astype(np.float32)
      surface_size = facets.shape[0]

      # Store information about surface.
      path = os.path.join("surface_dist", str(alpha), str(i))
      self.store.create_dataset(os.path.join(path, "surface_size"), data=surface_size)

      for chrm in self.chrms:
        chrm_pos = chrm.positions[0,:,:].astype(np.float32)
        surface_dists = np.min(points_tris_dists(facets, chrm_pos), axis=1)
        self.store.create_dataset(os.path.join(path, chrm.chrm), data=surface_dists)


  def __init__(self, nuc_slice_file, chrm_limit_dict=None, mode="r"):
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
    alpha_shape/k/simplices :: [[Int]] -- (n_simplicies, k), indices of k-simplices.
    alpha_shape/k/ab :: [(a, b)] -- (n_simplicies, 2), a and b values for k-simplices.
                        ^ -- NOTE: length of the two alpha_shape entries align.
    surface_dist/alpha_val/tag :: optional tag for this value of alpha.
    surface_dist/alpha_val/i/surface_size :: size of surface i for alpha
    """
    self.store = h5py.File(nuc_slice_file, mode, libvar="latest")
    self.nuc_slice_file = nuc_slice_file
    chromosomes = [x.decode("utf-8") for x in self.store["chrms"]]
    if not chrm_limit_dict:
      chrm_limit_dict = {chrm: (None, None) for chrm in chromosomes}

    self.chrms = Chromosomes(self.store, chromosomes, chrm_limit_dict)
    self._load_alpha_shape()

  @property
  def all_pos(self):
    all_positions = []
    for chrm in self.chrms:
      all_positions.append(chrm.positions[0,:,:])

    all_positions = np.vstack(all_positions)
    return(all_positions)



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

def all_in(tup, s):
  for t in tup:
    if not t in s:
      return False
  return True


def all_exterior_depths():
  """
  Calculate surfaces for these files, with alpha values manually chose to represent
  the external surface.
  """
  slice_path = "/mnt/SSD/LayeredNuc/frames/"
  nf_alpha_pairs = [("P2E8_ambig_10x_100kb.hdf5", 3.4),
                    ("P36D6_ambig_10x_100kb.hdf5", 4.0),
                    # ("1028_GTGAAA_06.hdf5", 4.0),
                    ("S1112_NXT-48_08.hdf5", 3.2),
                    ("S1112_NXT-58_21.hdf5", 3.5),
                    ("S1225_NXT-32_01.hdf5", 4.4),
                    ("P2I5_ambig_10x_100kb.hdf5", 3.3),
                    ("Q5_ambig_10x_100kb.hdf5", 3.1),
                    ("S1028_TGACCA_07.hdf5", 2.9),
                    ("S1112_NXT-52_15.hdf5", 3.7),
                    ("S1112_NXT-63_09.hdf5", 4.4),
                    ("S1227_NXT-34_02.hdf5", 3.5),
                    ("P2J8_ambig_10x_100kb.hdf5", 4.0),
                    ("Q6_ambig_10x_100kb.hdf5", 2.6),
                    ("S1112_NXT-44_04.hdf5", 2.7),
                    ("S1112_NXT-53_16.hdf5", 3.2),
                    ("S1112_NXT-65_11.hdf5", 2.9),
                    ("S1227_NXT-36_04.hdf5", 3.7),
                    ("P30E4_ambig_10x_100kb.hdf5", 2.6),
                    ("S1028_ACAGTG_01.hdf5", 2.7),
                    ("S1112_NXT-45_05.hdf5", 3.0),
                    ("S1112_NXT-55_18.hdf5", 4.5),
                    ("S1112_NXT-67_13.hdf5", 2.7),
                    ("S1227_NXT-41_06.hdf5", 3.2),
                    ("P30E8_ambig_10x_100kb.hdf5", 4.2),
                    ("S1028_CCGTCC_03.hdf5", 2.6),
                    ("S1112_NXT-46_06.hdf5", 2.6),
                    ("S1112_NXT-57_20.hdf5", 3.5),
                    ("S1112_NXT-68_14.hdf5", 2.6),
                    ("UpL13_ambig_10x_100kb.hdf5", 3.2)]
  for nf_name, alpha in nf_alpha_pairs:
    print(nf_name)
    path = os.path.join(slice_path, nf_name)
    nf = NucFrame(path, mode='a')
    nf.store_surface_dists(alpha)
    nf.store_surface_dists_tag(alpha, "EXTERNAL")

if __name__ == "__main__":
  import glob

  all_exterior_depths()
  logging.basicConfig(level=logging.INFO)
  # slice_path = "/mnt/SSD/LayeredNuc/frames/"
  # nf = NucFrame(os.path.join(slice_path, "Q5_ambig_10x_100kb.hdf5"))
  # nf.alpha_surface()
  # for nuc_file in glob.glob("/home/lpa24/dev/cam/data/edl_chromo/mm10/single_cell_nuc_100k/ambig/*"):
  #   slice_file = os.path.join(slice_path, os.path.splitext(os.path.basename(nuc_file))[0] + ".hdf5")
  #   nf = NucFrame.from_nuc(nuc_file, slice_file)
    # nf = NucFrame(slice_file)

  # old_file_path = "/home/lpa24/dev/cam/data/edl_chromo/mm10/old_single_cell_nuc_100k/"
  # old_files = ["S1028_GTGAAA_06_10x_100kb_mm10.nuc",
  #               "S1112_NXT-46_06_10x_100kb_mm10.nuc",
  #               "S1112_NXT-55_18_10x_100kb_mm10.nuc",
  #               "S1112_NXT-65_11_10x_100kb_mm10.nuc",
  #               "S1227_NXT-34_02_10x_100kb_mm10.nuc",
  #               "S1028_TGACCA_07_10x_100kb_mm10.nuc",
  #               "S1112_NXT-48_08_10x_100kb_mm10.nuc",
  #               "S1112_NXT-57_20_10x_100kb_mm10.nuc",
  #               "S1112_NXT-67_13_10x_100kb_mm10.nuc",
  #               "S1227_NXT-36_04_10x_100kb_mm10.nuc",
  #               "S1028_ACAGTG_01_10x_100kb_mm10.nuc",
  #               "S1112_NXT-44_04_10x_100kb_mm10.nuc",
  #               "S1112_NXT-52_15_10x_100kb_mm10.nuc",
  #               "S1112_NXT-58_21_10x_100kb_mm10.nuc",
  #               "S1112_NXT-68_14_10x_100kb_mm10.nuc",
  #               "S1227_NXT-41_06_10x_100kb_mm10.nuc",
  #               "S1028_CCGTCC_03_10x_100kb_mm10.nuc",
  #               "S1112_NXT-45_05_10x_100kb_mm10.nuc",
  #               "S1112_NXT-53_16_10x_100kb_mm10.nuc",
  #               "S1112_NXT-63_09_10x_100kb_mm10.nuc",
  #               "S1225_NXT-32_01_10x_100kb_mm10.nuc"]

  # old_paths = [os.path.join(old_file_path, x) for x in old_files]
  # for nuc_file in old_paths:
  #   print(nuc_file)
  #   try:
  #     base_name = os.path.splitext(os.path.basename(nuc_file))[0]
  #     name = "_".join(base_name.split("_")[:-3])
  #     slice_file = os.path.join(slice_path,  "{}.hdf5".format(name))
  #     nf = NucFrame.from_nuc(nuc_file, slice_file)
  #   except ValueError as e:
  #     print(e)


