"""
Functions to load a nucleus file (.nuc) as a pandas dataframe.
"""
import numpy as np
import h5py
import logging
import pandas as pd
from numba import jit
from collections import defaultdict
from os.path import splitext, basename, join
import os
import sys
from itertools import combinations_with_replacement, combinations
# from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import hashlib

from utils import bp_to_idx, filter_bps

# This is shit, sort it.
sys.path.append(os.path.abspath('../all_pairs/'))
from all_pairs_euc_dist import nuc_dist_pairs


class NucFrame:
  """
  Class to represent multiple nuc files in a pandas dataframe.
  """

  def __init__(self, frame_file, nuc_files=None, chrms=None, av_dist=False):
    """
    nuc_files :: [path_to_nuc]
    chrms :: ["X", "1", ..]
    store format:
    /positions/chr :: [bp]
    /chrms :: ["X", "1", ..]
    /contacts/nuc_name/chr/chr :: [[bp_a, bp_b, count]]
    /contact_dists/nuc_name:: [dist] (percentiles)
    /dists/nuc_name/chr/chr :: [[dist]]
    /bin_size :: bin_size
    /nuc_names :: [names]
    /files_hash :: hash
    /dist_oe/nuc_name/chrm :: [[obs_exp_value]]
    """

    store = h5py.File(frame_file, 'a', libver="latest")

    # If nuc_files are present, create the dataset.
    if nuc_files:
      new_hash = hashlib.md5(b"".join([x.encode(
          'utf-8') for x in sorted(nuc_files)])).hexdigest()
      try:
        old_hash = store["files_hash"][()]
      except KeyError:
        old_hash = ""
      # If hashes are not equal, delete the store and recalculate.
      if old_hash != new_hash:
        os.remove(frame_file)
        store = h5py.File(frame_file, 'a', libver="latest")
        store["files_hash"] = new_hash

        store["nuc_files"] = [x.encode("utf8") for x in nuc_files]

        if chrms == None:
          chrms = ["X"] + [str(x) for x in range(1, 20)]

        store["chrms"] = [x.encode('utf8') for x in chrms]

        store["bin_size"] = self._get_bin_size(nuc_files, store["chrms"])
        chrm_pos = self._get_positions_dict(nuc_files, chrms,
                                            store["bin_size"][()])
        for k, v in chrm_pos.items():
          store.create_dataset(os.path.join("chrm_pos", k), data=v)

        store["nuc_names"] = [
            x.encode("utf8") for x in self._clean_nuc_names(nuc_files)
        ]
        print(list(store["nuc_names"]))

        file_name_pairs = list(zip(nuc_files, store["nuc_names"][:]))

        # sets /contacts/nuc_name/chr/chr
        # Tim seems to have removed contacts from the nuc file.
        self._fill_contacts(store, file_name_pairs, chrms, store["chrm_pos"],
                            store["bin_size"])

        self._calc_dists(store,
                         file_name_pairs,
                         chrms,
                         store["chrm_pos"],
                         store["bin_size"][()],
                         av_dist=av_dist)
        self.dists = Dists(store["dists"])

        self._fill_contact_dists(store, store["chrm_pos"],
                                 store["bin_size"][()], store["contacts"],
                                 store["dists"])

    self.store = store
    self.chrms = [x.decode("utf8") for x in store["chrms"]]
    self.bin_size = store["bin_size"][()]
    self.chrm_pos = store["chrm_pos"]
    self.nuc_names = [x.decode("utf8") for x in store["nuc_names"]]
    self.contacts = store["contacts"]
    self.dists = Dists(store["dists"])
    self.contact_dists = store["contact_dists"]
    self.nuc_files = [x.decode("utf8") for x in store["nuc_files"]]
    #self.dist_oe = store["dist_oe"]

  def _get_bin_size(self, nuc_files, chrms):
    """
    Find common binsize across files. If more than one size,
    raise an error.
    """
    sizes = set()
    for nuc_file in nuc_files:
      particle = h5py.File(nuc_file, 'r')["structures"]["0"]["particles"]
      for chrm in chrms:
        positions = particle[chrm]["positions"][:]
        chrm_sizes = np.diff(positions)
        sizes = sizes.union(set(chrm_sizes))
    if len(sizes) != 1:
      raise ValueError("Inconsistent bin sizes: {}".format(len(sizes)))
    else:
      return (list(sizes)[0])

  def _get_positions_dict(self, nuc_files, chrms, bin_size):
    """
    Find the common positions between cells.
    """
    starts_dict = defaultdict(lambda: 0)
    ends_dict = defaultdict(lambda: np.inf)
    for nuc_file in nuc_files:
      nuc = h5py.File(nuc_file, 'r')
      chrm_parts = nuc["structures"]["0"]["particles"]
      for chrm in chrms:
        positions = chrm_parts[chrm]["positions"][:]

        prev_start = starts_dict[chrm]
        prev_end = ends_dict[chrm]

        if positions[0] > prev_start:
          starts_dict[chrm] = positions[0]
        if positions[-1] < prev_end:
          ends_dict[chrm] = positions[-1]

    return ({chrm: np.arange(starts_dict[chrm], ends_dict[chrm], bin_size)
             for chrm in chrms})

  def _clean_nuc_names(self, nuc_files):
    names = []
    for nuc_file in nuc_files:
      raw_name, _ = splitext(basename(nuc_file))
      raw_name = raw_name.replace("-", "_")
      name = "_".join(raw_name.split("_")[:3])
      names.append(name)
    return (names)

  def _fill_contacts(self, store, file_name_pairs, chrms, chrm_pos_dict,
                     bin_size):
    """
    Store the contacts from nuc_files.
    """
    # Prime with contacts
    for (nuc_file, file_name) in file_name_pairs:
      f = h5py.File(nuc_file, 'r')
      # NOTE HERE: Tim changed how this is stored for ambiguous stuff. Will have to deal with it.
      k = list(f["contacts"]["working"].keys())[0]
      contact_chrm_as = f["contacts"]["working"][k]
      for chrm_a in contact_chrm_as.keys():
        if chrm_a not in chrms:
          continue
        valid_pos_a = chrm_pos_dict[chrm_a]
        contact_chrm_bs = contact_chrm_as[chrm_a]
        for chrm_b in contact_chrm_bs.keys():
          if chrm_b not in chrms:
            continue

          valid_pos_b = chrm_pos_dict[chrm_b]
          contact_vals = contact_chrm_bs[chrm_b][:].T
          chrm_path = os.path.join("contacts", file_name.decode('utf8'),
                                   chrm_a, chrm_b)
          valid_idxs_a = filter_bps(contact_vals[:, 0],
                                    chrm_pos_dict[chrm_a], bin_size)
          valid_idxs_b = filter_bps(contact_vals[:, 1],
                                    chrm_pos_dict[chrm_b], bin_size)

          idxs = np.logical_and(valid_idxs_a, valid_idxs_b)
          dset = store.create_dataset(chrm_path, data=contact_vals[idxs, :])
          logging.info("Created {}".format(chrm_path))

  def _calc_dists(self,
                  store,
                  file_name_pairs,
                  chrms,
                  chrm_pos_dict,
                  bin_size,
                  av_dist=False):
    """
    Top level function, takes a list of nuc files and list of
    chromosomes. Saves the euclidean distances between all pairs of
    beads in a HDF file.
    NOTE: May need to tweak the grid size in the cuda functions if we
    change resolution much.

    if av_dist == True, store only the median distances between each pair.

    store :: hdf5 file object
    file_name_pairs :: [(file_path , file_name)]
    MUTATES THE HDF FILE:
    dist/cell_name/chrm_a/chrm_b/:[[[distance]]] (a_dim, b_dim, models)
    """

    # A:B == B:A.T, so just add one, and add an accessor method
    chrm_pairs = list(combinations_with_replacement(chrms, 2))
    for (chrm_a, chrm_b) in tqdm(chrm_pairs):
      for nuc_file, nuc_name in file_name_pairs:
        f = h5py.File(nuc_file, 'r')["structures"]["0"]["particles"]
        valid_a_idxs = filter_bps(f[chrm_a]["positions"],
                                  chrm_pos_dict[chrm_a][:], bin_size)
        valid_b_idxs = filter_bps(f[chrm_b]["positions"],
                                  chrm_pos_dict[chrm_b][:], bin_size)

        # Need to calculate valid indexes here. So, take the positions and get valid idxs.
        import numba
        try:
          a_to_b, b_to_a = nuc_dist_pairs(nuc_file, chrm_a, chrm_b)
        except numba.cuda.cudadrv.driver.CudaAPIError:
          import IPython
          IPython.embed()

        a_to_b = a_to_b[np.ix_(valid_a_idxs, valid_b_idxs)]
        b_to_a = b_to_a[np.ix_(valid_b_idxs, valid_a_idxs)]
        a_to_b_path = os.path.join("dists", nuc_name.decode('utf-8'), chrm_a,
                                   chrm_b)

        if av_dist:
          a_to_b = np.median(a_to_b, axis=2)
          b_to_a = np.median(b_to_a, axis=2)
        # a to b
        try:
          store.create_dataset(a_to_b_path, data=a_to_b)
        except RuntimeError as e:
          if chrm_a == chrm_b:
            logging.info("Could not create {}".format(a_to_b_path))
          else:
            logging.warning("Could not create {}".format(a_to_b_path))
        else:
          logging.info("Created {}".format(a_to_b_path))

  def _fill_contact_dists(self, store, positions, bin_size, contacts, dists):
    """
        positions (h5py dataset) :: chrm/[bp_pos]
        contacts (h5py dataset) :: name/chrm_a/chrm_b/[[bp_a, bp_b, count]]
        For each cell, find the 3d distances of contacts.
        """
    for cell in contacts.keys():
      cell_contacts = contacts[cell]
      contact_count = 0
      cell_contact_dists = []
      for chrm_a in cell_contacts.keys():
        chr_contacts_ = cell_contacts[chrm_a]
        for chrm_b in chr_contacts_.keys():
          chr_contacts = chr_contacts_[chrm_b][:]
          contact_count += chr_contacts.shape[0]
          chr_a_idxs, valid_a = bp_to_idx(chr_contacts[:, 0],
                                          positions[chrm_a][:], bin_size)
          chr_b_idxs, valid_b = bp_to_idx(chr_contacts[:, 1],
                                          positions[chrm_b][:], bin_size)

          valid = np.logical_and(valid_a, valid_b)
          chr_a_idxs = chr_a_idxs[valid]
          chr_b_idxs = chr_b_idxs[valid]

          a_b_dists = self.dists[cell, chrm_a, chrm_b][:][chr_a_idxs,
                                                          chr_b_idxs]
          cell_contact_dists.append(a_b_dists.flatten())
      cell_contact_dists = np.hstack(cell_contact_dists)
      cell_contact_dists.sort()
      vals = (cell_contact_dists.shape[0] *
              np.arange(0, 1.0, 0.01)).astype(np.int32)
      contact_dist_percentiles = cell_contact_dists[vals]
      store.create_dataset(
          os.path.join("contact_dists", cell),
          data=contact_dist_percentiles)

  def _calc_pseudo_cm(self, out_file, percentile=0.95):
    """
    From the distances matrices and contacts of each cell being considered,
    store the 10/15/20/../95/100 percentile distances for the contacts in that
    cell.
    """

    store = h5py.File(self.pseudo_contact_mats, 'w')
    for nuc_file in self.nuc_files:
      cell_dists = []
      cell_obj = self.dist_mat[nuc_file]["contacts"]
      for chrm_a in cell_obj.keys():
        a_store = cell_obj[chrm_a]
        for chrm_b in a_store.keys():
          a_bps = a_store[chrm_b][:, 0]
          a_pos = self.chrm_pos_dict[chrm_a]
          b_bps = a_store[chrm_b][:, 1]
          b_pos = self.chrm_pos_dict[chrm_b]
          counts = a_store[chrm_b][:, 2]
          a_idxs, valid_a_idx = bp_to_idx(a_bps, a_pos, self.bin_size)
          b_idxs, valid_b_idx = bp_to_idx(b_bps, b_pos, self.bin_size)

          valid_idxs = np.logical_and(valid_a_idx, valid_b_idx)
          a_idxs = a_idxs[valid_idxs]
          b_idxs = b_idxs[valid_idxs]

          all_dists = self.dist_mat[nuc_file][chrm_a][chrm_b][:]
          dists = all_dists[np.ix_(a_idxs, b_idxs)]
          cell_dists.extend(dists.flatten())

      print("done")
      # print(sorted(cell_dists))
      break

      # Find equivialant contact dist.
      # nuc =
      # Load all matrices in matpairs.
      # Threshold each matrix.
      # Save
      pass

  def track_to_idx(self, track_name, ref_nuc_file=None):

    f = h5py.File(ref_nuc_file, 'r')
    try:
      track = f["dataTracks"]["external"][track_name]
    except KeyError:
      track = f["dataTracks"]["derived"][track_name]

    idx_dict = {}
    value_dict = {}

    # Only use chrms that we care about and exist in the track
    for chrm in (set(self.chrms) & set(track.keys())):
      regions = track[chrm]["regions"][:]
      values = track[chrm]["values"][:]
      bb_pos = self.chrm_pos[chrm][:]
      start_idxs, valid_start_idxs = bp_to_idx(regions[:, 0], bb_pos,
                                               self.bin_size)
      end_idxs, valid_end_idxs = bp_to_idx(regions[:, 1], bb_pos,
                                           self.bin_size)
      valid_idxs = np.logical_and(valid_start_idxs, valid_end_idxs)

      start_idxs = start_idxs[valid_idxs]
      end_idxs = end_idxs[valid_idxs]
      values = values[valid_idxs]

      idx_dict[chrm] = np.vstack((start_idxs, end_idxs)).T
      value_dict[chrm] = values

    return (idx_dict, value_dict)

  def coords(self, nuc_file, chrm, struct=0):
    """
    Just choose the first structure. Could also average or something.
    """
    f = h5py.File(nuc_file, 'r')["structures"]["0"]
    coords = f["coords"][chrm][struct][:]

    bps = f["particles"][chrm]["positions"][:]
    idxs, valid_idxs = self.bps_to_idx(chrm, bps)
    idxs = idxs[valid_idxs]
    coords = coords[valid_idxs]

    return (coords)

  def depth_frame(self, recalc=False):
    """
    Return a DataFrame representing the depth of each bead in self.positions
    """
    if recalc:
      from os import sys, path
      sys.path.append("/home/lpa24/dev/cam/")
      from nucleus.NucApi import Nucleus, DERIVED

      for nuc_file in tqdm(self.nuc_files):

        nuc = Nucleus(nuc_file)
        nuc.calcDepths(range(10), self.chrms, radius=2.5, nVoxels=200, transInterface=False, label="nuc-depth")
        nuc.calcDepths(
            range(10),
            self.chrms,
            transInterface=True,
            label="trans_depth")
        nuc.calcDepths(
            range(10),
            self.chrms,
            separateChromos=True,
            label="chrm_depth")
        nuc.save()

    cells_dict = {}
    frms = []
    for cell_idx, nuc_file in enumerate(self.nuc_files):
      cell_dict = {}
      data_tracks = h5py.File(nuc_file, 'r')["dataTracks"]
      nuc_depth_track = data_tracks["derived"]["nuc-depth"]

      deepest = 0
      for chrm in self.chrms:
        val = np.max(nuc_depth_track[chrm]["values"][:, 0])
        if val > deepest:
          deepest = val

      for chrm in nuc_depth_track.keys():
        bps = nuc_depth_track[chrm]["regions"][:, 0]
        values = nuc_depth_track[chrm]["values"][:, 0]
        idxs, valid_idxs = self.bps_to_idx(chrm, bps)

        idxs = idxs[valid_idxs]
        values = values[valid_idxs]
        percs = (values.copy() / deepest) * 100
        bps = bps[valid_idxs]

        cell_dict[chrm] = (idxs, values)
        frms.append(pd.DataFrame({
            "cell": self.nuc_names[cell_idx],
            "chrm": chrm,
            "bps": bps,
            "idx": idxs,
            "depth": values,
            "percs": percs
        }))
    depth_frm = pd.concat(frms)
    return (depth_frm)

  def dists_frame(self,
                  chrm_a,
                  chrm_b,
                  pos_a=None,
                  pos_b=None,
                  recalc=False):
    """
    Given a pair of chromosomes (and positions optionally positions),
    create a frame of
    :: cell chrm_a chrm_b pos_a pos_b dist
    """
    if recalc:
      self._calc_dists()

    store = h5py.File(self.dist_mat, 'r')
    for nuc_file in self.nuc_files:
      nm, _ = os.path.splitext(os.path.basename(nuc_file))
      store_x = h5py.File(
          "/home/lpa24/dev/cam/data/mm10_esc_all_dists/all_by_all.h5", 'r')
      f = h5py.File(nuc_file, 'r')["structures"]["0"]["particles"]

      chrm_pairs = list(combinations_with_replacement(self.chrms, 2))
      for (chrm_a, chrm_b) in chrm_pairs:
        i_a = filter_bps(f[chrm_a]["positions"], self.chrm_pos_dict[chrm_a])
        i_b = filter_bps(f[chrm_b]["positions"], self.chrm_pos_dict[chrm_b])

        y = store_x[nm][chrm_a][chrm_b][:][np.ix_(i_a, i_b)]

        if not (np.all(np.equal(y, store[nuc_file][chrm_a][chrm_b][:]))):
          print(nuc_file, chrm_a, chrm_b)

    # a_pos =
    pass
    # Check names, chrms, and positions, raise if incompat.

  def loops_frame(self, pairs_dict):
    pass

  def domains_frame(self, pairs_dict):
    pass

  def _mk_cmap(self,
               chrm_pos_a,
               chrm_pos_b,
               bp_a,
               bp_b,
               counts,
               is_cis=False):
    @jit(nopython=True, nogil=True)
    def add_counts_to_cmap(cmap, idx_a, dx_b, counts, is_cis=is_cis):
      for i in range(idx_a.shape[0]):
        a = idx_a[i]
        b = idx_b[i]
        count = counts[i]
        cmap[a, b] += count
        if is_cis:
          cmap[b, a] += count
      return (cmap)

    idx_a, valid_a = bp_to_idx(bp_a, chrm_pos_a, self.bin_size)
    idx_b, valid_b = bp_to_idx(bp_b, chrm_pos_b, self.bin_size)

    valid = np.logical_and(valid_a, valid_b)
    idx_a = idx_a[valid]
    idx_b = idx_b[valid]
    counts = counts[valid]

    a_size = chrm_pos_a.shape[0]
    b_size = chrm_pos_b.shape[0]

    cmap = np.zeros((a_size, b_size))
    cmap = add_counts_to_cmap(cmap, idx_a, idx_b, counts)

    return (cmap)

  def create_pop_cmap(self, chrm_a, chrm_b, nuc_file):
    f = h5py.File(nuc_file, 'r')
    k = list(f["contacts"]["original"].keys())[0]
    contact_arr = f["contacts"]["original"][k][chrm_a][chrm_b][:].T

    # Get the chrm_pos dicts, then filter them
    chrm_pos_a = self.chrm_pos[chrm_a][:]
    chrm_pos_b = self.chrm_pos[chrm_b][:]

    is_cis = chrm_a == chrm_b
    cmap = self._mk_cmap(chrm_pos_a,
                         chrm_pos_b,
                         contact_arr[:, 0],
                         contact_arr[:, 1],
                         contact_arr[:, 2],
                         is_cis=is_cis)
    return (cmap)

  def create_sc_cmaps(self, nuc_name, chrm_a, chrm_b):
    contact_arr = self.contacts[nuc_name][chrm_a][chrm_b]
    chrm_pos_a = self.chrm_pos[chrm_a][:]
    chrm_pos_b = self.chrm_pos[chrm_b][:]
    is_cis = chrm_a == chrm_b
    cmap = self._mk_cmap(chrm_pos_a,
                         chrm_pos_b,
                         contact_arr[:, 0],
                         contact_arr[:, 1],
                         contact_arr[:, 2],
                         is_cis=is_cis)
    return (cmap)

  def obs_expected(self, genome_wide=False):
    """
    Calculate the obs/expected matrices for a given chromosome.
    Not IO bound -> go parallel.
    """

    futures = []
    max_size = np.max([self.chrm_pos[x].shape[0] for x in self.chrm_pos]) - 1

    with ProcessPoolExecutor() as executor:
      for name in self.nuc_names:
        chr_dists_cache = {c: np.median(self.dists[name, c, c],
                                        axis=2)
                           for c in self.chrms}
        futures.append(executor.submit(bg_dist_oe_worker, chr_dists_cache,
                                       max_size, self.chrms, name))
        print("submitted {}".format(name))

    for tup in as_completed(futures):
      name, sep_dists_dict = tup.result()
      for chrm in self.chrms:
        chr_dist_mat = self.dists[name, chrm, chrm]
        for size in np.arange(1, max_size):
          # Divide above diagonal
          row_idxs = np.arange(1, chr_dist_mat.shape[0] - size)
          col_idxs = row_idxs + size
          chr_dist_mat[row_idxs, col_idxs] /= sep_dists_dict[size]

          # Divide below diagonal
          row_idxs = np.arange(size, chr_dist_mat.shape[0])
          col_idxs = row_idxs - size
          chr_dist_mat[row_idxs, col_idxs] /= sep_dists_dict[size]
        cell_chr_path = os.path.join("dist_oe", name, chrm)
        try:
          del self.store[cell_chr_path]
        except KeyError:
          pass
        self.store.create_dataset(cell_chr_path, data=chr_dist_mat)
        print("Created {}, chrm {}".format(name, chrm))

    def load_contact_map(self, nuc_file, chrm_a, chrm_b):
      pass

  def bps_to_idx(self, chrm, bps):
    chrm_pos = self.chrm_pos[chrm][:]
    idxs, valid_idxs = bp_to_idx(bps, chrm_pos, self.bin_size)
    return (idxs, valid_idxs)


class Dists():
  def __init__(self, dataset):
    """
    dataset contains nuc_name/chr_a/chr_b
    """
    self.dataset = dataset

  def __getitem__(self, pos):
    name, a, b = pos
    try:
      return (self.dataset[name][a][b][:])
    except KeyError:
      return (self.dataset[name][b][a][:].T)


def bg_dist_oe_worker(chr_dist_cache, max_size, chrms, name):
  sep_dists_dict = {}

  for size in np.arange(1, max_size):
    size_dists = []
    for chrm in chrms:
      # chrm_dists_dset = chr_dist_cache[chrm]
      if size < chr_dist_cache[chrm].shape[0] - 1:
        dist_mat = chr_dist_cache[chrm]
        row_idxs = np.arange(1, dist_mat.shape[0] - size)
        col_idxs = row_idxs + size
        size_dists.append(dist_mat[row_idxs, col_idxs].flatten())
    sep_dists_dict[size] = np.median(np.hstack(size_dists))
  return (name, sep_dists_dict)


if __name__ == "__main__":
  import glob
  from sklearn.decomposition import PCA
  # nuc_files = glob.glob(
  #     "/home/lpa24/dev/cam/data/edl_chromo/mm10/single_cell_nuc_100k/*.nuc")

  # logging.basicConfig(level=logging.INFO

  # from os import remove
  # remove("/mnt/SSD/LayeredNuc/AmbigAvDistLayer.h5")
  # nuc_files = glob.glob(
  #     "/home/lpa24/dev/cam/data/edl_chromo/mm10/single_cell_nuc_100k/*_ambig_*.nuc")

  # nuc_files = [
  #            '/home/lpa24/dev/cam/data/edl_chromo/mm10/single_cell_nuc_100k/P2E8_ambig_10x_100kb.nuc', 
  #            '/home/lpa24/dev/cam/data/edl_chromo/mm10/single_cell_nuc_100k/P30E4_ambig_10x_100kb.nuc', 
  #            '/home/lpa24/dev/cam/data/edl_chromo/mm10/single_cell_nuc_100k/Q6_ambig_10x_100kb.nuc', 
  #            '/home/lpa24/dev/cam/data/edl_chromo/mm10/single_cell_nuc_100k/P2J8_ambig_10x_100kb.nuc', 
  #            '/home/lpa24/dev/cam/data/edl_chromo/mm10/single_cell_nuc_100k/P2I5_ambig_10x_100kb.nuc', 
  #            '/home/lpa24/dev/cam/data/edl_chromo/mm10/single_cell_nuc_100k/P30E8_ambig_10x_100kb.nuc', 
  #            '/home/lpa24/dev/cam/data/edl_chromo/mm10/single_cell_nuc_100k/UpL13_ambig_10x_100kb.nuc',
  #            '/home/lpa24/dev/cam/data/edl_chromo/mm10/single_cell_nuc_100k/P36D6_ambig_10x_100kb.nuc'
  # ]

  nf = NucFrame("/mnt/SSD/LayeredNuc/AmbigAvDistLayer.h5")
  #               nuc_files,
  #               av_dist=True)
  # print(nf.nuc_files)
  nf.depth_frame(recalc=True)
  # nf.depth_frame(recalc=False)

  # nf = NucFrame("/mnt/SSD/LayeredNuc/AvDistLayerStore.h5",
  #               nuc_files,
  #               av_dist=True)
  # nf.depth_frame(recalc=False)

  # cmap = nf.create_pop_cmap("12", "12", "/home/lpa24/dev/cam/data/edl_chromo/mm10/population_nuc/SLX-7671_hapsort_pop_mm10.nuc")
  # nf.create_sc_cmaps("12", "12")

  # nf.add_pop_contacts(
  # "/home/lpa24/dev/cam/data/edl_chromo/mm10/population_nuc/SLX-7671_hapsort_pop_mm10.nuc")
  # nf.obs_expected()

  # for name in nf.nuc_names:
  #     for chrm in nf.chrms:
  #         pca = PCA(n_components=2)
  #         dist_mat = np.mean(nf.dists[name][chrm][chrm][:], axis=2)
  #         print(dist_mat.shape)
  #         pca.fit(dist_mat)
  #         print(pca.explained_variance_ratio_)
  #         print(pca.components_.shape)
