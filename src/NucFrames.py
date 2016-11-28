import numpy as np
from collections import defaultdict
import h5py
from tqdm import tqdm

from .NucFrame import NucFrame
from .utils import bp_to_idx

class NucFrames(object):
  def __init__(self, nuc_frm_path_list):
    self.nuc_frm_path_list = nuc_frm_path_list
    self.bin_size = 100e3 # Sort this out. For now it is reasonable.
    self.chrm_limit_dict = None
    self.common_chrms = self._calc_common_chrms()
    self.chrm_limit_dict = self._calc_chrm_limit_dict()
    self.tracks = {}

  def _calc_chrm_limit_dict(self):
    """
    Read all nuc frames, and find the common limits for each chromosome.
    """
    starts_dict = defaultdict(lambda: 0)
    ends_dict = defaultdict(lambda: np.inf)
    for nf in self:
      for chrm in self.common_chrms:
        prev_start = starts_dict[chrm]
        prev_end = ends_dict[chrm]

        nf_bps = nf.chrms[chrm].bp_pos
        if nf_bps[0] > prev_start:
          starts_dict[chrm] = nf_bps[0]
        if nf_bps[-1] < prev_end:
          ends_dict[chrm] = nf_bps[-1]
    return({ chrm: (starts_dict[chrm], ends_dict[chrm]) for chrm in self.common_chrms})

  def _calc_common_chrms(self):
    chrms = None
    for nf in self:
      nf_chrms = {x.chrm for x in nf.chrms}
      if chrms is None:
        chrms = nf_chrms
      chrms = chrms.intersection(nf_chrms)
    return(chrms)

  def _load_nuc_frame(self, key):
    nuc_frm_path = self.nuc_frm_path_list[key]
    return(NucFrame(nuc_frm_path, self.chrm_limit_dict))

  def load_nuc_tracks(self, nuc_file, nuc_track_name, track_name=None, binary=True):
    if track_name is None:
      track_name = nuc_track_name

    nuc_file = h5py.File(nuc_file)
    track = nuc_file["dataTracks"]["external"][nuc_track_name]

    track_dict = {}

    for chrm in tqdm(set(self.common_chrms) & set(track.keys())):
      chrm_bps = self.chrm_bps(chrm)
      regions = track[chrm]["regions"][:].astype(np.int32)
      values = track[chrm]["values"][:, 0]

      vals = np.zeros(chrm_bps.shape, dtype=np.float64)
      counts = np.zeros(chrm_bps.shape, dtype=np.float64)

      starts = regions[:, 0]
      ends = regions[:, 1]
      a = np.argmax(np.abs(ends - starts))
      for value_idx, (start, end) in enumerate(zip(starts, ends)):
        if start > end:
          start, end = end, start
        track_region_bps = np.arange(start, end, 1000, dtype=np.int32) # TODO: Hard coding here is bad.
        idxs, valid = bp_to_idx(track_region_bps, chrm_bps, bin_size=self.bin_size)
        idxs = np.unique(idxs[valid])
        if binary:
          vals[idxs] = 1
        else:
          vals[idxs] += values[value_idx]
          counts[idxs] += (values[value_idx] != 0)
      if binary:
        vals = vals.astype(np.int32)
      else:
        counts_mask = counts != 0
        vals[counts_mask] = vals[counts_mask] / counts[counts_mask]

      track_dict[chrm] = vals

    self.tracks[track_name] = track_dict

  def chrm_bps(self, chrm):
    chrm_start, chrm_end = self.chrm_limit_dict[chrm]
    chrm_bps = np.arange(chrm_start, chrm_end, self.bin_size, dtype=np.int32)
    return(chrm_bps)

  def chrm_bp_to_idx(self, chrm, bps):
    chrm_bps = self.chrm_bps(chrm)
    idxs, valid = bp_to_idx(bps, chrm_bps, bin_size=self.bin_size)
    return (idxs, chrm_bps, valid)

  def valid_idxs(self, all_positions, lower, upper):
    # This is copied from Chromosome. Could be more elegant. Avoid reuse.
    if lower and upper:
      return(np.logical_and(all_positions >= lower, all_positions < upper))
    elif lower and not upper:
      return(all_positions >= lower)
    elif not lower and upper:
      return(all_positions < upper)
    else:
      return(~np.isnan(all_positions))


  def __getitem__(self, key):
    if isinstance(key, slice):
      return ([self[ii] for ii in range(*key.indices(len(self)))])
    elif isinstance(key, int):
      if key < 0:
        key += len(self)
      if key < 0 or key >= len(self):
        raise IndexError("The index {} is out of range.".format(key))
      return(self._load_nuc_frame(key))
    else:
      raise TypeError("Invalid argument type, {}".format(type(key)))

  def __len__(self):
    return(len(self.nuc_frm_path_list))
  def __iter__(self):
    for i, nf in enumerate(self.nuc_frm_path_list):
      yield(self._load_nuc_frame(i))

# Cis and trans classes?
