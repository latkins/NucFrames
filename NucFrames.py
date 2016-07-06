import numpy as np
from collections import defaultdict

from NucFrame import NucFrame

class NucFrames(object):
  def __init__(self, nuc_frm_path_list):
    self.nuc_frm_path_list = nuc_frm_path_list

    self.chrm_limit_dict = None
    self.common_chrms = self._calc_common_chrms()
    self.chrm_limit_dict = self._calc_chrm_limit_dict()

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

  def load_nuc_frame(self, key):
    nuc_frm_path = self.nuc_frm_path_list[key]
    return(NucFrame(nuc_frm_path, self.chrm_limit_dict))

  def __getitem__(self, key):
    if isinstance(key, slice):
      return ([self[ii] for ii in range(*key.indices(len(self)))])
    elif isinstance(key, int):
      if key < 0:
        key += len(self)
      if key < 0 or key >= len(self):
        raise IndexError("The index {} is out of range.".format(key))
      return(self.load_nuc_frame(key))
    else:
      raise TypeError("Invalid argument type, {}".format(type(key)))

  def __iter__(self):
    for i, nf in enumerate(self.nuc_frm_path_list):
      yield(self.load_nuc_frame(i))

# Cis and trans classes?


if __name__=="__main__":
  nfs = NucFrames(["/mnt/SSD/LayeredNuc/Q5_ambig_100kb.hdf5"])
  print(nfs.common_chrms)

