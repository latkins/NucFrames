from numba import jit
import numpy as np


def filter_bps(bp_arr, pos_arr, bin_size):
  """
    :: [bp] -> [bp] -> [idx]
    """
  low = pos_arr[0]
  up = pos_arr[-1] + bin_size
  idxs = np.logical_and(bp_arr >= low, bp_arr < up)
  return (idxs)


@jit("float32[:](int32[:], int32[:], double)", nopython=True, nogil=True, cache=True)
def _bps_to_idx(bps, positions, bin_size):

  idxs = np.empty(bps.shape[0], dtype=np.float32)
  idxs[:] = np.nan
  for i in range(bps.shape[0]):
    bp = bps[i]
    for j in range(positions.shape[0]):
      pos = positions[j]
      if (bp >= pos) and (bp < pos + bin_size):
        idxs[i] = j
        break
  return (idxs)


def bp_to_idx(bp_arr, pos_arr, bin_size):
  """
    :: [bp] -> [bp] -> bp -> ([idx], [valid_idx_idx])
    Array of idxs will be float, because int can't be nan.
    Return the array of index values for the bp_arr,
    with nan in invalid entries.
    Also return a list of valid indexes.
    """
  idx_arr = _bps_to_idx(bp_arr, pos_arr, bin_size)
  valid_input_idxs = ~np.isnan(idx_arr)

  return (idx_arr.astype(np.int32), valid_input_idxs)


if __name__ == "__main__":
  bin_size = 1
  bp_arr = np.array([3, 4, 5, 6, 7, 10])
  pos_arr = np.array([0, 1, 5, 6, 7, 9, 10])

  print(bp_to_idx(bp_arr, pos_arr, bin_size))
