import numpy as np
import pandas as pd


def rog(coords):
  """
  Calculate the radius of gyration for a set of coordinates.
  """
  n, dim = coords.shape
  mean = np.mean(coords, axis=0)
  x = coords - mean
  return(np.square(x).sum() / n)

def tad_set_monte_carlo(nf, seps, B):
  pass

def tad_monte_carlo(nf, tad_frm, B):
  new_tad_frms = []

  tad_per_cell_frms = []

  bg_dict = {}
  tad_dict = {}
  for chrm, df in tad_frm.groupby(["chrm"]):
    df = df.reset_index()
    max_idx = nf.coords(nf.nuc_files[0], chrm).shape[0]

    tad_starts = df.start_idx.values
    tad_ends = df.end_idx.values

    seps = tad_ends - tad_starts

    # Get the sum of rog values for each TAD, across cells.
    tad_values = np.zeros(tad_starts.shape[0])
    for nuc_name, nuc_file in zip(nf.nuc_names, nf.nuc_files):
      coords = nf.coords(nuc_file, chrm)
      for i in range(tad_starts.shape[0]):
        cell_tad_rog_val = rog(coords[tad_starts[i]:tad_ends[i]])

        tad_per_cell_frms.append({
          "cell": nuc_name,
          "start_bp": df.ix[i, "start_bp"],
          "end_bp": df.ix[i, "end_bp"],
          "start_idx": df.ix[i, "start_idx"],
          "end_idx": df.ix[i, "end_idx"],
          "rog": cell_tad_rog_val,
          "chrm": chrm
        })

        tad_values[i] += cell_tad_rog_val

    tad_values = np.array(tad_values)
    tad_dict[chrm] = tad_values


    background = np.zeros((len(seps), B), dtype=np.float64)
    # Shape (num_tads, num_samples)
    bg_start, bg_end = random_sample_(seps, max_idx, B)
    # Calculate the background rog values.
    for nuc_file in nf.nuc_files:
      coords = nf.coords(nuc_file, chrm)
      for sep_i in range(bg_start.shape[0]):
        sample_start = bg_start[sep_i, :]
        sample_end = bg_end[sep_i, :]
        for j in range(sample_start.shape[0]):
          start = sample_start[j]
          end = sample_end[j]
          background[sep_i, j] += rog(coords[start:end])
    bg_dict[chrm] = background

    tad_pvals = []
    # How often do we see a background sample with summed ROG less than
    # what that of the TAD ?
    for tad_idx in range(tad_values.shape[0]):
      truths = np.sum(background[tad_idx] <= tad_values[tad_idx])

      p = (1 + truths) / (1 + B)
      tad_pvals.append(p)

    df["rog_pval"] = tad_pvals
    df["tad_rog_sum"] = tad_values
    new_tad_frms.append(df)

  tad_per_cell_frm = pd.DataFrame(tad_per_cell_frms)
  tad_frm = pd.concat(new_tad_frms)
  return(tad_dict, bg_dict, tad_frm, tad_per_cell_frm)


def random_sample_(seps, max_idx, B):
  a, b = random_sample(seps, max_idx, B)
  return(a.T, b.T)

def random_sample(seps, max_idx, B):
  """
  Given a list of integer separations, a maximum index, and a number of samples,
  randomly sample B times.
  """
  samples_a = []
  samples_b = []
  for sep in seps:
    starts = np.random.random_integers(0, max_idx - sep, B)
    ends = starts + sep
    samples_a.append(starts)
    samples_b.append(ends)

  # Transpose to fit with circular permute
  samples_a = np.array(samples_a).T
  samples_b = np.array(samples_b).T
  return (samples_a, samples_b)


