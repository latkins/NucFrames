
class NucSlice:
  """
  Class to represent a Nuc file, in a consistent fashion.


  -- Derived from the structure of a .nuc file.
  -- Used by a NucFrame
  -- Calculates the maximum information possible (e.g. all distances)

  """

  def __init__(self, nuc_file=None, nuc_slice_file=None):
    """Passing only nuc_slice_file will read that file from disk.
    Passing both will create NucSlice at the specified location.
    """

    if nuc_file and nuc_slice_file is False:
      raise TypeError("Neither nuc_file or nuc_slice_file were specified")

    if nuc_file and not nuc_slice_file:
      raise TypeError("No nuc_slice_file specified.")

    if nuc_slice_file and not nuc_file:
      # Read nuc_slice_file
      pass
    else:
      # Create new nuc_slice_file from nuc_file
      pass

