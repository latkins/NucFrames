from scipy.spatial import Delaunay
import numpy as np
from itertools import combinations

def alpha_shape(points, alpha):
  """
  VALID ONLY IN 3D.

  1. Calculate Delaunay triangulation of points
  2. Inspect all simplicies from triangulation, accept those whose 
  circumsphere is empty with radius smaller than alpha.
  3. All simplicies

  """
  # Annoyingly this merges all simplicies. We actually want 2 dimensional
  # simplices, but get 3 dimentional ones. So, we must convert.
  tri = Delaunay(points, qhull_options="QJ Pp")
  # print("done")
  # simplices = _2d_simplices(tri.simplices.copy())
  # print(simplices)
  print(tri.simplices.copy())
  for simplex in tri.simplices:
    coords = points[simplex]
    circumradius, circumcentre = circumsphere(coords)
    # print(circumradius < alpha)
  pass

def _2d_simplices(qhull_simplices):
  """
  Each qhull simplex (in R^3) is a length 4 list of indices.
  We want to convert this into the unique list of length 3 vertices.
  """
  simplices = set()
  for simplex in qhull_simplices:
    simplices = simplices.union({tuple(sorted(x)) for x in (combinations(simplex, 3))})
  return(simplices)

def filter_simplicies(simplices, alpha):
  """
  Given a list of 2-simplices (aka 3 points in R^3), filter those whose circumsphere is empty
  for radius alpha.
  """
  pass

def circumcircle(coords):
  A, B, C = coords
  a = A - C
  b = B - C
  pass

def circumsphere(coords):
  """
  Calculate the circumradius and circumcenter of a tetrahedron in R^3.
  coords :: (4, 3)
  """
  xyz1sq = np.sum(np.square(coords[0, :]))
  xyz2sq = np.sum(np.square(coords[1, :]))
  xyz3sq = np.sum(np.square(coords[2, :]))
  xyz4sq = np.sum(np.square(coords[3, :]))

  sqsum = np.sum(np.square(coords), axis=1)
  ones = np.ones(4, dtype=np.float64)

  Dx = np.linalg.det(np.stack([sqsum, coords[:, 1], coords[:, 2], ones], axis=1))

  Dy = -1 * np.linalg.det(np.stack([sqsum, coords[:, 0], coords[:, 2], ones], axis=1))

  Dz = np.linalg.det(np.stack([sqsum, coords[:, 0], coords[:, 1], ones], axis=1))

  a = np.linalg.det(np.concatenate([coords, ones[:, np.newaxis]], axis=1))

  c = np.linalg.det(np.concatenate([sqsum[:, np.newaxis], coords], axis=1))

  r = np.sqrt(Dx**2 + Dy**2 + Dz**2 - 4 * a * c) / (2 * abs(a))


  return(r)


if __name__=="__main__":
  np.random.seed(42)
  points = np.random.random((10, 3))
  alpha_shape(points, 1.5)
