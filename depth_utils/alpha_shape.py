from scipy.spatial import Delaunay, ConvexHull
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
  coords = tri.points
  ch = tri.convex_hull

  # Values of a and b for each simplex.
  simplex_a = {}
  simplex_b = {}

  for simplex in tri.simplices:
    simp_set = frozenset(simplex)
    sigma = circumsphere(coords[simplex])
    simplex_a[simp_set] = None
    simplex_b[simp_set] = (sigma, np.inf)
  print("done")

  # Need lookup from simplex to ALL higher simplicies.

  # print(tri.simplices.shape)
  simplices = _2d_simplices(tri.simplices)
  for simplex in simplices:
    print(simplex)
    for x in simplex:
      idx = (tri.vertex_to_simplex[x])
      print(x, idx, tri.simplices[idx])

    break
  print(len(simplices))


def _2d_simplices(qhull_simplices):
  """
  Each qhull simplex (in R^3) is a length 4 list of indices.
  We want to convert this into the unique list of length 3 vertices.
  """
  simplices = set()
  for simplex in qhull_simplices:
    for x in combinations(simplex, 3):
      simplices.add(frozenset(x))
    for x in combinations(simplex, 2):
      simplices.add(frozenset(x))

  return(simplices)

def filter_simplicies(simplices, alpha):
  """
  Given a list of 2-simplices (aka 3 points in R^3), filter those whose circumsphere is empty
  for radius alpha.
  """
  pass

def circumcircle(coords):
  """
  Calculate the circumradius and circumcenter of a triangle in R^3
  """
  A, B, C = coords
  a = A - C
  b = B - C

  ab_diff = a - b

  ab_cross = np.cross(a, b)
  norm_ab_cross = np.sqrt(ab_cross.dot(ab_cross))

  r = (np.sqrt(a.dot(a)) *
       np.sqrt(b.dot(b)) *
       np.sqrt(ab_diff.dot(ab_diff)))  / (2 * norm_ab_cross)
  return(r)


def circumsphere(coords):
  """
  Calculate the circumradius and circumcenter of a tetrahedron in R^3.
  coords :: (4, 3)
  """

  sqsum = np.sum(np.square(coords), axis=1)
  ones = np.ones(4, dtype=np.float64)

  a = np.linalg.det(np.concatenate([coords, ones[:, np.newaxis]], axis=1))
  Dx = np.linalg.det(np.stack([sqsum, coords[:, 1], coords[:, 2], ones], axis=1))
  Dy = -1 * np.linalg.det(np.stack([sqsum, coords[:, 0], coords[:, 2], ones], axis=1))
  Dz = np.linalg.det(np.stack([sqsum, coords[:, 0], coords[:, 1], ones], axis=1))
  c = np.linalg.det(np.concatenate([sqsum[:, np.newaxis], coords], axis=1))

  r = np.sqrt(Dx**2 + Dy**2 + Dz**2 - 4 * a * c) / (2 * abs(a))

  return(r)


if __name__=="__main__":
  np.random.seed(42)
  points = np.random.random((1000, 3))
  alpha_shape(points, 0.5)
