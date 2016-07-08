from scipy.spatial import Delaunay, ConvexHull
import numpy as np
from itertools import combinations
from collections import defaultdict


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

  ch_lookup = SuperSimplexLookup(tri.convex_hull)

  # Values of a and b for each simplex.
  simplex_a = {}
  simplex_b = {}

  for simplex in tri.simplices:
    simp_set = frozenset(simplex)
    sigma = circumsphere(coords[simplex])
    simplex_a[simp_set] = sigma
    simplex_b[simp_set] = sigma

  sslu = SuperSimplexLookup(tri.simplices)
  for simplex in sslu.simplices:
    super_simplices = {s
                       for s in sslu.lookup[simplex]
                       if len(s) > len(simplex)}
    a_vals = set()
    for super_simplex in super_simplices:
      a = simplex_a[super_simplex]
      a_vals.add(a)
      # b = simplex_b[super_simplex]
    simplex_a[simplex] = min(a_vals)

    # try:
    #   ch_lookup.lookup[simplex]
    # except KeyError:
    #   super_simplices = {s
    #                      for s in super_simplices
    #                      if len(s) == len(tri.convex_hull[0])}
    #   simplex_b[simplex] = max({simplex_b[x] for x in super_simplices})
    #   pass
    # else:
    #   simplex_b[simlpex] = np.inf

  print(len(sslu.simplices), simplex_a)


class SuperSimplexLookup(object):
  def __init__(self, d_simplices):
    """
    Given a list of d_simplicies, create all k simplicies (k < d),
    and a lookup from a simplex to all supersimplicies.
    """
    _simplices, self.lookup = self.k_simplices(d_simplices)
    # Need to sort as we want largest K at the start,
    # since the algorithm relies on this fact.
    d = len(d_simplices[0])
    self.simplices = sorted(
        filter(lambda x: len(x) < d, _simplices),
        key=lambda x: len(x),
        reverse=True)

  def k_simplices(self, d_simplices):
    """
    Each qhull simplex (in R^3) is a length 4 list of indices.
    We want to convert this into the unique list of length 3 vertices.
    """
    lookup = defaultdict(lambda: set())
    simplices = set()
    d = len(d_simplices[0])
    for simplex in d_simplices:
      simplex_key = frozenset(simplex)
      for k in range(d, 1, -1):
        for x in combinations(simplex, k):
          frozen_subsimplex = frozenset(x)
          # Add lookup simplex -> supersimplex.
          lookup[frozen_subsimplex].add(simplex_key)
          # Add subsimplex to list of simplices.
          simplices.add(frozen_subsimplex)

    return (simplices, lookup)


def circumcircle(coords):
  """
  Calculate the circumradius and circumcenter of a triangle in R^3
  """
  A, B, C = coords
  a = A - C
  b = B - C

  def mag(x):
    return (np.sqrt(x.dot(x)))

  ab_diff = a - b

  ab_cross = np.cross(a, b)
  norm_ab_cross = mag(ab_cross)

  r = (mag(a) * mag(b) * mag(ab_diff)) / (2 * norm_ab_cross)

  centre = (np.cross(
      np.square(mag(a)) * b - np.square(mag(b)) * a, ab_cross) /
            (2 * np.square(norm_ab_cross))) + C

  return (centre, r)


def circumsphere(coords):
  """
  Calculate the circumradius and circumcenter of a tetrahedron in R^3.
  coords :: (4, 3)
  """

  sqsum = np.sum(np.square(coords), axis=1)
  ones = np.ones(4, dtype=np.float64)

  a = np.linalg.det(np.concatenate([coords, ones[:, np.newaxis]], axis=1))
  Dx = np.linalg.det(np.stack(
      [sqsum, coords[:, 1], coords[:, 2], ones],
      axis=1))
  Dy = -1 * np.linalg.det(np.stack(
      [sqsum, coords[:, 0], coords[:, 2], ones],
      axis=1))
  Dz = np.linalg.det(np.stack(
      [sqsum, coords[:, 0], coords[:, 1], ones],
      axis=1))
  c = np.linalg.det(np.concatenate([sqsum[:, np.newaxis], coords], axis=1))

  r = np.sqrt(Dx**2 + Dy**2 + Dz**2 - 4 * a * c) / (2 * abs(a))

  denom = d * a
  centre = np.array([Dx / denom, Dy / denom, Dz / denom])

  return (centre, r)


if __name__ == "__main__":
  np.random.seed(42)
  points = np.random.random((3, 3))
  # alpha_shape(points, 0.5)
  print(circumcircle(points))
