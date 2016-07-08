from scipy.spatial import Delaunay, cKDTree
import numpy as np
from itertools import combinations
from collections import defaultdict

from .circumsphere import simplex_circumsphere

def alpha_shape(points):
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

  kdTree = cKDTree(coords)

  d = len(tri.simplices[0])
  ch_lookup = set()
  for simplex in tri.convex_hull:
    for k in range(d, 1, -1):
      for x in combinations(simplex, k):
        ch_lookup.add(frozenset(x))

  # Values of a and b for each simplex.
  simplex_a = {}
  simplex_b = {}

  for simplex in tri.simplices:
    simp_set = frozenset(simplex)
    _, sigma = simplex_circumsphere(coords[simplex])
    simplex_a[simp_set] = sigma
    simplex_b[simp_set] = sigma

  sslu = SuperSimplexLookup(tri.simplices)
  for simplex in sslu.simplices:
    centre, r = simplex_circumsphere(coords[np.array(list(simplex))])
    ball_points = kdTree.query_ball_point(centre, r)
    if len(ball_points) == 0:
      simplex_a[simplex] = r
    else:
      super_simplices = {s
                        for s in sslu.lookup[simplex]
                        if len(s) > len(simplex)}
      a_vals = set()
      for super_simplex in super_simplices:
        a = simplex_a[super_simplex]
        a_vals.add(a)
        # b = simplex_b[super_simplex]
      simplex_a[simplex] = min(a_vals)

    if simplex in ch_lookup:
      simplex_b[simplex] = np.inf
    else:
      super_simplices = {s
                        for s in sslu.lookup[simplex]
                         if len(s) == d}
      b_val = max({simplex_b[s] for s in super_simplices})
      simplex_b[simplex] = b_val

  return(simplex_a, simplex_b)


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
