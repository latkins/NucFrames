from scipy.spatial import Delaunay, cKDTree
import numpy as np
from itertools import combinations, permutations
from collections import defaultdict

from functools import lru_cache

from .circumsphere import simplex_circumsphere

@lru_cache(maxsize=128)
def _mk_idxs(input_list_len, size):
  return([sorted(list(x)) for x in combinations(range(input_list_len), size)])

def circular_subgroup(input_list, size):
  """
  list: (n, )
  size: int
  Return circular subgroups of list, of size.

  Example: list = [ a, b, c, d ], size = 3
  returns ([ (a, b, c), (b, c, d), (a, c, d), (a, b, d) ])
  """
  input_list = np.asarray(input_list)
  idxs = _mk_idxs(len(input_list), size)

  return [ tuple(input_list[list(idx)]) for idx in idxs ]

class AlphaShape(object):

  @classmethod
  def from_points(cls, points):
    interval_dict, coords = cls.alpha_intervals(cls, points)
    return(cls(interval_dict, points))

  def __init__(self, interval_dict, coords):
    self.interval_dict = interval_dict
    self.coords = coords

  def get_facets(self, alpha, k=3):
    """
    Given an alpha value, return the vertices that make up the
    corresponding surface.
    """
    valid_simplices = {v for v, (a, b) in self.interval_dict.items() if (a <= alpha and b >= alpha)}

    # Consider only k-simplices
    valid_simplices = filter(lambda x: len(x) == k, valid_simplices)

    return(valid_simplices)


  def alpha_intervals(self, points):
    """
    CURRENTLY VALID ONLY IN 3D.
    Following Edelsbrunner:1994bg, calculate the a/b values that define the intervals B, I, for each simplex.
    For each simplex, delta_T:
      Iff alpha isin B_T, then delta_T is on the surface of the alpha shape.
      Iff alpha isin I_T, then delta_T is in the interior of the alpha shape.
    """
    tri = Delaunay(points, qhull_options="QJ Pp")
    coords = tri.points

    kdTree = cKDTree(coords)

    d = len(tri.simplices[0])
    ch_lookup = set()
    for simplex in tri.convex_hull:
      for k in range(d, 1, -1):
        for x in circular_subgroup(simplex, k):
          ch_lookup.add(x)

    # Values of a and b for each simplex.
    simplex_ab = {}

    for simplex in tri.simplices:
      simp_set = tuple(simplex)
      _, sigma = simplex_circumsphere(coords[simplex])
      simplex_ab[simp_set] = (sigma, sigma)

    sslu = SuperSimplexLookup(tri.simplices)
    for simplex in sslu.simplices:
      centre, r = simplex_circumsphere(coords[np.array(list(simplex))])
      ball_points = kdTree.query_ball_point(centre, r)

      if len(ball_points) == 0:
        simplex_a_value = r
      else:
        super_simplices = {s
                          for s in sslu.lookup[simplex]
                          if len(s) > len(simplex)}
        a_vals = set()
        for super_simplex in super_simplices:
          (a, _) = simplex_ab[super_simplex]
          a_vals.add(a)
        simplex_a_value = min(a_vals)

      if simplex in ch_lookup:
        simplex_b_value = np.inf
      else:
        super_simplices = {s
                          for s in sslu.lookup[simplex]
                          if len(s) == d}
        simplex_b_value = max({simplex_ab[s][1] for s in super_simplices})

      simplex_ab[simplex] = (simplex_a_value, simplex_b_value)

    return(simplex_ab, coords)


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
      simplex_key = tuple(simplex)
      for k in range(d, 1, -1):
        for subsimplex in circular_subgroup(simplex, k):
          # Add lookup simplex -> supersimplex.
          lookup[subsimplex].add(simplex_key)
          # Add subsimplex to list of simplices.
          simplices.add(subsimplex)

    return (simplices, lookup)
