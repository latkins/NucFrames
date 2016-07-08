import numpy as np
from hypothesis import given
import hypothesis.strategies as s
from itertools import combinations

from alpha_shape import circumsphere, circumcircle

def regular_tetrahedron_coords(radius):
  """
  For a given radius, calculate a regular tetrahedron with a
  circumsphere of that radius.
  The calculated circumsphere should equal the original radius.
  """
  a = (4 * radius) / np.sqrt(6)

  return(np.array([[1/3 * np.sqrt(3) * a, 0, 0],
                   [-1/6 * np.sqrt(3) * a, 0.5 * a, 0],
                   [-1/6 * np.sqrt(3) * a, -0.5 * a, 0],
                   [0, 0, 1/3 * np.sqrt(6) * a]], np.float64))

@given(s.floats(min_value=0.1, max_value=100000, allow_nan=False, allow_infinity=False))
def test_circumsphere_sphere(radius):
  coords = regular_tetrahedron_coords(radius)

  result = circumsphere(coords)

  np.testing.assert_almost_equal(result, radius, decimal=5)

  tri_max = 0
  for indices in combinations(range(4), 3):
    tri = coords[indices, :]
    tri_alpha = circumcircle(tri)
    if tri_alpha > tri_max:
      tri_max = tri_alpha

  # np.testing.assert_almost_equal(tri_max, radius, decimal=5)
  assert tri_max <= radius


if __name__=="__main__":
  test_circumsphere_sphere()
