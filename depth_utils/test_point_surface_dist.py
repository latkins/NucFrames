from hypothesis import given, assume, settings
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st

import numpy as np

from itertools import combinations

from point_surface_dist import point_project, point_tri_dists

@st.composite
def flat_tri(draw):
  """
  Triangle in the xy plane in general position.
  """
  tri = draw(arrays(np.int32, (3, 2),
  elements=st.integers(min_value=-1000, max_value=1000)))

  x1 = draw(st.integers(min_value=-1000, max_value=1000))
  x2 = draw(st.integers(min_value=-1000, max_value=1000))
  x3 = draw(st.integers(min_value=-1000, max_value=1000))

  assume(not x1 == x2)
  assume(not x2 == x3)
  assume(not x1 == x3)

  y1 = draw(st.integers(min_value=-1000, max_value=1000))
  y2 = draw(st.integers(min_value=-1000, max_value=1000))
  y3 = draw(st.integers(min_value=-1000, max_value=1000))

  assume(not y1 == y2)
  assume(not y2 == y3)
  assume(not y1 == y3)

  tri = np.array([[x1, y1, 1],
                  [x2, y2, 1],
                  [x3, y3, 1]], dtype=np.float32)

  assume(not check_line(tri[0], tri[1], tri[2]))

  assume(not np.all(tri[0] == tri[1]))
  assume(not np.all(tri[0] == tri[2]))
  assume(not np.all(tri[1] == tri[2]))

  return(tri)

def check_line(a, b, p):
  t = np.isclose((p[0] - a[0]) / (b[0] - a[0]), (p[1] - a[1]) / (b[1] - a[1]))
  return(t)


@given(flat_tri(), st.integers(min_value=-1000, max_value=1000))
def test_inner(flat_tri, height):
  """
  Test the projection is correct if a point is above a triangle.
  """
  height = float(height)
  # In the middle
  mid = np.mean(flat_tri, axis=0)
  mid[-1] += height
  p0 = mid

  p1 = flat_tri[0, :][None, :]
  p2 = flat_tri[1, :][None, :]
  p3 = flat_tri[2, :][None, :]

  p10 = p0 - p1
  p01 = p1 - p0
  p12 =  p2 - p1
  p13 =  p3 - p1

  p10_norm = np.linalg.norm(p10, axis=1)
  p01_norm = np.linalg.norm(p01, axis=1)

  np_ = np.cross(p12, p13, axis=1)
  np_norm = np.linalg.norm(np_, axis=1)

  _, _, dist = point_project(p0, p10, p01_norm, np_, np_norm)
  # Need to work out how to make this consistent in terms of sign.
  assert(np.allclose(np.abs(dist), np.abs([height])))

@given(flat_tri(), st.integers(min_value=-1000, max_value=1000))
def test_dists(flat_tri, height):
  height = float(height)
  # On a point
  for point in flat_tri:
    dists = point_tri_dists(flat_tri[None,:,:], point[None, :])

  for (a, b) in combinations(range(flat_tri.shape[0]), 2):
    # In an edge
    e = flat_tri[a] - flat_tri[b]
    e_mid = e / 2
    dists = point_tri_dists(flat_tri[None,:,:], e_mid[None, :])
    print(dists)

  # Above an edge
  e_mid[-1] += height
  dists = point_tri_dists(flat_tri[None,:,:], e_mid[None, :])
  print(dists, height)

  # In the middle
  mid = np.mean(flat_tri, axis=0)
  dists = point_tri_dists(flat_tri[None,:,:], mid[None, :])
  print(dists)

  # Above the middle
  mid[-1] += height
  dists = point_tri_dists(flat_tri[None,:,:], mid[None, :])
  print(dists, height)

if __name__=="__main__":
  with settings(max_examples=100000):
    test_inner()
    test_dists()
