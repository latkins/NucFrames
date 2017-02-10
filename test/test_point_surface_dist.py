import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from hypothesis import given, assume, settings
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st

import numpy as np

from itertools import combinations

from .point_surface_dist import points_tris_dists

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
def inner(flat_tri, height):
  """
  Test the projection is correct if a point is above a triangle.
  """
  height = float(height)
  # In the middle
  mid = np.mean(flat_tri, axis=0)
  point = mid.copy()
  point[-1] += height

  dist = points_tris_dists(flat_tri[None, :], point[None, :])

  assert(np.allclose(np.abs(dist), np.abs([height])))


@given(flat_tri(), st.integers(min_value=-1000, max_value=1000))
def dists(flat_tri, height):
  height = float(height)

  # All triangles have z = 1 to stop some errors.
  # On a point
  for point in flat_tri:
    dists = points_tris_dists(flat_tri[None,:,:], point[None, :])
    assert(np.allclose(np.abs(dists), 0))

  for (a, b) in combinations(range(flat_tri.shape[0]), 2):
    # In an edge
    e_mid = np.mean(np.array([flat_tri[a], flat_tri[b]]), axis=0)
    dists = points_tris_dists(flat_tri[None,:,:], e_mid[None, :])
    assert(np.allclose(np.abs(dists), 0))

    # Above an edge
    e_mid[-1] += height
    dists = points_tris_dists(flat_tri[None,:,:], e_mid[None, :])
    assert(np.allclose(np.abs(dists), np.abs([height])))

    # Outside an edge
    # ...

  # In the middle
  mid = np.mean(flat_tri, axis=0)
  print(mid)
  dists = points_tris_dists(flat_tri[None,:,:], mid[None, :])
  assert(np.allclose(np.abs(dists), 0))

  # Above the middle
  mid[-1] += height
  dists = points_tris_dists(flat_tri[None,:,:], mid[None, :])
  assert(np.allclose(np.abs(dists), np.abs([height])))


def test_region0():
  facets = np.array([[[0, 0, 0],
                      [2, 0, 0],
                      [0, 2, 0]]], dtype=np.float32)

  # In plane
  points = np.array([[1, 1, 0]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 0.0

  points = np.array([[0, 0, 0]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 0.0

  points = np.array([[2, 0, 0]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 0.0

  points = np.array([[0, 2, 0]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 0.0

  # Above
  points = np.array([[1, 1, 2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 2.0

  points = np.array([[0, 0, 2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 2.0

  points = np.array([[2, 0, 2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 2.0

  points = np.array([[0, 2, 2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 2.0

  # Below
  points = np.array([[1, 1, -2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 2.0

  points = np.array([[0, 0, -2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 2.0

  points = np.array([[2, 0, -2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 2.0

  points = np.array([[0, 2, -2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 2.0


def test_region1():
  facets = np.array([[[0, 0, 0],
                      [2, 0, 0],
                      [0, 2, 0]]], dtype=np.float32)

  # On edge
  points = np.array([[1, 1, 0]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 0

  # In plane
  points = np.array([[2, 2, 0]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == np.sqrt(2.0)

  # Above
  points = np.array([[2, 2, 2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  d = np.linalg.norm(points - np.array([1,1,0]))
  assert dists == d

  # Below
  points = np.array([[2, 2, -2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  d = np.linalg.norm(points - np.array([1,1,0]))
  assert dists == d


def test_region2():
  facets = np.array([[[0, 0, 0],
                      [2, 0, 0],
                      [0, 2, 0]]], dtype=np.float32)


  # On point
  points = np.array([[0, 2, 0]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 0

  # In plane
  points = np.array([[-1, 4, 0]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  d = np.linalg.norm(points - np.array([0,2,0]))
  assert dists == d

  # Above
  points = np.array([[-1, 4, 1]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  d = np.linalg.norm(points - np.array([0,2,0]))
  assert dists == d

  # Below
  points = np.array([[-1, 4, -1]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  d = np.linalg.norm(points - np.array([0,2,0]))
  assert dists == d


def test_region3():
  facets = np.array([[[0, 0, 0],
                      [2, 0, 0],
                      [0, 2, 0]]], dtype=np.float32)

  # On edge
  points = np.array([[0, 1, 0]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 0

  # In plane
  points = np.array([[-1, 1, 0]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 1.0

  # Above
  points = np.array([[-1, 1, 2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  d = np.linalg.norm(points - np.array([0,1,0]))
  assert dists == d

  # Below
  points = np.array([[-1, 1, -2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  d = np.linalg.norm(points - np.array([0,1,0]))
  assert dists == d


def test_region4():
  facets = np.array([[[0, 0, 0],
                      [2, 0, 0],
                      [0, 2, 0]]], dtype=np.float32)


  # On point
  points = np.array([[0, 0, 0]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 0

  # In plane
  points = np.array([[-1, -1, 0]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == np.sqrt(2)

  # Above
  points = np.array([[-1, -1, 2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  d = np.linalg.norm(points - np.array([0,0,0]))
  assert dists == d

  # Below
  points = np.array([[-1, -1, -2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  d = np.linalg.norm(points - np.array([0,0,0]))
  assert dists == d


def test_region5():
  facets = np.array([[[0, 0, 0],
                      [2, 0, 0],
                      [0, 2, 0]]], dtype=np.float32)

  # On edge
  points = np.array([[1, 0, 0]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 0

  # In plane
  points = np.array([[1, -1, 0]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 1.0

  # Above
  points = np.array([[1, -1, 2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  d = np.linalg.norm(points - np.array([1,0,0]))
  assert dists == d

  # Below
  points = np.array([[1, -1, -2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  d = np.linalg.norm(points - np.array([1,0,0]))
  assert dists == d


def test_region6():
  facets = np.array([[[0, 0, 0],
                      [2, 0, 0],
                      [0, 2, 0]]], dtype=np.float32)


  # On point
  points = np.array([[2, 0, 0]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  assert dists == 0

  # In plane
  points = np.array([[3, -0.5, 0]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  d = np.linalg.norm(points - np.array([2,0,0]))
  assert dists == d

  # Above
  points = np.array([[3, -0.5, 2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  d = np.linalg.norm(points - np.array([2,0,0]))
  assert dists == d

  # Below
  points = np.array([[3, -0.5, -2]], dtype=np.float32)
  dists = points_tris_dists(facets, points)
  d = np.linalg.norm(points - np.array([2,0,0]))
  assert dists == d

