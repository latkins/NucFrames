import numba as nb
import numpy as np
from numpy.core.umath_tests import inner1d

@nb.jit(
  nb.types.Tuple((nb.float32, nb.float32, nb.float32))(
    nb.float32, nb.float32, nb.float32, nb.float32, nb.float32,
    nb.float32, nb.float32, nb.float32, nb.float32),
  nopython=True,
  cache=True,
  nogil=True)
def reg0(s, t, det, a, b, c, d, e, f):
  invDet = 1.0 / det
  s = s * invDet
  t = t * invDet
  sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 *
                                                     e) + f
  if sqrdistance < 0:
    sqrdistance = 0
  dist = np.sqrt(sqrdistance)
  return (dist, s, t)


@nb.jit(
    nb.types.Tuple((nb.float32, nb.float32, nb.float32))(
        nb.float32, nb.float32, nb.float32, nb.float32, nb.float32,
        nb.float32),
    nopython=True,
    cache=True,
    nogil=True)
def reg1(a, b, c, d, e, f):
  numer = c + e - b - d
  if numer <= 0:
    s = 0.0
    t = 1.0
    sqrdistance = c + 2.0 * e + f
  else:
    denom = a - 2.0 * b + c
    if numer >= denom:
      s = 1.0
      t = 0.0
      sqrdistance = a + 2.0 * d + f
    else:
      s = numer / denom
      t = 1 - s
      sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0
                                                         * e) + f

  if sqrdistance < 0:
    sqrdistance = 0

  dist = np.sqrt(sqrdistance)
  return (dist, s, t)


@nb.jit(
    nb.types.Tuple((nb.float32, nb.float32, nb.float32))(
        nb.float32, nb.float32, nb.float32, nb.float32, nb.float32,
        nb.float32),
    nopython=True,
    cache=True,
    nogil=True)
def reg2(a, b, c, d, e, f):
  tmp0 = b + d
  tmp1 = c + e
  # minimum on edge s+t=1
  if tmp1 > tmp0:
    numer = tmp1 - tmp0
    denom = a - 2.0 * b + c
    if numer >= denom:
      s = 1.0
      t = 0.0
      sqrdistance = a + 2.0 * d + f
    else:
      s = numer / denom
      t = 1 - s
      sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 *
                                                       e) + f
  # minimum on edge s=0
  else:
    s = 0.0
    if tmp1 <= 0.0:
      t = 1
      sqrdistance = c + 2.0 * e + f
    else:
      if e >= 0.0:
        t = 0.0
        sqrdistance = f
      else:
        t = -e / c
        sqrdistance = e * t + f

  if sqrdistance < 0:
    sqrdistance = 0

  dist = np.sqrt(sqrdistance)
  return (dist, s, t)


@nb.jit(
    nb.types.Tuple((nb.float32, nb.float32, nb.float32))(
        nb.float32, nb.float32, nb.float32, nb.float32, nb.float32,
        nb.float32),
    nopython=True,
    cache=True,
    nogil=True)
def reg3(a, b, c, d, e, f):
  s = 0.0
  if e >= 0:
    t = 0.0
    sqrdistance = f
  else:
    if -e >= c:
      t = 1.0
      sqrdistance = c + 2.0 * e + f
    else:
      t = -e / c
      sqrdistance = e * t + f

  if sqrdistance < 0:
    sqrdistance = 0

  dist = np.sqrt(sqrdistance)
  return (dist, s, t)


@nb.jit(
    nb.types.Tuple((nb.float32, nb.float32, nb.float32))(
        nb.float32, nb.float32, nb.float32, nb.float32, nb.float32,
        nb.float32),
    nopython=True,
    cache=True,
    nogil=True)
def reg4(a, b, c, d, e, f):
  if d < 0:
    t = 0.0
    if -d >= a:
      s = 1.0
      sqrdistance = a + 2.0 * d + f
    else:
      s = -d / a
      sqrdistance = d * s + f
  else:
    s = 0.0
    if e >= 0.0:
      t = 0.0
      sqrdistance = f
    else:
      if -e >= c:
        t = 1.0
        sqrdistance = c + 2.0 * e + f
      else:
        t = -e / c
        sqrdistance = e * t + f

  if sqrdistance < 0:
    sqrdistance = 0
  dist = np.sqrt(sqrdistance)
  return (dist, s, t)


@nb.jit(
    nb.types.Tuple((nb.float32, nb.float32, nb.float32))(
        nb.float32, nb.float32, nb.float32, nb.float32, nb.float32,
        nb.float32),
    nopython=True,
    cache=True,
    nogil=True)
def reg5(a, b, c, d, e, f):
  t = 0
  if d >= 0:
    s = 0
    sqrdistance = f
  else:
    if -d >= a:
      s = 1
      sqrdistance = a + 2.0 * d + f
      # GF 20101013 fixed typo d*s ->2*d
    else:
      s = -d / a
      sqrdistance = d * s + f
  if sqrdistance < 0:
    sqrdistance = 0
  dist = np.sqrt(sqrdistance)
  return (dist, s, t)


@nb.jit(
    nb.types.Tuple((nb.float32, nb.float32, nb.float32))(
        nb.float32, nb.float32, nb.float32, nb.float32, nb.float32,
        nb.float32),
    nopython=True,
    cache=True,
    nogil=True)
def reg6(a, b, c, d, e, f):
  tmp0 = b + e
  tmp1 = a + d
  if tmp1 > tmp0:
    numer = tmp1 - tmp0
    denom = a - 2.0 * b + c
    if numer >= denom:
      t = 1.0
      s = 0
      sqrdistance = c + 2.0 * e + f
    else:
      t = numer / denom
      s = 1 - t
      sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0
                                                         * e) + f

  else:
    t = 0.0
    if tmp1 <= 0.0:
      s = 1
      sqrdistance = a + 2.0 * d + f
    else:
      if d >= 0.0:
        s = 0.0
        sqrdistance = f
      else:
        s = -d / a
        sqrdistance = d * s + f
  dist = np.sqrt(sqrdistance)
  return (dist, s, t)


@nb.jit(
    nb.types.Tuple((nb.float32, nb.float32, nb.float32))(nb.float32[:, :],
                                                         nb.float32[:]),
    nopython=True,
    cache=True,
    nogil=True)
def point_tri_dist(facet, point):
  """
  Facet: float32, (3, 3)
  point: float32 (3, )
  """
  B = facet[0, :]
  e0 = facet[1, :] - B
  e1 = facet[2, :] - B
  D = B - point

  a = np.dot(e0, e0)
  b = np.dot(e0, e1)
  c = np.dot(e1, e1)
  d = np.dot(e0, D)
  e = np.dot(e1, D)
  f = np.dot(D, D)

  det = a * c - b * b
  s = b * e - c * d
  t = b * d - a * e

  if s + t <= det:
    if s < 0:
      if t < 0:
        # Region 4
        dist, s, t = reg4(a, b, c, d, e, f)
      else:
        # Region 3
        dist, s, t = reg3(a, b, c, d, e, f)
    elif t < 0:
      # Region 5
      dist, s, t = reg5(a, b, c, d, e, f)
    else:
      # Region 0
      dist, s, t = reg0(s, t, det, a, b, c, d, e, f)
  else:
    if s < 0:
      # Region 2
      dist, s, t = reg2(a, b, c, d, e, f)
    elif t < 0:
      # Region 6
      dist, s, t = reg6(a, b, c, d, e, f)
    else:
      # Region 1
      dist, s, t = reg1(a, b, c, d, e, f)

  return (dist, t, s)


@nb.jit(nb.float32[:](nb.float32[:,:,:], nb.float32[:]),
        nopython=True, nogil=True, cache=True)
def point_tris_dists(facets_arr, point):
  dists = np.empty(facets_arr.shape[0], dtype=np.float32)
  for f_idx in range(facets_arr.shape[0]):
    facet = facets_arr[f_idx, :]
    dist, _, _ = point_tri_dist(facet, point)
    dists[f_idx] = dist

  return(dists)


@nb.jit(nb.float32[:,:](nb.float32[:,:,:], nb.float32[:,:]),
        nopython=True, nogil=True, cache=True)
def points_tris_dists(facets_arr, points):
  dists = np.empty((points.shape[0], facets_arr.shape[0]), dtype=np.float32)
  for p_idx in range(points.shape[0]):
    point = points[p_idx, :]
    facets_dists = point_tris_dists(facets_arr, point)
    dists[p_idx] = facets_dists
  return(dists)
