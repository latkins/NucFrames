import numpy as np
from numpy.core.umath_tests import inner1d


def point_tri_dists(facets, points):
  # triangle vertices
  p1 = facets[:, 0, :].astype(np.float32)
  p2 = facets[:, 1, :].astype(np.float32)
  p3 = facets[:, 2, :].astype(np.float32)

  p1_norm = np.linalg.norm(p1, axis=1)
  p2_norm = np.linalg.norm(p2, axis=1)
  p3_norm = np.linalg.norm(p3, axis=1)

  p21 = p1 - p2
  p12 = p2 - p1
  p32 = p2 - p3
  p23 = p3 - p2
  p31 = p1 - p3
  p13 = p3 - p1

  p21_norm = np.linalg.norm(p21, axis=1)
  p12_norm = np.linalg.norm(p12, axis=1)
  p32_norm = np.linalg.norm(p32, axis=1)
  p23_norm = np.linalg.norm(p23, axis=1)
  p31_norm = np.linalg.norm(p31, axis=1)
  p13_norm = np.linalg.norm(p13, axis=1)

  v1 = (p21 / p21_norm[:, None]) + (p31 / p31_norm[:, None])
  v2 = (p32 / p32_norm[:, None]) + (p12 / p12_norm[:, None])
  v3 = (p13 / p13_norm[:, None]) + (p23 / p23_norm[:, None])

  np_ = np.cross(p12, p13)

  np_norm = np.linalg.norm(np_, axis=1)
  """
  print(points.shape, p1.shape)
  p10_ = np.einsum("ab,cb -> acb", points, -1 *  p1)
  p01_ = np.einsum("ab,cb -> acb", -1 * points, p1)
  print(p10_.shape, p01_.shape)
  p10_norm_ = np.linalg.norm(p10_, axis=2)
  p01_norm_ = np.linalg.norm(p01_, axis=2)
  print(p10_norm_.shape)
  """

  # Will use the convention that i = ', so pi should be read as p'
  all_dists = []
  for i in range(points.shape[0]):
    p0 = points[i]
    p10 = p0 - p1
    p01 = p1 - p0
    p02 = p2 - p0
    p03 = p3 - p0
    p10_norm = np.linalg.norm(p10, axis=1)
    p01_norm = np.linalg.norm(p01, axis=1)
    p02_norm = np.linalg.norm(p02, axis=1)
    p03_norm = np.linalg.norm(p03, axis=1)

    # Projection onto plane.
    p0i, p00i, p00i_norm = point_project(p0, p10, p01_norm, np_, np_norm)

    p0i1 = p1 - p0i
    p0i2 = p2 - p0i
    p0i3 = p3 - p0i
    p10i = p0i - p1
    p20i = p0i - p2
    p30i = p0i - p3
    p0i1_norm = np.linalg.norm(p0i1, axis=1)
    p0i2_norm = np.linalg.norm(p0i2, axis=1)
    p0i3_norm = np.linalg.norm(p0i3, axis=1)

    # Is it anticlockwise of v{1,2,3} ?
    f1 = inner1d(np.cross(v1, p0i1), np_) > 0
    f2 = inner1d(np.cross(v2, p0i2), np_) > 0
    f3 = inner1d(np.cross(v3, p0i3), np_) > 0

    # Determine the side the point is closest to.
    m12 = f1 & ~f2
    m23 = f2 & ~f3
    m31 = f3 & ~f1

    distances = p00i_norm.copy()
    distances[m12] = calc_side_dist(p12[m12], p0i[m12], p1_norm[m12],
                                    p2_norm[m12], p01_norm[m12],
                                    p02_norm[m12], p0i1[m12], p0i1_norm[m12],
                                    p0i2[m12], np_[m12], p00i_norm[m12])
    distances[m23] = calc_side_dist(p23[m23], p0i[m23], p2_norm[m23],
                                    p3_norm[m23], p02_norm[m23],
                                    p03_norm[m23], p0i2[m23], p0i2_norm[m23],
                                    p0i3[m23], np_[m23], p00i_norm[m23])
    distances[m31] = calc_side_dist(p31[m31], p0i[m31], p3_norm[m31],
                                    p1_norm[m31], p03_norm[m31],
                                    p01_norm[m31], p0i3[m31], p0i3_norm[m31],
                                    p0i1[m31], np_[m31], p00i_norm[m31])
    return (distances)

    # # Which are outside the triangle, for their respective side?
    # n12 = (inner1d(np.cross(p0i1[m12], p0i2[m12]), np_[m12]) < 0)
    # n23 = (inner1d(np.cross(p0i2[m23], p0i3[m23]), np_[m23]) < 0)
    # n31 = (inner1d(np.cross(p0i3[m31], p0i1[m31]), np_[m31]) < 0)

    # # Final distance array. Default to closest distance to plane, change those
    # # that don't fall within the triangle.
    # dists = p00i_norm.copy()

    # # for m12 & n12 - those outside the triangle, nearest the side p12
    # r12 = np.cross(np.cross(p0i2, p0i1), p12)
    # r23 = np.cross(np.cross(p0i3, p0i2), p23)
    # r31 = np.cross(np.cross(p0i1, p0i3), p31)

    # r12_norm = np.linalg.norm(r12, axis=1)
    # r23_norm = np.linalg.norm(r23, axis=1)
    # r31_norm = np.linalg.norm(r31, axis=1)

    # cos_gamma_12 = inner1d(p0i1, r12) / (p0i1_norm * r12_norm)
    # cos_gamma_23 = inner1d(p0i2, r23) / (p0i2_norm * r23_norm)
    # cos_gamma_31 = inner1d(p0i3, r31) / (p0i3_norm * r31_norm
    #                                      )    # Is this sign correct?

    # # edge p12
    # p0i_p0ii_norm = p0i1_norm * cos_gamma_12
    # p0i_p0ii = p0i_p0ii_norm[:, None] * (r12 / r12_norm[:, None])
    # p0ii = p0i + p0i_p0ii
    # p0ii_norm = np.linalg.norm(p0ii, axis=1)

    # t12 = (p0ii_norm[m12][n12] - p1_norm[m12][n12]) / (
    #     p2_norm[m12][n12] - p1_norm[m12][n12])
    # t23 = (p0ii_norm[m23][n23] - p2_norm[m23][n23]) / (
    #     p3_norm[m23][n23] - p2_norm[m23][n23])
    # t31 = (p0ii_norm[m31][n31] - p3_norm[m31][n31]) / (
    #     p1_norm[m31][n31] - p3_norm[m31][n31])
    # # t < 0, p0 closest to p1. d = p01_norm
    # # t > 1, p0 closest to p2. d = p02_norm
    # # 0 <= t <= 1, p0 closest to edge. d = sqrt(p0i_p0ii_norm**2 + p00i_norm**2)
    # dist_12_out = np.empty_like(t12)
    # dist_12_out[t12 < 0] = p01_norm[m12][n12][t12 < 0]
    # dist_12_out[t12 > 1] = p02_norm[m12][n12][t12 > 1]
    # dist_12_out[(0 <= t12) & (t12 <= 1)] = np.sqrt(np.square(p0i_p0ii_norm[
    #     m12][n12][(0 <= t12) & (t12 <= 1)]) + np.square(p00i_norm[m12][n12][(
    #         0 <= t12) & (t12 <= 1)]))
    # dists[m12][n12] = dist_12_out

    # # edge p23
    # p0i_p0ii_norm = p0i1_norm * cos_gamma_23
    # p0i_p0ii = p0i_p0ii_norm[:, None] * (r23 / r23_norm[:, None])
    # p0ii = p0i + p0i_p0ii
    # p0ii_norm = np.linalg.norm(p0ii, axis=1)

    # dist_23_out = np.empty_like(t23)
    # dist_23_out[t23 < 0] = p01_norm[m23][n23][t23 < 0]
    # dist_23_out[t23 > 1] = p02_norm[m23][n23][t23 > 1]
    # dist_23_out[(0 <= t23) & (t23 <= 1)] = np.sqrt(np.square(p0i_p0ii_norm[
    #     m23][n23][(0 <= t23) & (t23 <= 1)]) + np.square(p00i_norm[m23][n23][(
    #         0 <= t23) & (t23 <= 1)]))
    # dists[m23][n23] = dist_23_out

    # # edge p31
    # p0i_p0ii_norm = p0i1_norm * cos_gamma_31
    # p0i_p0ii = p0i_p0ii_norm[:, None] * (r31 / r31_norm[:, None])
    # p0ii = p0i + p0i_p0ii
    # p0ii_norm = np.linalg.norm(p0ii, axis=1)

    # dist_31_out = np.empty_like(t31)
    # dist_31_out[t31 < 0] = p01_norm[m31][n31][t31 < 0]
    # dist_31_out[t31 > 1] = p02_norm[m31][n31][t31 > 1]
    # dist_31_out[(0 <= t31) & (t31 <= 1)] = np.sqrt(np.square(p0i_p0ii_norm[
    #     m31][n31][(0 <= t31) & (t31 <= 1)]) + np.square(p00i_norm[m31][n31][(
    #         0 <= t31) & (t31 <= 1)]))
    # dists[m31][n31] = dist_31_out

    # all_dists.append(dists)
    # return (np.array(all_dists))
  """
  for i in range(points[:100].shape[0]):
  p0 = points[i]
  p10 = p10_[i]

  #dists_ = np.einsum('ij,ij->i', v0_point_diff, v0_point_diff)

  dists = inner1d(np_, p10)
  dists /= np_norm

  print(np.max(np.abs(dists)))

  min_dist_idx = np.argmin(np.abs(dists))

  min_dist = dists[min_dist_idx]

  all_dists.append(min_dist)

  all_dists = np.array(all_dists)
    """


def point_project(p0, p10, p01_norm, np_, np_norm):
  """
  This function projects each point onto the plane of all triangles.

  p0: The point, from which we are measuring the distance to each triangle. (3,)
  p10: The vector from the first vertex of each triangle to the plane. (n_triangles, 3)
  p01_norm: Norm of vector from the point to the first vertex of each triangle. (n_triangles, )
  p10_norm: Norm of vector from the first vertex of each triangle to the point. (n_triangles, )
  np_: The normal vectors for each triangle (n_triagles, 3)
  np_norm: The norm of the normal vectors for each triangle (n_triangles, )

  Returns:
  p0i: The intersection of p0 with the plane (n_triangles, 3)
  p00i: The vector from p0 to the intersection, p0i. (n_triangles, 3)
  p00i_norm: Norm (aka length) of the point to the intersection point. (n_triangles, )

  """
  cos_alpha = inner1d(p10, np_) / (p01_norm * np_norm)

  p00i_norm = p01_norm * cos_alpha
  p00i = (-1 * p00i_norm)[:, None] * (np_ / np_norm[:, None])
  p0i = p0 + p00i

  return (p0i, p00i, p00i_norm)


def calc_side_dist(pab, p0i, pa_norm, pb_norm, p0a_norm, p0b_norm, p0ia,
                   p0ia_norm, p0ib, np_, p00i_norm):
  """
  NOTE: n_points here is just the points that are closest to this edge.
  pab: vector between a & b, the two vertices forming the edge p0 is closest to (n_points, 3)
  p0i: projection of p0 onto the plane formed by the triangle. (n_points, 3)
  pa_norm: magnitude of vertex a. (n_points,)
  pb_norm: magnitude of vertex b. (n_points,)
  p0a_norm: magnitude of vector between vertex a and p0. (n_points,)
  p0b_norm: magnitude of vector between vertex b and p0. (n_points,)
  p0ia: vector between p0's projection onto the triangle plane, and vertex a. (n_points, 3)
  p0ia_norm: norm of p0ia. (n_points, )
  p0ib: vector between p0's projection onto the triangle plane, and vertex b. (n_points, 3)
  np_: triangle normal vectors (n_points, 3)
  p00i_norm: normal of vector between p0 and plane intersection, p0i. (n_points,)

  Points are nearest the specified side, but unknown if they are internal.
  """
  # No points passed, return empty list.
  if len(p00i_norm) == 0:
    return(p00i_norm)
  # Areas where point is outside of triangle.
  tri_m = inner1d(np.cross(p0ia, p0ib), np_) < 0

  # r: direction of p0i to p0ii
  r = np.cross(np.cross(p0ib, p0ia), pab)
  r_norm = np.linalg.norm(r, axis=1)

  cos_gamma = inner1d(p0ia, r) / (p0ia_norm * r_norm)

  # p0ii: projection of p0i (plane projection) with the given edge.
  p0i0ii_norm = p0ia_norm * cos_gamma
  p0i0ii = p0i0ii_norm[:, None] * (r / r_norm[:, None])
  p0ii = p0i + p0i0ii
  p0ii_norm = np.linalg.norm(p0ii, axis=1)

  # t < 0, p0 closest to pa. d = p0a_norm
  # t > 1, p0 closest to pb. d = p0b_norm
  # 0 <= t <= 1, p0 closest to edge. d = sqrt(p0i0ii_norm**2 + p00i_norm**2)
  t = (p0ii_norm - pa_norm) / (pb_norm - pa_norm)
  t_01 = (0 <= t) & (t <= 1)

  dists = p00i_norm.copy()
  dists[tri_m & (t < 0)] = p0a_norm[tri_m & (t < 0)]
  dists[tri_m & (t > 1)] = p0b_norm[tri_m & (t > 1)]
  dists[tri_m & t_01] = np.sqrt(np.square(p0i0ii_norm[tri_m & t_01]) +
                               np.square(p00i_norm[tri_m & t_01]))

  return (dists)
