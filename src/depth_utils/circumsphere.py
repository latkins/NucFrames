import numpy as np

def simplex_circumsphere(coords):
  vertices, d = coords.shape

  if d == 3:
    if vertices == 4:
      return(_circumsphere(coords))
    elif vertices == 3:
      return(_circumcircle(coords))
    elif vertices == 2:
      diff_vec = coords[0] - coords[1]
      circumradius = np.sqrt(diff_vec.dot(diff_vec)) / 2
      centre = np.mean(coords, axis=0)
      return(centre, circumradius)
  else:
    raise ValueError("Currenlty only implemented for R^3")

def _circumcircle(coords):
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


def _circumsphere(coords):
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

  denom = 2 * a
  centre = np.array([Dx / denom, Dy / denom, Dz / denom])

  return (centre, r)
