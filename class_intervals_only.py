#
# The class interval code is from class_interval.py by Carson Farmer 
# Source: http://carsonfarmer.com/2010/09/adding-a-bit-of-classification-to-qgis/
# The QGIS specific code has been removed
#

import math, random

def equal(values, classes=5):
  """
  Equal interval algorithm in Python
  
  Returns breaks based on dividing the range of 'values' into 'classes' parts.
  """

  _min = min(values)
  _max = max(values)
  unit = (_max - _min) / classes
  res = [_min + k*unit for k in range(classes+1)]
  return res
  
#def quantile(values, classes=5):
def quantile(vals, classes=5):
  """
  Quantum GIS quantile algorithm in Python
  
  Returns values taken at regular intervals from the cumulative 
  distribution function (CDF) of 'values'.
  """

  values = sorted(vals)
  n = len(values)
  breaks = []
  for i in range(classes):
    q = i / float(classes)
    a = q * n
    aa = int(q * n)
    r = a - aa
    Xq = (1 - r) * values[aa] + r * values[aa+1]
    breaks.append(Xq)
  breaks.append(values[n-1])
  return breaks

def rpretty(dmin, dmax, n=5):
  """
  R's pretty algorithm implemented in Python
  Code based on R implementation from 'labeling' R package 
  
  Compute a sequence of about 'n+1' equally spaced 'round' values
  which cover the range of the values in 'values'.  The values are chosen
  so that they are 1, 2 or 5 times a power of 10.

  Parameters:
    dmin        : minimum of the data range
    dmax        : maximum of the data range
    n           : number of class intervals
  """

  min_n = int(n / 3) # Nonnegative integer giving the minimal number of intervals
  shrink_sml = 0.75  # Positive numeric by a which a default scale is shrunk 
                     # in the case when range(x) is very small (usually 0).
  high_u_bias = 1.5  # Non-negative numeric, typically > 1. The interval unit 
                     # is determined as {1,2,5,10} times b, a power of 10.
                     # Larger high.u.bias values favor larger units
  u5_bias = 0.5 + 1.5 * high_u_bias
                     # Non-negative numeric multiplier favoring 
                     # factor 5 over 2.
  h = high_u_bias
  h5 = u5_bias
  ndiv = n

  dx = dmax - dmin
  if dx == 0 and dmax == 0:
    cell = 1.0
    i_small = True
    U = 1
  else:
    cell = max(abs(dmin), abs(dmax))
    if h5 >= 1.5 * h + 0.5:
      U = 1 + (1.0/(1 + h))
    else:
      U = 1 + (1.5 / (1 + h5))
    i_small = dx < (cell * U * max(1.0, ndiv) * 1e-07 * 3.0)

  if i_small:
    if cell > 10:
      cell = 9 + cell / 10
      cell = cell * shrink_sml
    if min_n > 1:
      cell = cell / min_n
  else:
    cell = dx
    if ndiv > 1:
      cell = cell / ndiv
  if cell < 20 * 1e-07:
    cell = 20 * 1e-07
  
  base = 10.0**math.floor(math.log10(cell))
  unit = base
  if (2 * base) - cell < h * (cell - unit):
    unit = 2.0 * base
    if (5 * base) - cell < h5 * (cell - unit):
      unit = 5.0 * base
      if (10 * base) - cell < h * (cell - unit):
        unit = 10.0 * base
  # Maybe used to correct for the epsilon here??
  ns = math.floor(dmin / unit + 1e-07)
  nu = math.ceil(dmax / unit - 1e-07)

  # Extend the range out beyond the data. Does this ever happen??
  while ns * unit > dmin + (1e-07 * unit):
    ns = ns-1
  while nu * unit < dmax - (1e-07 * unit):
    nu = nu+1
  # If we don't have quite enough labels, extend the range out to make more (these labels are beyond the data :( )
  k = math.floor(0.5 + nu-ns)
  if k < min_n:
    k = min_n - k
    if ns >= 0:
      nu = nu + k / 2
      ns = ns - k / 2 + k % 2
    else:
      ns = ns - k / 2
      nu = nu + k / 2 + k % 2
    ndiv = min_n
  else:
    ndiv = k
  graphmin = ns * unit
  graphmax = nu * unit
  count = int(math.ceil(graphmax - graphmin)/unit)
  res = [graphmin + k*unit for k in range(count+1)]
  if res[0] < dmin:
    res[0] = dmin
  if res[-1] > dmax:
    res[-1] = dmax
  return res
  
def pretty(values, classes=5):
  """
  Helper function for rpretty, which implemets R's pretty algorithm

  Returns a number of breaks not necessarily equal to 'classes' using 
  rpretty, but likely to be legible.

  Parameters:
    values : list of input values
    classes     : number of class intervals
  """
  _min = min(values)
  _max = max(values)
  return rpretty(_min, _max, classes)
  
def std_dev(values, classes=5):
  """
  Python implementation of the standard deviation class interval algorithm
  as implemented in the 'classInt' package available for 'R'.
  
  Returns breaks based on 'pretty' of the centred and scaled values of 'values',
  and may have a number of classes different from 'classes'.
  """

  mean = 0.0
  sd2 = 0.0
  N = len(values)
  _min = min(values)
  _max = max(values)
  for i in values:
    mean += i
  mean = mean / N
  for i in values:
    sd = i - mean
    sd2 += sd * sd
  sd2 = math.sqrt(sd2 / N)
  res = rpretty((_min-mean)/sd2, (_max-mean)/sd2, classes)
  res2 = [(val*sd2)+mean for val in res]
  return res2
  
def jenks_sample(values, maxsize=1000, classes=5):
  if len(values) > maxsize:
    size = min(maxsize, max(maxsize, len(values)*0.10))
    sample = [min(values), max(values)]
    while len(sample) < size-2:
      i = random.randint(0, len(values)-1)
      sample.append(values.pop(i))
  else:
    sample = values
  return jenks(sample, classes)

#def jenks(values, classes=5):
def jenks(vals, classes=5):
  """
  Jenks Optimal (Natural Breaks) algorithm implemented in Python.
  The original Python code comes from here:
  http://danieljlewis.org/2010/06/07/jenks-natural-breaks-algorithm-in-python/
  and is based on a JAVA and Fortran code available here:
  https://stat.ethz.ch/pipermail/r-sig-geo/2006-March/000811.html
  
  Returns class breaks such that classes are internally homogeneous while 
  assuring heterogeneity among classes.
  
  """

  #values.sort()
  values = sorted(vals)
  mat1 = []
  for i in range(0,len(values)+1):
    temp = []
    for j in range(0,classes+1):
        temp.append(0)
    mat1.append(temp)
  mat2 = []
  for i in range(0,len(values)+1):
    temp = []
    for j in range(0,classes+1):
        temp.append(0)
    mat2.append(temp)
  for i in range(1,classes+1):
    mat1[1][i] = 1
    mat2[1][i] = 0
    for j in range(2,len(values)+1):
        mat2[j][i] = float('inf')
  v = 0.0
  for l in range(2,len(values)+1):
    s1 = 0.0
    s2 = 0.0
    w = 0.0
    for m in range(1,l+1):
      i3 = l - m + 1
      val = float(values[i3-1])
      s2 += val * val
      s1 += val
      w += 1
      v = s2 - (s1 * s1) / w
      i4 = i3 - 1
      if i4 != 0:
        for j in range(2,classes+1):
          if mat2[l][j] >= (v + mat2[i4][j - 1]):
            mat1[l][j] = i3
            mat2[l][j] = v + mat2[i4][j - 1]
    mat1[l][1] = 1
    mat2[l][1] = v
  k = len(values)
  kclass = []
  for i in range(0,classes+1):
    kclass.append(0)
  kclass[classes] = float(values[len(values) - 1])
  kclass[0] = float(values[0])
  countNum = classes
  while countNum >= 2:
    id = int((mat1[k][countNum]) - 2)
    kclass[countNum - 1] = values[id]
    k = int((mat1[k][countNum] - 1))
    countNum -= 1
  return kclass

def main():
  print ("Quantile: ", quantile(values, classes=5))
  print ("Equal Interval: ", equal(values, classes=5))
  print ("R's Pretty: ", pretty(values, classes=5))
  print ("Standard Deviation: ", std_dev(values, classes=5))
  print ("Natural Breaks (Jenks): ", jenks(values, classes=5))
  print ("Sampled Jenks: ", jenks_sample(values, classes=5))

if __name__ == '__main__':
    main()
