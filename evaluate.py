import numpy as np
import math
import scipy
from scipy.spatial.distance import euclidean,cosine,cdist

def fx_calc_map_label(image, text, label, k = 0, dist_method='COS'):
  print("...start...")
  if dist_method == 'L2':
    dist = cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0] #5621
  if k == 0:
    k = numcases
  res = []
  result = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    for j in range(k):
      if label[i] == label[order[j]]:
        r += 1
        p += (r / (j + 1))
    if r > 0:
      result +=[label[i]]
      res += [p / r]
    else:
      res += [0]
  print("...end...")

  return np.mean(res)

def mAP(image, text, label, k = 10, dist_method='COS'):
    if dist_method == 'L2':
      dist = cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
      dist = cdist(image, text, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
      k = numcases
    res = []
    result = []
    rr = []
    for i in range(numcases):
        order = ord[i]
        hits, sum_precs = 0, 0.0
        for j in range(k):
          if label[i] == label[order[j]]:
              hits += 1
              sum_precs += hits / (j + 1.0)
              rr += [1/(j + 1.0)]
        if hits > 0:
            # return sum_precs / len(ground_truth)
            result += [sum_precs/hits]
        else:
            result += [0]
    # pre = hits / (1.0 * numcases if len(label) != 0 else 1)
    # rec = hits / (1.0 * len(label) if len(label) != 0 else 1)
    # print(pre,rec)
    return np.mean(result),np.mean(rr)

def mRR(image, text, label, k = 0, dist_method='COS'): 
    if dist_method == 'L2':
        dist = cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
      dist = cdist(image, text, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
      k = numcases
    res = []
    result = []
