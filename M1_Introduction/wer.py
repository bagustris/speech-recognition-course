import numpy as np


def string_edit_distance(ref=None, hyp=None):

    if ref is None or hyp is None:
        RuntimeError("ref and hyp are required, cannot be None")

    x = ref
    y = hyp
    tokens = len(x)
    if (len(hyp)==0):
        return (tokens, tokens, tokens, 0, 0)

    # p[ix,iy] consumed ix tokens from x, iy tokens from y
    p = np.PINF * np.ones((len(x) + 1, len(y) + 1)) # track total errors
    e = np.zeros((len(x)+1, len(y) + 1, 3)) # track deletions, insertions, substitutions
    p[0] = 0
    for ix in range(len(x) + 1):
        for iy in range(len(y) + 1):
            cst = np.PINF*np.ones([3])
            s = 0
            if ix > 0:
                cst[0] = p[ix - 1, iy] + 1 # deletion cost
            if iy > 0:
                cst[1] = p[ix, iy - 1] + 1 # insertion cost
            if ix > 0 and iy > 0:
                s = (1 if x[ix - 1] != y[iy -1] else 0)
                cst[2] = p[ix - 1, iy - 1] + s # substitution cost
            if ix > 0 or iy > 0:
                idx = np.argmin(cst) # if tied, one that occurs first wins
                p[ix, iy] = cst[idx]

                if (idx==0): # deletion
                    e[ix, iy, :] = e[ix - 1, iy, :]
                    e[ix, iy, 0] += 1
                elif (idx==1): # insertion
                    e[ix, iy, :] = e[ix, iy - 1, :]
                    e[ix, iy, 1] += 1
                elif (idx==2): # substitution
                    e[ix, iy, :] = e[ix - 1, iy - 1, :]
                    e[ix, iy, 2] += s

    edits = int(p[-1,-1])
    deletions, insertions, substitutions = e[-1, -1, :]
    return (tokens, edits, deletions, insertions, substitutions)


