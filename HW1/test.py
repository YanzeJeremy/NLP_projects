import numpy as np
import itertools
from collections import Counter

traind = [
    "just plain boring".split(),
    "entirely predictable and lacks energy".split(),
    "no surprises and very few laughs".split(),
    "very powerful".split(),
    "the most fun film of the summer".split()]

yd = [0, 0, 0, 1, 1]
testd = "entirely predictable and lacks energy".split()

def train_naive_bayes(X, y):
    Ndoc = len(X)
    logpc = {}
    bigdoc = {}
    logpwc = {}
    V = set(itertools.chain(*X))
    for i, c in enumerate(list(set(y))):
        cindex = [_i for _i,_c in enumerate(y) if _c == c]
        Nc = len(cindex)
        logpc[c] = np.log(Nc/Ndoc)
        bigdoc[c] = list(itertools.chain(*[X[_i] for _i in cindex]))
        for w in V:
            countc = Counter(bigdoc[c])
            count_wc = countc[w]
            logpwc[w,c] = np.log((count_wc+1)/(sum(countc.values())+len(V)))
    return logpc, logpwc, V


logpc, logpwc, V = train_naive_bayes(traind, yd)

def test_naive_bayes(testd, logpc, logpwc, y ,V):
    sum_c = {}
    for c in set(y):
        sum_c[c] = logpc[c]
        for i in range(len(testd)):
            w = testd[i]
            if w in V:
                sum_c[c] = sum_c[c] + logpwc[w,c]
    sortres = sorted(sum_c.items(), key=lambda x: x[1],reverse=True)
    print(sortres)
    print(sortres[0][0])
    return sortres[0][0]

test_naive_bayes(testd, logpc, logpwc, yd, V)