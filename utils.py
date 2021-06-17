from itertools import tee

def pairwiseunique(iterable):
    # return all unique pairwise combinations in a list, e.g. l=[i, j] -> (i, j)

    i, j = tee(iterable)
    next(j, None)

    return zip(i, j)


def pairwise(iterable):
    # return all pairwise combinations in a list, e.g. l=[i, j] -> (i, j), (j, i)

    pairs = []
    for i in range(0,len(iterable)):
        for j in range(0,len(iterable)):
            if (i!=j):
                pairs.append((iterable[i], iterable[j]))

    return pairs
