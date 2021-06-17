def pairwise(iterable):
    # return all pairwise combinations in a list

    pairs = []
    for i in range(0,len(iterable)):
        for j in range(0,len(iterable)):
            if (i!=j):
                pairs.append((iterable[i], iterable[j]))

    return pairs
