from itertools import combinations
import os
from datetime import datetime


def pairwiseunique(iterable):
    """return all unique pairwise combinations in list, l=[i, j] -> (i, j)"""

    pairs = combinations(iterable, 2)

    return list(pairs)


def pairwise(iterable):
    """return all pairwise combinations in list, l=[i, j] -> (i, j), (j, i)"""

    pairs = []
    for i in range(0, len(iterable)):
        for j in range(0, len(iterable)):
            if (i != j):
                pairs.append((iterable[i], iterable[j]))

    return pairs


def appendline(filename, text):
    """append text as new line at end of file"""

    with open(filename, "a+") as infile:
        infile.seek(0)
        readtext = infile.read(10)
        if len(readtext) > 0:
            infile.write("\n")
        infile.write(text)


def getdate():
    """get the current date """

    dateTimeObj = datetime.now()
    dateObj = dateTimeObj.date()

    out_time = dateObj.strftime("%b-%d-%Y")

    return out_time


def getenddate(enddate):
    """get the current date """

    out_time = enddate.strftime("%b-%d-%Y")

    return out_time


def createoutputdir(lineage, output, enddate):
    """create output directory structure"""

    out_time = getenddate(enddate)

    out_list = ['prediction', 'validation', 'logs/prediction',
                'logs/validation', 'additional-plots/prediction/VAR',
                'additional-plots/validation/VAR']

    for elem in out_list:
        out_dir = os.path.join(out_time, lineage, elem)
        results_dir = os.path.join(output, out_dir)
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

    cross_dir = os.path.join(output, out_time, 'cross-correlation')
    if not os.path.isdir(cross_dir):
        os.makedirs(cross_dir)
