from itertools import tee
import os
from datetime import datetime

def pairwiseunique(iterable):
    """return all unique pairwise combinations in a list, e.g. l=[i, j] -> (i, j)"""

    i, j = tee(iterable)
    next(j, None)

    return zip(i, j)


def pairwise(iterable):
    """return all pairwise combinations in a list, e.g. l=[i, j] -> (i, j), (j, i)"""

    pairs = []
    for i in range(0,len(iterable)):
        for j in range(0,len(iterable)):
            if (i!=j):
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


def createoutputdir(lineage):
    """create output directory structure"""

    script_dir = os.path.dirname(__file__)

    out_time = getdate()

    out_list = ['prediction', 'validation', 'logs', 'additional-plots']

    for elem in out_list:
        out_dir = os.path.join(out_time, lineage ,elem)
        results_dir = os.path.join(script_dir, out_dir)
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

