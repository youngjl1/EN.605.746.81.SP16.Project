from __future__ import division
from datetime import datetime
from math import exp, log, sqrt
from sklearn.feature_extraction import FeatureHasher
from sklearn.utils import murmurhash3_32
from sklearn.neighbors import KNeighborsClassifier
from scipy import sparse
import cPickle as pickle
import math

num_samples = 100000
train = 'c:/criteo/smaller_datasets/train-' + str(num_samples)  # path to training file
D = 2 ** 20   # number of weights use for learning

# A. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-12), 10e-12)
    return -log(p) if y == 1. else -log(1. - p)

# B. Apply hash trick of the original csv row
# for simplicity, we treat both integer and categorical features as categorical
# INPUT:
#     csv_row: a csv dictionary, ex: {'Lable': '1', 'I1': '357', 'I2': '', ...}
#     D: the max index that we can hash to
# OUTPUT:
#     x: a list of indices that its value is 1
def get_x(row, D):
    x = []  # 0 is the index of the bias term
    for i, value in enumerate(row):
        #index = int(value + str(i), 16) % D  # weakest hash ever ;)
        index = murmurhash3_32(value + str(i), seed=0) % D
        x.append(index)
        continue

    return (x)  # x contains indices of features that have a value of 1

def get_distance(a,b):
    lena = len(a)
    lenb = len(b)
    lenab = len(a & b)
    return math.sqrt(lena + lenb - (lenab * 2))



k_folds = 5
fold_size = num_samples / k_folds
final_tloss = []
final_vloss = []

for idx in range(k_folds):
    v_start = idx*fold_size
    v_end = v_start + fold_size - 1

    data = []
    ys = []
    nearest = []
    k = 39
    vdata = []
    vys = []
    for t, raw in enumerate(open(train)):
        row = raw.split('\t')
        row[len(row)-1] = row[len(row)-1][:-1] # remove the trailing new line
        y = 1. if row[0] == '1' else 0.

        del row[0]  # can't let the model peek the answer
        x = set(get_x(row, D))


        if t >= v_start and t <= v_end:
            vdata.append(x)
            vys.append(y)
        else:
            data.append(x)
            ys.append(y)

        if(t % 10000 == 0 and t > 0):
            print "parsing:  " + str(t)


    if(False):
        v_count = 0
        loss = 0
        for t, x in enumerate(vdata):
            distances = []
            y = vys[t]
            for i, item in enumerate(data):
                if i < k:
                    distances.append([i, get_distance(x, item)])
                else:
                    if i == k:
                        distances = sorted(distances, key=lambda item: item[1])
                    distance = get_distance(x, item)
                    if distance < distances[k-1][1]:
                        del distances[k-1]
                        distances.append([i, distance])
                        distances = sorted(distances, key=lambda item: item[1])

            #distances = sorted(distances, key=lambda item: item[1])
            sum = 0
            for i in range(k):
                sum += ys[distances[i][0]]
            p = sum / k

            loss += logloss(p, y)
            v_count += 1

            if v_count % 10 == 0:
                print('%s\tK: %d\tV_set encountered: %d\tcurrent logloss: %f' % (
                    datetime.now(), k, v_count, loss/v_count))
            #if v_count == 2:
            #    break
        final_vloss.append(loss/v_count)
    if True:
        v_count = 0
        loss = 0
        for t, x in enumerate(data):
            distances = []
            y = ys[t]
            for i, item in enumerate(data):
                if i < k:
                    distances.append([i, get_distance(x, item)])
                else:
                    if i == k:
                        distances = sorted(distances, key=lambda item: item[1])
                    distance = get_distance(x, item)
                    if distance < distances[k-1][1]:
                        del distances[k-1]
                        distances.append([i, distance])
                        distances = sorted(distances, key=lambda item: item[1])

            #distances = sorted(distances, key=lambda item: item[1])
            sum = 0
            for i in range(k):
                sum += ys[distances[i][0]]
            p = sum / k

            loss += logloss(p, y)
            v_count += 1

            if v_count % 10 == 0:
                print('%s\tK: %d\tV_set encountered: %d\tcurrent logloss: %f' % (
                    datetime.now(), k, v_count, loss/v_count))
            #if v_count == 2:
            #    break
        final_tloss.append(loss/v_count)


print "final training loss"
print final_tloss
print "final validation loss"
print final_vloss





