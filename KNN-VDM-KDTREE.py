from __future__ import division
from datetime import datetime
from math import exp, log, sqrt
from sklearn.feature_extraction import FeatureHasher
from sklearn.utils import murmurhash3_32
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KDTree
from scipy import sparse
import cPickle as pickle
import math

num_samples = 1000000
use_both = False
train = 'c:/criteo/smaller_datasets/train-' + str(num_samples)  # path to training file
D = 2 ** 20   # number of weights use for learning
p_d = 10e-10

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
def get_x(row, value_count_yes, value_count_no, y, average_num_values, increment_value_counts = True):
    x = []
    for i in range(13):
        if row[i] != "":
            val = math.log(float(row[i]),2) if float(row[i]) > 0 else float(row[i])
        else:
            val = math.log(average_num_values[i],2) if average_num_values[i] > 0 else average_num_values[i]
        #val = val/10
        x.append(val)

    if increment_value_counts:
        for i, value in enumerate(row):
            if (i >= 13):
                value_count = value_count_no if y == 0 else value_count_yes
                if value in value_count[i-13]:
                    value_count[i-13][value] += 1
                else:
                    value_count[i-13][value] = 1

    return [x,row[13:]]


def get_results(data, vdata, ys, vys, k):
    tree = KDTree(data)
    v_count = 0
    loss = 0


    for t, x in enumerate(vdata):
        y = vys[t]

        dist, ind = tree.query([x], k=k)
        sum = 0
        for i in ind[0]:
            sum += ys[i]
        p = sum / k

        loss += logloss(p, y)
        v_count += 1

        if v_count % 1000 == 0:
            print('%s\tK: %d\tV_set encountered: %d\tcurrent logloss: %f' % (
                datetime.now(), k, v_count, loss/v_count))


    return loss/v_count

k_folds = 5
fold_size = num_samples / k_folds
final_tloss = []
final_vloss = []
average_num_values = [3, 105, 26, 7, 18538, 116, 16, 12, 106, 0, 2, 0, 8]

for idx in range(k_folds):
    v_start = idx*fold_size
    v_end = v_start + fold_size - 1

    data = []
    ys = []
    nearest = []
    k = 80
    vdata = []
    vys = []

    value_count_yes = [{} for i in range(30)]
    value_count_no = [{} for i in range(30)]

    for t, raw in enumerate(open(train)):
        row = raw.split('\t')
        row[len(row)-1] = row[len(row)-1][:-1] # remove the trailing new line
        y = 1. if row[0] == '1' else 0.

        del row[0]  # can't let the model peek the answer


        if t >= v_start and t <= v_end:
            x = get_x(row, value_count_yes, value_count_no, y, average_num_values, increment_value_counts=False)
            vdata.append(x)
            vys.append(y)
        else:
            x = get_x(row, value_count_yes, value_count_no, y, average_num_values)
            data.append(x)
            ys.append(y)

        if(t % 10000 == 0 and t > 0):
            print "parsing:  " + str(t)

    m = 1
    for j, cat_vals in enumerate(vdata):
        cat_vals = cat_vals[1]
        new_cols = []
        for i, cat_val in enumerate(cat_vals):
            if cat_val not in value_count_yes[i]:
                value_count_yes[i][cat_val] = 0
            if cat_val not in value_count_no[i]:
                value_count_no[i][cat_val] = 0

            yes_dist = ((value_count_yes[i][cat_val]+m*p_d)/(value_count_yes[i][cat_val] + value_count_no[i][cat_val] + m))
            no_dist = ((value_count_no[i][cat_val]+m*p_d)/(value_count_yes[i][cat_val] + value_count_no[i][cat_val] + m))
            new_cols.append(yes_dist)
            new_cols.append(no_dist)
        # replace below with just new_cols
        if use_both:
            vdata[j] = vdata[j][0] + new_cols
            print "col length:  " + str(len(new_cols))
        else:
            #vdata[j] = new_cols
            vdata[j] = vdata[j][0]

    for j, cat_vals in enumerate(data):
        cat_vals = cat_vals[1]
        new_cols = []
        for i, cat_val in enumerate(cat_vals):
            if cat_val not in value_count_yes[i]:
                value_count_yes[i][cat_val] = 0
            if cat_val not in value_count_no[i]:
                value_count_no[i][cat_val] = 0

            yes_dist = ((value_count_yes[i][cat_val]+m*p_d)/(value_count_yes[i][cat_val] + value_count_no[i][cat_val] + m))
            no_dist = ((value_count_no[i][cat_val]+m*p_d)/(value_count_yes[i][cat_val] + value_count_no[i][cat_val] + m))
            new_cols.append(yes_dist)
            new_cols.append(no_dist)
        # replace below with just new_cols
        if use_both:
            data[j] = data[j][0] + new_cols
            print "col length:  " + str(len(new_cols))
        else:
            #data[j] = new_cols
            data[j] =  data[j][0]

    if(True):
        final_vloss.append(get_results(data, vdata, ys, vys, k))

    if True:
        final_tloss.append(get_results(data, data, ys, ys, k))


print "P = " + str(p_d)
print "Size: " + str(num_samples)
print "Use both: " + str(use_both)
print "final training loss"
print final_tloss
print "final validation loss"
print final_vloss





