from __future__ import division
from datetime import datetime
from math import exp, log, sqrt
from sklearn.feature_extraction import FeatureHasher
from sklearn.utils import murmurhash3_32
import numpy as np

num_samples = 10000000
train = 'c:/criteo/smaller_datasets/train-' + str(num_samples)  # path to training file

p_d= 10e-2
D = 2 ** 25   # number of weights use for learning

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

    return x  # x contains indices of features that have a value of 1

def count_feature_vals(x, counter):
    for i in x:
        counter[i] += 1
    return counter

m = 1
def get_cond_probs(x, counter, total, print_probs = False):
    probs = 1
    for i in x:
        num_occurrences = counter[i]
        #if(num_occurrences == 0):
        #    print("Oh shit. Might have to dirichlet")
        #    exit(1)
        # dirichlet
        prob = (num_occurrences+m*p_d)/(total+m)
        probs *= prob
        if(print_probs):
            print "prob: %f\toccurs: %d\ttotal: %d" % (prob, num_occurrences, total)

    return probs


final_vloss = []
final_tloss = []

k_folds = 5
fold_size = num_samples / k_folds

for idx in range(k_folds):
    v_start = idx*fold_size
    v_end = v_start + fold_size - 1

    n_yes = [0.] * D  # number of times we've encountered a feature associated with yes
    n_no = [0.] * D  # number of times we've encountered a feature associated with no
    total_yes = 0  # number of examples that are yes
    total_no = 0  # number of examples that are no

    for t, raw in enumerate(open(train)):
        if t >= v_start and t <= v_end:
            continue

        row = raw.split('\t')
        row[len(row)-1] = row[len(row)-1][:-1] # remove the trailing new line
        y = 1. if row[0] == '1' else 0.

        del row[0]  # can't let the model peek the answer
        x = get_x(row, D)

        if(y == 1):
            total_yes += 1
            n_yes = count_feature_vals(x, n_yes)
        else:
            total_no += 1
            n_no = count_feature_vals(x, n_no)
        #if(t % 10000 == 0 and t > 0):
            #print('%s\tencountered: %d\t' % (datetime.now(), t))

    prob1_yes = total_yes/(total_no+total_yes)
    prob1_no = total_no/(total_no+total_yes)

    vloss = 0
    tloss = 0
    v_count = 0
    t_count = 0
    for t, raw in enumerate(open(train)):
        if t >= v_start and t <= v_end:
            row = raw.split('\t')
            row[len(row)-1] = row[len(row)-1][:-1] # remove the trailing new line
            y = 1. if row[0] == '1' else 0.

            del row[0]  # can't let the model peek the answer
            x = get_x(row, D)

            yes_chance = prob1_yes * get_cond_probs(x, n_yes, total_yes, print_probs=False)
            no_chance = prob1_no * get_cond_probs(x, n_no, total_no)

            p = (yes_chance/(yes_chance+no_chance))

            #if(t % 1000 == 0 and t > 0):
            #    print (p)
            #    print [yes_chance, no_chance]

            vloss += logloss(p, y)
            #if(v_count % 10000 == 0 and v_count > 0):
                #print('%s\tencountered: %d\tCurrent Logloss: %f' % (datetime.now(), v_count, vloss/v_count))
            v_count += 1
        else:
            row = raw.split('\t')
            row[len(row)-1] = row[len(row)-1][:-1] # remove the trailing new line
            y = 1. if row[0] == '1' else 0.

            del row[0]  # can't let the model peek the answer
            x = get_x(row, D)

            yes_chance = prob1_yes * get_cond_probs(x, n_yes, total_yes, print_probs=False)
            no_chance = prob1_no * get_cond_probs(x, n_no, total_no)

            p = (yes_chance/(yes_chance+no_chance))

            #if(t % 1000 == 0 and t > 0):
            #    print (p)
            #    print [yes_chance, no_chance]

            tloss += logloss(p, y)
            #if(t_count % 10000 == 0 and t_count > 0):
            #    print('%s\tencountered: %d\tCurrent Logloss: %f' % (
            #            datetime.now(), t_count, tloss/t_count))
            t_count += 1
    final_vloss.append(vloss/v_count)
    final_tloss.append(tloss/t_count)
    #print "vcount: %d" % v_count
    #print "tcount: %d" % t_count

print "p = " + str(p_d)

print "training loss: "
print final_tloss

print "validation loss: "
print final_vloss