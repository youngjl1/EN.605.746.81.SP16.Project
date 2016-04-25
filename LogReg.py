'''
Note: Code is adapted from https://www.kaggle.com/c/criteo-display-ad-challenge/forums/t/10322/beat-the-benchmark-with-less-then-200mb-of-memory
'''


from datetime import datetime
from math import exp, log, sqrt
from sklearn.utils import murmurhash3_32

# parameters #################################################################


D = 2 ** 25   # number of weights use for learning
alpha = .1    # learning rate for sgd optimization


# function definitions #######################################################

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
    x = [0]  # 0 is the index of the bias term
    for i, value in enumerate(row):
        #index = int(value + str(i), 16) % D  # weakest hash ever ;)
        index = murmurhash3_32(value + str(i), seed=0) % D
        x.append(index)
    return x  # x contains indices of features that have a value of 1


# C. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def get_p(x, w):
    wTx = 0.
    for i in x:  # do wTx
        wTx += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid


# D. Update given model
# INPUT:
#     w: weights
#     n: a counter that counts the number of times we encounter a feature
#        this is used for adaptive learning rate
#     x: feature
#     p: prediction of our model
#     y: answer
# OUTPUT:
#     w: updated model
#     n: updated count
def update_w(w, n, x, p, y):
    for i in x:
        # alpha / (sqrt(n) + 1) is the adaptive learning rate heuristic
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1
        w[i] -= (p - y) * alpha / (sqrt(n[i]) + 1.)
        n[i] += 1.

    return w, n


# training and testing #######################################################

# initialize our model
w = [0.] * D  # weights
n = [0.] * D  # number of times we've encountered a feature


data_set_size = 100000
k_folds = 5
fold_size = data_set_size / k_folds
final_tloss = []
final_vloss = []
train = 'c:/criteo/smaller_datasets/train-' + str(data_set_size)  # path to training file


val_set = "c:/criteo/temp/val_set.txt"


for idx in range(k_folds):
    v_start = idx*fold_size
    v_end = v_start + fold_size - 1

    # initialize our model
    w = [0.] * D  # weights
    n = [0.] * D  # number of times we've encountered a feature

    # start training a logistic regression model using on pass sgd
    if True:
        loss = 0.
        count = 0
        for t, raw in enumerate(open(train)):
            if t >= v_start and t <= v_end:
                continue

            row = raw.split('\t')
            row[len(row)-1] = row[len(row)-1][:-1] # remove the trailing new line
            y = 1. if row[0] == '1' else 0.

            del row[0]  # can't let the model peek the answer

            # main training procedure
            # step 1, get the hashed features
            x = get_x(row, D)

            # step 2, get prediction
            p = get_p(x, w)
            # for progress validation, useless for learning our model
            loss += logloss(p, y)
            if count % 10000 == 0 and count > 1:
                print('%s\tencountered: %d\tcurrent logloss: %f' % (
                    datetime.now(), t, loss/count))

            # step 3, update model with answer
            w, n = update_w(w, n, x, p, y)
            count += 1
        final_tloss.append(loss/count)


    loss = 0.
    count = 0
    for t, raw in enumerate(open(train)):
        if t >= v_start and t <= v_end:
            row = raw.split('\t')
            row[len(row)-1] = row[len(row)-1][:-1] # remove the trailing new line
            y = 1. if row[0] == '1' else 0.

            del row[0]  # can't let the model peek the answer

            # main training procedure
            # step 1, get the hashed features
            x = get_x(row, D)

            # step 2, get prediction
            p = get_p(x, w)

            # for progress validation, useless for learning our model
            loss += logloss(p, y)
            if count % 10000 == 0 and count > 1:
                print('%s\tV_set encountered: %d\tcurrent logloss: %f' % (
                    datetime.now(), t, loss/count))
            count += 1
            if(t == v_end):
                break
    final_vloss.append(loss/count)


print "training loss: "
print final_tloss

print "validation loss: "
print final_vloss

# testing (build kaggle's submission file)
#with open('submission1234.csv', 'w') as submission:
#    submission.write('Id,Predicted\n')
#    for t, row in enumerate(DictReader(open(test))):
#        Id = row['Id']
#        del row['Id']
#        x = get_x(row, D)
#        p = get_p(x, w)
#        submission.write('%s,%f\n' % (Id, p))