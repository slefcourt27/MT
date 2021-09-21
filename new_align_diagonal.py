#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
import numpy as np

import random
optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)


sys.stderr.write("Training with EM...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]

e_count = defaultdict(int)
fe_count = defaultdict(int)

# Dictionary for mapping to word and word to index which corresponds to place in matrix
e_correspondance = {}
f_correspondance = {}

# Add null word
# e_correspondance['NULL'] = 0
# f_correspondance['NULL'] = 0

f_idx = 0 # idx of the french word and serves as size
e_idx = 0 # idx of the eng words & serves as size

# Iterate over data and generate vocabulary
for (n, (f, e)) in enumerate(bitext):
    for f_i in set(f):
        if f_i not in f_correspondance.keys():
            f_correspondance[f_i] = f_idx
            f_idx += 1

        for e_i in set(e):
            e_count[e_i] += 1
            fe_count[f_i, e_i] += 1
            if e_i not in e_correspondance.keys():
                e_correspondance[e_i] = e_idx
                e_idx += 1


    for e_sentence in e:
        for e_i in set(e_sentence):
            e_count[e_i] += 0




# Create theta matrix and fill uniformly
theta = np.full((f_idx, e_idx), 1/e_idx)

# For lookup later
e_key_list = list(e_correspondance.keys())
e_val_list = list(e_correspondance.values())
f_key_list = list(f_correspondance.keys())
f_val_list = list(f_correspondance.values())

null_constant = 1 # amount of null words added to each source sentence

# Expectation Maximization (EM) in a nutshell

# 2. assign probabilities to the missing data
iter = 0
while iter < 4:
    for key in fe_count.keys():
        fe_count[key] = 0

    for (n, (f, e)) in enumerate(bitext):
        # f.insert(0, "NULL")
        # e.insert(0, "NULL")
        for key in e_count.keys(): # for all sent pairs set total_s [e] = 0
            e_count[key] = 0

        for f_i in set(f):
            for e_j in set(e):
                matrix_f_i = f_correspondance[f_i]
                matrix_e_j = e_correspondance[e_j]
                if e_j not in e_count.keys():
                    sys.stdout.write(e_j, "not in e count.")
                    if (f_i, e_j) not in fe_count.keys():
                        print("Pair not in the count")
                    exit(-1)
                e_count[e_j] += theta[matrix_f_i][matrix_e_j]
                fe_count[(f_i, e_j)] += theta[matrix_f_i][matrix_e_j] / e_count[e_j]

# 3. estimate model parameters from completed data
#     # Iterate through the matrix row by col
#     row = 0
#     col = 0
#
#     num_rows = theta.shape[0] # french word
#     num_cols = theta.shape[1] # english word
#     while row < num_rows:
#         f_i = f_key_list[f_val_list.index(row)]
#         while col < num_cols:
#             e_j = e_key_list[e_val_list.index(col)]
#             print("E_j is", e_j)
#             print("Pair count: ", fe_count[(f_i, e_j)])
#
#             theta[row][col] = fe_count[(f_i, e_j)] / e_count[e_j] # fe_count[(f_i, e_j)] + null_constant / (e_count[f_i] + null_constant * e_idx)
#             col += 1
#         row += 1
#
    # print("Keys in e count: ", len(e_count.keys()))
    for (n, (f, e)) in enumerate(bitext):
        for f_i in set(f):
            for e_j in set(e):
                matrix_f_i = f_correspondance[f_i]
                matrix_e_j = e_correspondance[e_j]

                theta[matrix_f_i][matrix_e_j] = fe_count[(f_i, e_j)] / e_count[e_j]
    iter += 1



# Additions
# giving extra weight to the probability of alignment
# to the null word,

# Decoding function
# TODO: NULL at 0 position
for (f, e) in bitext:
    # f.insert(0, "NULL")
    # e.insert(0, "NULL")
    # Using the dictionary , if f_i is in the key, select the max out of all
    for (i, f_i) in enumerate(f):
        largest = 0
        subscript = -1
        for (j, e_j) in enumerate(e):
            if (f_i, e_j) not in fe_count.keys():
                continue

            matrix_f_i = f_correspondance[f_i]
            matrix_e_j = e_correspondance[e_j]
            weight = theta[matrix_f_i][matrix_e_j]
            print("Weight", weight, j)
            if weight > largest:
                largest = weight
                subscript = j

        if subscript != -1:
            sys.stdout.write("%i-%i " % (i, subscript))
        sys.stdout.write("\n")

# Referenced http://mt-class.org/jhu/assets/papers/alopez-model1-tutorial.pdf
# and pseudocode from https://www.cis.uni-muenchen.de/~fraser/readinggroup/model1.html
# initialize t(e|f) uniformly
#  do until convergence
#    set count(e|f) to 0 for all e,f
#    set total(f) to 0 for all f
#    for all sentence pairs (e_s,f_s)
#      set total_s(e) = 0 for all e
#      for all words e in e_s
#        for all words f in f_s
#          total_s(e) += t(e|f)
#      for all words e in e_s
#        for all words f in f_s
#          count(e|f) += t(e|f) / total_s(e)
#          total(f)   += t(e|f) / total_s(e)
#    for all f
#      for all e
#        t(e|f) = count(e|f) / total(f)
