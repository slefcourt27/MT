print#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
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

bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)
theta = defaultdict(float)
null_constant = 4 # amount of null words added to each source sentence


# Get count of unique foreign words


# sentence, foreign sentence, english sentence in all
for (n, (f, e)) in enumerate(bitext):
  # foreign word in foreign sentence
  for f_i in set(f):
    # count of foreign word inc
    f_count[f_i] += 1

    # Eng word found w/ foreign word. inc count
    for e_j in set(e):
      fe_count[(f_i,e_j)] += 1

  for e_j in set(e):
    e_count[e_j] += 1

  if n % 500 == 0:
    sys.stderr.write(".")

#
# print(bitext[0][0])
# print(bitext[0][1])

# Expectation Maximization (EM)
vocab_size = len(f_count.keys())
# initialize model parameters
# 1. initialize model parameters (e.g. uniform)

for key in fe_count.keys():
    theta[key] = 1/vocab_size # size of french vocabulary + NULL word
    # Add null word in foreign language vocabulary
    theta[(None, key[1])] = [1/vocab_size] * null_constant
# 2. assign probabilities to the missing data
iter = 0
while iter < 3:

    idx = 0
    pair_count = len(fe_count.keys())
    fe_count = defaultdict(int, fe_count.fromkeys(fe_count, 0))
    f_count = defaultdict(int, f_count.fromkeys(f_count, 0))
    # while idx < pair_count and idx < vocab_size:
    #     if idx < pair_count:
    #         fe_count.keys()[idx] = 0
    #     if idx < vocab_size:
    #         f_count.keys()[idx] = 0
    #
    #     idx += 1

    for (n, (f, e)) in enumerate(bitext):

        e_count = defaultdict(int, e_count.fromkeys(e_count, 0))

        for e_j in set(e):
            for f_i in set(f):
                e_count[e_j] += theta[(f_i, e_j)]
                fe_count[(f_i, e_j)] += theta[(f_i, e_j)] / e_count[e_j]
                f_count[f_i] += theta[(f_i, e_j)] / e_count[e_j]
# 3. estimate model parameters from completed data
    for f_i in f_count.keys():
        for e_j in e_count.keys():
            # Smoothing such that the rare words are not calculate with too much confidence
            # According to paper https://aclanthology.org/P04-1066.pdf
            theta[(f_i, e_j)] = (fe_count[(f_i, e_j)] + null_constant) / (f_count[f_i] + null_constant * vocab_size)

    iter += 1

# Additions
# giving extra weight to the probability of alignment
# to the null word,

# Decoding function
# TODO: NULL at 0 position
for (f, e) in bitext:
  # Using the dictionary , if f_i is in the key, select the max out of all
  for (i, f_i) in enumerate(f):
    max = 0
    subscript = -1
    for (j, e_j) in enumerate(e):
      if (f_i, e_j) not in theta.keys():
          continue

      if theta[(f_i, e_j)] > max:
        max = theta[(f_i, e_j)]
        subscript = j

    if e[subscript] is not None:
        sys.stdout.write("%i-%i " % (i,subscript))
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
