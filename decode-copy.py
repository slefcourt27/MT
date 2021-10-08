#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple
import copy
import itertools

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxsize, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]
max_perm_len = 1
# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]
sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of
  # the first i words of the input sentence. You should generalize
  # this so that they can represent translations of *any* i words.
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase")
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None)
  stacks = [{} for _ in f]
  # Each of these will have one word in it
  words_remaining = [{} for _ in f]
  stacks[0][lm.begin()] = initial_hypothesis
  words_remaining[0][lm.begin()] = list(f)

  # Initialization
  for idx, french_word in enumerate(f):
      # initialize hypothesis with inital hyp and then this
      stacks[idx][lm.begin()] = initial_hypothesis
      words_remaining[idx][lm.begin()] = list(f)
      lm_state = stacks[idx][lm.begin()].lm_state
      sentence = list(f)

      for phrase in tm[(french_word,)]:
          logprob = stacks[idx][lm.begin()].logprob + phrase.logprob
          for word in phrase.english.split():
            (lm_state, word_logprob) = lm.score(lm_state, word)
            logprob += word_logprob
            # Only doing this because it runs once
            words_remaining[idx][lm_state] = sentence[:]
            words_remaining[idx][lm_state].remove(french_word)
          new_hypothesis = hypothesis(logprob, lm_state, stacks[idx][lm.begin()], phrase)
          stacks[idx][lm_state] = new_hypothesis


  # Go through sentence creating stacks starting from different words
  # End condition when there are no more words remaining
  while True:
      stacks_complete = 0
      if stacks_complete == len(sentence):
          break
      for i, stack in enumerate(stacks):
        h_complete = 0
        for h in sorted(stack.values(),key=lambda h: -h.logprob)[:opts.s]: # prune
          # print(i)
          # print(words_remaining[i])
          # print("=======")
          # print(h)
          if len(words_remaining[i][h.lm_state]) == 0:
              h_complete += 1
              continue
          if h_complete == opts.s:
              stacks_complete += 1
              break
          # Create permutations
          perm_len = 1
          permutations = []
          while perm_len <= max_perm_len:
              permutations += list(itertools.permutations(words_remaining[i][h.lm_state], perm_len))
              perm_len += 1
          for p in permutations:
            if p in tm:
              for phrase in tm[p]:
                logprob = h.logprob + phrase.logprob
                lm_state = h.lm_state
                for word in phrase.english.split():
                  (lm_state, word_logprob) = lm.score(lm_state, word)
                  logprob += word_logprob

                for f_word in p:
                    words_remaining[i][lm_state] = words_remaining[i][h.lm_state][:]
                    words_remaining[i][lm_state].remove(f_word)

                logprob += lm.end(lm_state) if len(words_remaining) == 0 else 0.0

                new_hypothesis = hypothesis(logprob, lm_state, h, phrase)
                if lm_state not in stacks[i] or stacks[i][lm_state].logprob < logprob: # second case is recombination
                  stacks[i][lm_state] = new_hypothesis

  max_h_from_each_stack = []
  for stack in stacks:
      max_h_from_each_stack.append(max(stack[-1].values(), key=lambda h: h.logprob))


  winner = max(max_h_from_each_stack[-1].values(), key=lambda h: h.logprob)
  print("WINNER")
  # print(len(stacks[-1]))
  print("===============")
  print(winner)
  def extract_english(h):
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  print(extract_english(winner))
  exit(1)
  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
