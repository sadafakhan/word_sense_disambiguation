"""Program that performs Word Sense Disambiguation based on noun groups, using
Resnik’s method and WordNet-based similarity measure. Computes and compares similarity scores for a set of
human judgments."""

import os
import sys
import itertools
import nltk
from nltk.corpus import *
from nltk.corpus.reader import information_content
from collections import defaultdict
from scipy.stats import spearmanr

nltk.download('wordnet')
information_content_file = sys.argv[1]
wsd_test_filename = sys.argv[2]
judgement_file = sys.argv[3]
output_filename = sys.argv[4]

# Load information content values for WordNet from a file.
wnic = wordnet_ic.ic(information_content_file)

# Read in a file of (probe word, noun group) pairs
wsd_tests = open(os.path.join(os.path.dirname(__file__), wsd_test_filename), 'r').read().split("\n")[:-1]

# Read in a file of human judgments of similarity between pairs of words.
judgements = open(os.path.join(os.path.dirname(__file__), judgement_file), 'r').read().split("\n")[:-1]

with open(output_filename, 'w') as g:
    # For each (probe word, noun group) pair:
    for pair in wsd_tests:
        pair = pair.split("\t")
        probe = pair[0]
        noun_group = pair[1].split(",")
        probe_senses = wordnet.synsets(probe, pos='n')
        senses = defaultdict(float)

        # Use “Resnik similarity” based on WordNet & information content to compute the preferred WordNet sense for the
        # probe word given the noun group context.
        for context_word in noun_group:
            context_senses = wordnet.synsets(context_word, pos='n')
            sense_comparisons = list(itertools.product(probe_senses, context_senses))

            # word similarity value (i.e. max(sim_concept(c1,c2)))
            v = 0

            # the most informative LCS for probe and context word
            milcs = None

            # find highest IC value of the LCSs across all sense comparisons
            for (c1,c2) in sense_comparisons:
                common = c1.common_hypernyms(c2)

                # find the hypernym with the highest encountered IC and update values
                for parent in common:
                    if information_content(parent, wnic) > v:
                        v = information_content(parent, wnic)
                        milcs = parent

            # add support to the senses of the probe that have have milcs as an ancestor
            for probe_sense in probe_senses:
                if milcs in probe_sense._all_hypernyms:
                    senses[probe_sense] += v

            # b. for each(probe word, noun group word) pair, print the similarity between the probe & context
            g.write("(" + probe + ", " + context_word + ", " + str(v) + ") ")
        g.write("\n")

        # On a separate line, print out the preferred sense, by synsetID, of the word.
        preferred_sense = max(senses.keys(), key=(lambda k: senses[k]))
        g.write(preferred_sense.name() + "\n")

    human_ratings = []
    resnik_scores = []

    # For each word pair in the file:
    for pair in judgements:
        entry = pair.split(",")
        word1 = entry[0]
        word2 = entry[1]
        human_score = float(entry[2])
        human_ratings.append(human_score)

        # Compute the similarity between the two words, using the Resnik similarity measure
        word1_senses = wordnet.synsets(word1, pos='n')
        word2_senses = wordnet.synsets(word2, pos='n')
        sense_comparisons = list(itertools.product(word1_senses, word2_senses))
        senses = defaultdict(float)

        # word similarity value (i.e. max(sim_concept(c1,c2)))
        v = 0

        # find highest IC value of the LCSs across all sense comparisons
        for (c1, c2) in sense_comparisons:
            common = c1.common_hypernyms(c2)

            # find the hypernym with the highest encountered IC and update values
            for parent in common:
                if information_content(parent, wnic) > v:
                    v = information_content(parent, wnic)
        resnik_scores.append(v)

        # Print out the similarity as: wd1,wd2:similarity
        g.write(word1 + "," + word2 + ":" + str(v) + "\n")

    # Compute and print the Spearman correlation between the Resnik similarity scores & the human-generated scores
    correlation = spearmanr(resnik_scores, human_ratings)[0]
    print(resnik_scores)
    print(human_ratings)
    g.write("Correlation:" + str(correlation))
