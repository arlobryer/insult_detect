#This file contains the various metrics used to optimise the
#feature selection.
# A good description for the general idea can be found at:
#http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

#Information gain on each word
#use a chi^2 to evalute the information gain of each word
def word_score(f_words, label_f_words):
    t_wc = label_f_words[True].N()
    f_wc = label_f_words[False].N()
    tot_wc = t_wc + f_wc
    word_scores = {}
    for word, freq in f_words.iteritems():
        t_score = BigramAssocMeasures.chi_sq(label_f_words[True][word],
                                            (freq, t_wc), tot_wc)
        f_score = BigramAssocMeasures.chi_sq(label_f_words[False][word],
                                            (freq, f_wc), tot_wc)
        word_scores[word] = t_score + f_score

    return word_scores
