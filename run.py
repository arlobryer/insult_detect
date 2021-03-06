#!/usr/bin/env python
import dat_read
import sys
import csv
import classify
import metrics
import text_extract as te

def comment_reader(f):
    comment = csv.reader(open(f, 'rb'))
    return comment

def output_writer(f):
    write = csv.writer(open(f, 'wb'))
    return write

def extract(tok):
    return dat_read.comment_feat(tok, wordfeat.get_list())

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print 'Syntax is: ./run training_dset.csv analysis_dset.csv output.csv'
        sys.exit()
    train_file = sys.argv[1]
    in_file = sys.argv[2]
    out_file = sys.argv[3]
    #initialising the training file, analysis file,
    #output file, wordfeatures object and how many features to take
    c = comment_reader(train_file)
    test = comment_reader(in_file)
    w = output_writer(out_file)
    wordfeat = dat_read.features()
    
    lines = list(c)
    i = 0
    print 'There are ' + str(len(lines)) + ' training comments.'
    tot = int(raw_input('How many should we analyse for potential words? '))
    #Construct the word feature set
    for r in lines[1:]:
        post = dat_read.comment(r, train = True)
        # print post
        wordfeat(post)
        i+=1
        if i%10 == 0:
            print 'Analysing words...' + str(i)
        if i == tot:
            break
    # print 'This is the word feature set:'
    # print wordfeat.get_list()
    
    # print metrics.word_score(wordfeat.get_freq(), wordfeat.get_lfreq())
    n = 12500
    w_score_list = metrics.word_score(wordfeat.get_freq(), wordfeat.get_lfreq())
    print 'Creating list of ' + str(n) + ' best words of ' + str(len(w_score_list))
    best = te.get_nbest_words(w_score_list, n)
    wordfeat.set_freq(best)
    print wordfeat

    #extract features from each comment in training file
    c1 = int(raw_input('How many comments should we feat extract on? '))
    print 'There are ' + str(wordfeat.get_freq().B()) + ' word features.'
    i = 0
    toks = []
    for r in lines[1:]:
        i+=1
        toks += dat_read.comment(r, train = True).tokenise_com()
        if i%10 == 0:
            print 'Tokenising...' + str(i)
        if i == c1:
            break
    trainset = classify.tset(extract, toks)
    # print 'This is the training set:'
    # print trainset
    print 'Training the classifier...this could take some time.'
    # prog = raw_input('Would you like to see the progress (y/n)?')
    classif = classify.train_maxent(trainset)
    print '*'*8
    print '*'*8
    test_lines = list(test)
    print 'There are ' + str(len(test_lines)) + ' analysis comments.'
    t = int(raw_input('How many should we classify? '))
    i = 0
    for r in test_lines[1:]:
        i += 1
        if i%10 == 0:
            print 'Classifying...' + str(i)
        p_true = classif.prob_classify(extract(dat_read.comment(r))).prob(True)
        w.writerow([p_true, dat_read.comment(r).content])
        if i == t:
            break
    print 'Done. Output is at: ' + out_file
