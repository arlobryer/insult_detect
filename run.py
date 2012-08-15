#!/usr/bin/env python
import dat_read
import sys
import csv
import classify

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
    feat_list_length = 100
    
    lines = list(c)
    i = 0
    print 'There are ' + str(len(lines)) + ' comments.'
    tot = int(raw_input('How many should we analyse for potential words? '))
    #Construct the word feature set
    for r in lines[1:]:
        post = dat_read.comment(r, train = True)
        wordfeat(post)
        i+=1
        if i%10 == 0:
            print i
        if i == tot:
            break
    print 'This is the word feature set:'
    print wordfeat.get_list()
    
    #extract features from each comment in training file
    c1 = int(raw_input('How many comments should we feat extract on? '))
    print 'There are : ' + str(wordfeat.get_freq().B()) + ' word features.'
    i = 0
    toks = []
    for r in lines[1:]:
        i+=1
        toks += dat_read.comment(r, train = True).tokenise()
        if i == c1:
            break
    for t in toks:
        trainset = classify.tset(extract, toks)
    classif = classify.trainclassifier(trainset)
    print 'These are the most informative features:'
    print classif.most_informative_features()
    print '*'*8
    print '*'*8
    test_lines = list(test)
    print 'There are ' + str(len(test_lines)) + ' analysis comments.'
    t = int(raw_input('How many should we classify? '))
    i = 0
    for r in test_lines[1:]:
        i += 1
        p_true = classif.prob_classify(extract(dat_read.comment(r).get_raw_lc())).prob(True)
        w.writerow([p_true, dat_read.comment(r).content])
        if i == t:
            break
        
