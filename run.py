#!/usr/bin/env python
import dat_read
import sys
import csv
import classify

wordfeat = dat_read.features()

def comment_reader(f):
    comment = csv.reader(open(f, 'r'))
    return comment    

def extract(tok):
    return dat_read.comment_feat(tok, wordfeat.get_tlist(50))

if __name__ == "__main__":
    in_file = sys.argv[1]
    c = comment_reader(in_file)
    lines = list(c)
    i = 0
    print 'There are ' + str(len(lines)) + ' comments.'
    tot = int(raw_input('How many should we analyse for potential words? '))
    #Construct the word feature set
    for r in lines[1:]:
        post = dat_read.comment(r)
        wordfeat(post)
        i+=1
        if i%10 == 0:
            print i
        if i == tot:
            break
    print 'This is the word feature set:'
    print wordfeat.get_tlist(50)
    
    #extract features from each comment
    c1 = int(raw_input('How many comments should we feat extract on? '))
    print 'There are : ' + str(wordfeat.get_freq().B()) + ' word features.'
    x = int(raw_input('Truncate at (int)?'))
    i = 0
    toks = []
    for r in lines[1:]:
        i+=1
        toks += dat_read.comment(r).tokenise()
        if i == c1:
            break
    print 'These are the tokens we\'re using'
    print toks
    for t in toks:
        trainset = classify.tset(extract, toks)
    
    print 'This is the training set: '
    print trainset
    classif = classify.trainclassifier(trainset)
    print classif.labels()
    print classif.most_informative_features()
    t = int(raw_input('How many should we classify? '))
    i = 0
    for r in lines[1:]:
        i += 1
        print str(i) + ' ' + str(classif.prob_classify(extract(dat_read.comment(r).get_raw_lc())).prob(True))
        if i == t:
            break
        
