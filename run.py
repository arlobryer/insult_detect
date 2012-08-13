#!/usr/bin/env python
import dat_read
import sys
import csv
import classify

wordfeat = dat_read.features()

def comment_reader(f):
    comment = csv.reader(open(f, 'rb'))
    return comment

def output_writer(f):
    write = csv.writer(open(f, 'wb'))
    return write

def extract(tok):
    return dat_read.comment_feat(tok, wordfeat.get_tlist(50))

if __name__ == "__main__":
    in_file = sys.argv[1]
    if len(sys.argv) > 2:
        out_file = sys.argv[2]
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
    for t in toks:
        trainset = classify.tset(extract, toks)
    classif = classify.trainclassifier(trainset)
    print classif.labels()
    print classif.most_informative_features()
    t = int(raw_input('How many should we classify? '))
    i = 0
    w = output_writer(out_file)
    for r in lines[1:]:
        i += 1
        p_true = classif.prob_classify(extract(dat_read.comment(r).get_raw_lc())).prob(True)
        w.writerow([p_true, dat_read.comment(r).content])
        if i == t:
            break
        
