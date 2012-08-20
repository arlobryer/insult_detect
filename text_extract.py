import nltk

wanted = ['PRP$', 'PRP', 'NN', 'VBP', 'ADJ', 'MOD', 'UH', '!']

def get_words(com):
    w = nltk.word_tokenize(com)
    w = nltk.pos_tag(w)
    #maybe remove some of the non words in the list?    
    out = []
    for t in w:
        if (t[1] in wanted):
            out.append(t)
    return out

def get_wlist(w_list):
    wordlist = [w[0].lower() for w in w_list]
    w_type_list = nltk.FreqDist(nltk.pos_tag(wordlist))
    return w_type_list

#http://nltk.googlecode.com/svn/trunk/doc/book/ch07.html

def info_extract(com):
    w = nltk.sent_tokenize(com)
    w = [nltk.word_tokenize(sent) for sent in w]
    w = [nltk.pos_tag(sent) for sent in w]
    out = []
    for t in w:
        if (t[1] in wanted):
            out.append(t)
    return out
    
