import nltk

wanted = ['PRP$', 'PRP', 'NN', 'NNS', 'VB', 'VBD',
          'VBZ', 'VBG', 'VBP', 'VBN', 'JJ', 'JJR', 'JJS', 
          'MOD', 'UH', '!']
#list of tags for Part-of-speech at:
#http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

grammar = 'NP: {<DT>?<JJ>*<NN>}'

def get_words(com):
    w = nltk.word_tokenize(com)
    w = nltk.pos_tag(w)
    # print w
    #maybe remove some of the non words in the list?    
    out = []
    for t in w:
        if (t[1] in wanted):
            out.append(t)
    # chunker(grammar, out)
    return out

def get_wlist(w_list):
    wordlist = [w[0].lower() for w in w_list]
    return wordlist

def get_wtype_list(w_list):
    wordlist = [w[0].lower() for w in w_list]
    w_type_list = nltk.FreqDist(nltk.pos_tag(wordlist))
    return w_type_list


def get_nbest_words(w_scores, n):
    nbest = sorted(w_scores.iteritems(), key = lambda (word, score):score,
                  reverse = True)[0:n]
    nbest_words = set(word for word, score in nbest)
    return nbest_words


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

def chunker(grammar, sent):
    parser = nltk.RegexpParser(grammar)
    res = parser.parse(sent)
    res.draw()
    return res
    
    
