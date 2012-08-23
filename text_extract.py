import nltk
from nltk.corpus import stopwords

wanted = ['PRP$', 'PRP', 'NN', 'NNS', 'VB', 'VBD',
          'VBZ', 'VBG', 'VBP', 'VBN', 'JJ', 'JJR', 'JJS', 
          'UH', '!']
#list of tags for Part-of-speech at:
#http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

grammar = r"""NP: {<JJ>*<NN>}
           {<PRP>}"""

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
    # clean_wordlist = stopword_clean(wordlist, stopwords.words('english'))
    w_type_list = nltk.FreqDist(nltk.pos_tag(wordlist))
    return w_type_list

#http://nltk.googlecode.com/svn/trunk/doc/book/ch07.html

def info_extract(com):
    sent = nltk.sent_tokenize(com)
    words = [nltk.word_tokenize(s) for s in sent]
    w_pos = [nltk.pos_tag(w) for w in words]
    # print w_pos
    out = []
    for c in w_pos:
        for t in c:
            if (t[1] in wanted):
                out.append(t)
    # print chunker(grammar, out)
    return out


def chunker(grammar, sent):
    parser = nltk.RegexpParser(grammar)
    res = parser.parse(sent)
    # res.draw()
    return res

def stopword_clean(l, stops):
    l2 = [w for w in l if w not in stops]
    return l2

    
    
