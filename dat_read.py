import nltk
import datetime

def get_words(com):
    w = nltk.word_tokenize(com)
    w = nltk.pos_tag(w)
    #maybe remove some of the non words in the list?
    wanted = ['PRP$', 'PRP', 'NN', 'VBP', 'ADJ', 'MOD', 'UH', '!']
    out = []
    for t in w:
        if (t[1] in wanted):
            out.append(t)
    return out

def get_wlist(w_list):
    wordlist = [w[0].lower() for w in w_list]
    w_type_list = nltk.FreqDist(nltk.pos_tag(wordlist))
    return w_type_list

def comment_feat(tok, w_feat = None):
    features = {}
    for w in w_feat:
        features[('contains-word(%s)' %w[0])] = w[0] in tok
    # features['date'] = com.get_date()
    #get the date in later
    return features

def make_date(da):
    date = None
    if da!='':
      y = int(da[:4])
      m = int(da[4:6])
      d = int(da[6:8])
      h = int(da[8:10])
      dt = datetime.date(y, m, d)
      date = (dt.weekday(), h)
    return date

class features:
    def __init__(self):
        self.freq = nltk.FreqDist()
    def __call__(self, com):
        self.freq.update(get_wlist(com.wlist()))
    def get_freq(self):
        return self.freq
    def get_list(self):
        return list(self.freq)
    def get_tlist(self, trunc):
        return self.freq.keys()[0:trunc]

class comment:
    def __init__(self, com, train = None):
        if train == True:
            self.insult = com[0]
            self.date = make_date(com[1])
            self.content = com[2]
        else:
            self.insult = None
            self.date = make_date(com[0])
            self.content = com[1]
    def __str__(self):
        return 'Is insult: ' + self.insult + '\n'\
               'Date: ' + str(self.date) + '\n'\
               'Content: ' + self.content
    def wlist(self):
        return get_words(self.content)
    def get_date(self):
        return self.date
    def get_content(self):
        return self.content.rstrip('"').lstrip('"')
    def get_raw_lc(self):
        return ([w.lower() for w in nltk.word_tokenize(self.get_content())])
    def tokenise(self):
        return([[self.get_raw_lc(), bool(int(self.insult))]])
