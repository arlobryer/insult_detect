import nltk
import text_extract as te
import datetime

def comment_feat(tok, w_feat = None):
    raw = tok.get_raw_lc()
    features = {}
    #make a bag of words
    for w in w_feat:
        features[('contains-word(%s)' %w[0])] = w[0] in raw
    if tok.date is not None:
        features['hour'] = tok.date[1]
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
        self.freq.update(te.get_wlist(com.wlist()))
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
        return 'Is insult: ' + str(self.insult) + '\n'\
               'Date: ' + str(self.date) + '\n'\
               'Content: ' + self.content
    def wlist(self):
        return te.get_words(self.content)
    def get_date(self):
        return self.date
    def get_content(self):
        return self.content.rstrip('"').lstrip('"')
    def get_raw_lc(self):
        return ([w.lower() for w in nltk.word_tokenize(self.get_content())])
    def tokenise(self):
        return([[self.get_raw_lc(), bool(int(self.insult))]])
    def tokenise_com(self):
        return([[self, bool(int(self.insult))]])
