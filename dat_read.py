import nltk
import text_extract as te
import datetime

def comment_feat(tok, w_feat = None):
    raw = tok.get_raw_lc()
    features = {}
    #make a bag of words
    for w in w_feat:
        features[('contains-word(%s)' %w)] = w in raw
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
        #freq of all the words
        self.freq = nltk.FreqDist()
        #freq of words by class (insult/not insult)
        self.label_freq = nltk.ConditionalFreqDist()
    def __call__(self, com):
        #update the dist of all words
        self.freq.update(te.get_wlist(com.wlist()))
        #update the conditional dists
        self.label_freq[com.is_insult()].update(te.get_wlist(com.wlist()))
    def get_freq(self):
        return self.freq
    def get_list(self):
        """Return a list ordered in decreasing frequency"""
        #seems to return the same thing as .samples()
        return self.freq.keys()
    def get_tlist(self, trunc):
        return self.freq.keys()[0:trunc]

class comment:
    def __init__(self, com, train = None):
        if train == True:
            self.insult = bool(int(com[0]))
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
    def is_insult(self):
        return self.insult
    def get_date(self):
        return self.date
    def get_content(self):
        return self.content.rstrip('"').lstrip('"')
    def get_raw_lc(self):
        return ([w.lower() for w in nltk.word_tokenize(self.get_content())])
    def tokenise(self):
        return([[self.get_raw_lc(), self.insult]])
    def tokenise_com(self):
        return([[self, self.insult]])
