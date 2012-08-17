import nltk
#We will use the nltk NaiveBayesClassifer 
def tset(extractor, tok):
    """function wrapping the apply_feature function. Should pass a
    feature extracting function which returns a featureset - dict mapping features to
    feature values. Tok are tokens which extractor will be applied to."""
    trainset = nltk.classify.apply_features(extractor, tok)
    return trainset

def trainclassifier(trainset):
    return nltk.NaiveBayesClassifier.train(trainset)

def NBCtrain(labeled_featuresets, estimator=nltk.ELEProbDist):
    """A copy of the nltk.NaiveBayesClassifer.train(...)
    method to allow inspection of what the method is actually doing
    and how long it's taking"""
    """ 
    @param labeled_featuresets: A list of classified featuresets, 
             i.e., a list of tuples C{(featureset, label)}. 
          """ 
    label_freqdist = nltk.FreqDist() 
    feature_freqdist = nltk.defaultdict(nltk.FreqDist) 
    feature_values = nltk.defaultdict(set) 
    fnames = set() 

    print 'There are ' + str(len(labeled_featuresets)) + ' labeled featuresets'
    # Count up how many times each feature value occured, given 
    # the label and featurename.
    print 'Counting feature value occurence'
    i = 0
    for featureset, label in labeled_featuresets: 
        label_freqdist.inc(label)
        for fname, fval in featureset.items(): 
            # Increment freq(fval|label, fname) 
            feature_freqdist[label, fname].inc(fval) 
            # Record that fname can take the value fval. 
            feature_values[fname].add(fval) 
            # Keep a list of all feature names. 
            fnames.add(fname)
        if i % 10 == 0:
            print 'At featureset...' + str(i)
        i+=1
   
    # If a feature didn't have a value given for an instance, then 
    # we assume that it gets the implicit value 'None.'  This loop 
    # counts up the number of 'missing' feature values for each 
    # (label,fname) pair, and increments the count of the fval 
    # 'None' by that amount. 
    for label in label_freqdist: 
        num_samples = label_freqdist[label] 
        for fname in fnames: 
            count = feature_freqdist[label, fname].N() 
            feature_freqdist[label, fname].inc(None, num_samples-count) 
            feature_values[fname].add(None) 
   
    # Create the P(label) distribution
    print 'Making the P(label) distribution...'
    label_probdist = estimator(label_freqdist) 

   
    # Create the P(fval|label, fname) distribution
    print 'Making the P(fval|label, fname) distribution from '\
    + str(len(feature_freqdist.items()))\
    + ' feature freqs...'
    feature_probdist = {} 
    for ((label, fname), freqdist) in feature_freqdist.items(): 
        probdist = estimator(freqdist, bins=len(feature_values[fname])) 
        feature_probdist[label,fname] = probdist 
                 
    return nltk.NaiveBayesClassifier(label_probdist, feature_probdist)
