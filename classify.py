import nltk
#We will use the nltk NaiveBayesClassifer first
#pass the whole file of comments to coms
def tset(extractor, tok):
    """function wrapping the apply_feature function. Should pass a
    feature extracting function which returns a featureset - dict mapping features to
    feature values. Tok are tokens which extractor will be applied to."""
    trainset = nltk.classify.apply_features(extractor, tok)
    return trainset

def trainclassifier(trainset):
    return nltk.NaiveBayesClassifier.train(trainset)
