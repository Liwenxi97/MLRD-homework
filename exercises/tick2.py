from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, split_data
import math
from collections import Counter
import nltk


def calculate_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to prior probability
    """
    resultDict = {}
    sumData = len(training_data)
    posData, negData = 0, 0
    for iDict in training_data:
        if iDict['sentiment'] == 1:
            posData += 1
        elif iDict['sentiment'] == -1:
            negData += 1
    

    resultDict[1] = math.log(posData/sumData)
    resultDict[-1] = math.log(negData/sumData)
    return resultDict


def calculate_unsmoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the unsmoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    negLen = 0
    posLen = 0
    posResult = {}
    negResult = {}
    resultDict = {}
    
    for iDict in training_data:
        for word in iDict['text']:
            if iDict['sentiment'] == 1:
                posLen += 1
                posResult[word] = posResult.setdefault(word,0) + 1
                negResult[word] = negResult.setdefault(word, 0)
                
            else:
                negLen += 1
                negResult[word] = negResult.setdefault(word,0) + 1
                posResult[word] = posResult.setdefault(word, 0)

    posDict = {}
    negDict = {}
    for word in posResult.keys():
        posOcc = posResult[word]
        if posOcc != 0:
            posDict[word] = math.log(posOcc / posLen)
        else:
            posDict[word] = 0
        negOcc = negResult[word]
        if negOcc != 0:
            negDict[word] = math.log(negOcc / negLen)  
        else:
            negDict[word] = 0

    resultDict[1] = posDict
    resultDict[-1] = negDict
    return resultDict



def calculate_smoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment. Use the smoothing
    technique described in the instructions (Laplace smoothing).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: Dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    negLen = 0
    posLen = 0
    posResult = {}
    negResult = {}
    
    resultDict = {}

    for iDict in training_data:
        for word in iDict['text']:
            if iDict['sentiment'] == 1:
                posLen += 1
                posResult[word] = posResult.setdefault(word,0) + 1
                negResult[word] = negResult.setdefault(word, 0)
                
            else:
                negLen += 1
                negResult[word] = negResult.setdefault(word,0) + 1
                posResult[word] = posResult.setdefault(word, 0)

        
   
    vocab_cardinality = len(negResult.keys() | posResult.keys())

    posDict = {}
    negDict = {}
    for word in posResult.keys():
        posOcc = posResult[word] + 1
        posDict[word] = math.log(posOcc / (posLen + vocab_cardinality))
        negOcc = negResult[word] + 1
        negDict[word] = math.log(negOcc / (negLen + vocab_cardinality))

    resultDict[1] = posDict
    resultDict[-1] = negDict
    return resultDict


def predict_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                          class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior probability
    @return: predicted sentiment [-1, 1] for the given review
    """
    
    posTokenLogPro = 0
    negTokenLogPro = 0
    for token in review:
        if token in log_probabilities[1]:
           posTokenLogPro += log_probabilities[1][token]
        
        if token in log_probabilities[-1]:
            negTokenLogPro += log_probabilities[-1][token]
       
    
    if class_log_probabilities[1] + posTokenLogPro >= class_log_probabilities[-1] + negTokenLogPro:
        return 1
    else:
        return -1


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    from exercises.tick1 import accuracy, predict_sentiment, read_lexicon

    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    training_data, validation_data = split_data(review_data, seed=0)
    train_tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in training_data]
    

    dev_tokenized_data = [read_tokens(fn['filename']) for fn in validation_data]
    validation_sentiments = [x['sentiment'] for x in validation_data]

    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    preds_simple = []
    for review in dev_tokenized_data:
        pred = predict_sentiment(review, lexicon)
        preds_simple.append(pred)

    acc_simple = accuracy(preds_simple, validation_sentiments)
    print(f"Your accuracy using simple classifier: {acc_simple}")



    class_priors = calculate_class_log_probabilities(train_tokenized_data)
    
    
    unsmoothed_log_probabilities = calculate_unsmoothed_log_probabilities(train_tokenized_data)
    
    
    preds_unsmoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, unsmoothed_log_probabilities, class_priors)
        preds_unsmoothed.append(pred)

    acc_unsmoothed = accuracy(preds_unsmoothed, validation_sentiments)
    print(f"Your accuracy using unsmoothed probabilities: {acc_unsmoothed}")

    smoothed_log_probabilities = calculate_smoothed_log_probabilities(train_tokenized_data)
    preds_smoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_smoothed.append(pred)

    acc_smoothed = accuracy(preds_smoothed, validation_sentiments)
    print(f"Your accuracy using smoothed probabilities: {acc_smoothed}")


if __name__ == '__main__':
    main()
