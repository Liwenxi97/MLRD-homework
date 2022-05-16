from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, print_binary_confusion_matrix
from exercises.tick1 import accuracy
from exercises.tick2 import predict_sentiment_nbc, calculate_smoothed_log_probabilities, \
    calculate_class_log_probabilities
import random

def generate_random_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, random.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    random.shuffle(training_data)
    
    subNum = len(training_data) // n    
    resultList = [[] for _ in range(n)]
    for i in range(n-1):
        resultList[i].extend(training_data[i*subNum:(i+1)*subNum])

    resultList[n-1].extend(training_data[(n-1)*subNum:])
    return resultList

    # random.shuffle(training_data)
    
    # for i in range((len(training_data) // n) * n):
    #     result[i % n].append(training_data[i])
    # return result



def generate_stratified_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, stratified.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    random.shuffle(training_data)
    resultList = [[] for _ in range(n)]

    posReview = []
    negReview = []

    for iDict in training_data:
        if iDict['sentiment'] == 1:
            posReview.append(iDict)
        if iDict['sentiment'] == -1:
            negReview.append(iDict)

    posSubNum = len(posReview) // n
    negSubNum = len(negReview) // n

    for i in range(n-1):
        subList = posReview[i*posSubNum:(i+1)*posSubNum] + negReview[i*negSubNum:(i+1)*negSubNum]
        resultList[i].extend(subList)
    
    resultList[n-1].extend(posReview[(n-1)*posSubNum:] + negReview[(n-1)*negSubNum:])

    return resultList

    # random.shuffle(training_data)
    # pos = []
    # neg = []
    # for review in training_data:
    #     if review["sentiment"] == 1:
    #         pos.append(review)
    #     else:
    #         neg.append(review)
    # training_data = pos + neg
    # result = [[] for _ in range(n)]
    # for i in range((len(training_data) // n) * n):
    #     result[i % n].append(training_data[i])
    # return result


def cross_validate_nbc(split_training_data: List[List[Dict[str, Union[List[str], int]]]]) -> List[float]:
    """
    Perform an n-fold cross validation, and return the mean accuracy and variance.

    @param split_training_data: a list of n folds, where each fold is a list of training instances, where each instance
        is a dictionary with two fields: 'text' and 'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or
        -1, for positive and negative sentiments.
    @return: list of accuracy scores for each fold
    """
    accuracies = []
    for i in range(len(split_training_data)):
        validation_data = split_training_data[i]
        # dev_tokenized_data = [x['text'] for x in validation_data]
        # validation_sentiments = [x['sentiment'] for x in validation_data]
        training_data = split_training_data[:i] + split_training_data[i+1:]
        flat_training_data = []
        for j in range(len(training_data)):
            for k in range(len(training_data[j])):
                flat_training_data.append(training_data[j][k])
    

        logProp = calculate_smoothed_log_probabilities(flat_training_data)
        classProp = calculate_class_log_probabilities(flat_training_data)
        pred_sentiments = []
        validation_sentiments = []

        for review in validation_data:
            validation_sentiments.append(review['sentiment'])
            pred = predict_sentiment_nbc(review['text'],logProp,classProp)
            pred_sentiments.append(pred)

        acc = accuracy(pred_sentiments, validation_sentiments)
        accuracies.append(acc)

        # accuracy([predict_sentiment_nbc(review["text"], log_probabilities, class_log_probabilities) for review in to_test],
        #     [review["sentiment"] for review in to_test])

    return accuracies



def cross_validation_accuracy(accuracies: List[float]) -> float:
    """Calculate the mean accuracy across n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: mean accuracy over the cross folds
    """
    mean = sum(accuracies) / len(accuracies)
    return mean



def cross_validation_variance(accuracies: List[float]) -> float:
    """Calculate the variance of n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: variance of the cross fold accuracies
    """
    mean = cross_validation_accuracy(accuracies)
    count = 0
    for acc in accuracies:
        count += (acc - mean) ** 2

    return count / len(accuracies)



def confusion_matrix(predicted_sentiments: List[int], actual_sentiments: List[int]) -> List[List[int]]:
    """
    Calculate the number of times (1) the prediction was POS and it was POS [correct], (2) the prediction was POS but
    it was NEG [incorrect], (3) the prediction was NEG and it was POS [incorrect], and (4) the prediction was NEG and it
    was NEG [correct]. Store these values in a list of lists, [[(1), (2)], [(3), (4)]], so they form a confusion matrix:
                     actual:
                     pos     neg
    predicted:  pos  [[(1),  (2)],
                neg   [(3),  (4)]]

    @param actual_sentiments: a list of the true (gold standard) sentiments
    @param predicted_sentiments: a list of the sentiments predicted by a system
    @returns: a confusion matrix
    """
    pp, pn, np, nn = 0, 0, 0, 0
    confusion_matrix = [ [], [] ]
    for i in range(len(predicted_sentiments)):
        if predicted_sentiments[i] == 1 and actual_sentiments[i] == 1:
            pp += 1
        if predicted_sentiments[i] == 1 and actual_sentiments[i] == -1:
            pn += 1
        if predicted_sentiments[i] == -1 and actual_sentiments[i] == 1:
            np += 1
        if predicted_sentiments[i] == -1 and actual_sentiments[i] == -1:
            nn += 1
    confusion_matrix[0].append(pp)
    confusion_matrix[0].append(pn)
    confusion_matrix[1].append(np)
    confusion_matrix[1].append(nn)

    return confusion_matrix


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    # First test cross-fold validation
    folds = generate_random_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Random cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Random cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Random cross validation variance: {variance}\n")

    folds = generate_stratified_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Stratified cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Stratified cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Stratified cross validation variance: {variance}\n")

    # Now evaluate on 2016 and test
    class_priors = calculate_class_log_probabilities(tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(tokenized_data)

    preds_test = []
    test_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_test'))
    test_tokens = [read_tokens(x['filename']) for x in test_data]
    test_sentiments = [x['sentiment'] for x in test_data]
    for review in test_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_test.append(pred)

    acc_smoothed = accuracy(preds_test, test_sentiments)
    print(f"Smoothed Naive Bayes accuracy on held-out data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_test, test_sentiments))

    preds_recent = []
    recent_review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_2016'))
    recent_tokens = [read_tokens(x['filename']) for x in recent_review_data]
    recent_sentiments = [x['sentiment'] for x in recent_review_data]
    for review in recent_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_recent.append(pred)

    acc_smoothed = accuracy(preds_recent, recent_sentiments)
    print(f"Smoothed Naive Bayes accuracy on 2016 data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_recent, recent_sentiments))


if __name__ == '__main__':
    main()
