import os
from typing import List, Dict, Union

from utils.sentiment_detection import load_reviews, read_tokens, read_student_review_predictions, print_agreement_table

from exercises.tick5 import generate_random_cross_folds, cross_validation_accuracy

import math


def nuanced_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c) for nuanced sentiments.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to prior probability
    """
    resultDict = {}
    posNum, neuNum, negNum = 0, 0, 0

    for review in training_data:
        if review['sentiment'] == 1:
            posNum += 1
        elif review['sentiment'] == -1:
            negNum += 1
        else:
            neuNum += 1
    resultDict[1] = math.log(posNum / len(training_data))
    resultDict[0] = math.log(neuNum / len(training_data))
    resultDict[-1] = math.log(negNum / len(training_data))

    return resultDict



def nuanced_conditional_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a nuanced sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    resultDict = {}
    posDict = {}
    negDict = {}
    neuDict = {}
    posLen = 0
    negLen = 0
    neuLen = 0
    for review in training_data:
        for word in review['text']:
            if review['sentiment'] == 1:
                posLen += 1
                posDict[word] = posDict.setdefault(word,0) + 1
                negDict[word] = negDict.setdefault(word,0)
                neuDict[word] = neuDict.setdefault(word,0)
            elif review['sentiment'] == -1:
                negLen += 1
                negDict[word] = negDict.setdefault(word,0) + 1
                posDict[word] = posDict.setdefault(word,0)
                neuDict[word] = neuDict.setdefault(word,0)
            else:
                neuLen += 1
                neuDict[word] = neuDict.setdefault(word,0) + 1
                posDict[word] = posDict.setdefault(word,0)
                negDict[word] = negDict.setdefault(word,0)

    vocab = len (posDict.keys() | negDict.keys() | neuDict.keys())

    posResult = {}
    negResult = {}
    neuResult = {}
    for token in posDict.keys():
        posDict[token] = math.log((posDict[token] + 1) / (posLen + vocab))
        negDict[token] = math.log((negDict[token] + 1) / (negLen + vocab))
        neuDict[token] = math.log((neuDict[token] + 1) / (neuLen + vocab))
    
    resultDict[1] = posDict
    resultDict[-1] = negDict
    resultDict[0] = neuDict

    return resultDict


def nuanced_accuracy(pred: List[int], true: List[int]) -> float:
    """
    Calculate the proportion of predicted sentiments that were correct.

    @param pred: list of calculated sentiment for each review
    @param true: list of correct sentiment for each review
    @return: the overall accuracy of the predictions
    """
    
    correct = 0
    for i in range(len(pred)):
        if pred[i] == true[i]:
            correct += 1
    
    return correct / len(pred)


def predict_nuanced_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                                  class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior probability
    @return: predicted sentiment [-1, 0, 1] for the given review
    """
    posValue,negValue,neuValue = 0, 0, 0 
    for token in review:
        if token in log_probabilities[1]:
            posValue += log_probabilities[1][token]
        if token in log_probabilities[-1]:
            negValue += log_probabilities[-1][token]
        if token in log_probabilities[0]:
            neuValue += log_probabilities[0][token]

    maxProb = max(posValue+class_log_probabilities[1], negValue+class_log_probabilities[-1], neuValue+class_log_probabilities[0])

    if maxProb == posValue+class_log_probabilities[1]:
        return 1
    elif maxProb == negValue+class_log_probabilities[-1]:
        return -1
    else:
        return 0



def calculate_kappa(agreement_table: Dict[int, Dict[int,int]]) -> float:
    """
    Using your agreement table, calculate the kappa value for how much agreement there was; 1 should mean total agreement and -1 should mean total disagreement.

    @param agreement_table:  For each review (1, 2, 3, 4) the number of predictions that predicted each sentiment
    @return: The kappa value, between -1 and 1
    """
    
    possible_rater_pairs = (list(agreement_table.values())[0][1] + list(agreement_table.values())[0][-1]) * (list(agreement_table.values())[0][1] + list(agreement_table.values())[0][-1] - 1) 

    portion = 0
    p_e_value_pos = 0
    p_e_value_neg = 0
    for i in agreement_table.keys():
        portion += (agreement_table[i][1] * (agreement_table[i][1] - 1) + agreement_table[i][-1] * (agreement_table[i][-1] - 1)) / possible_rater_pairs
        p_e_value_pos += agreement_table[i][1] / (agreement_table[i][1] + agreement_table[i][-1])
        p_e_value_neg += agreement_table[i][-1] / (agreement_table[i][1] + agreement_table[i][-1])


    p_a_value = portion / len(agreement_table.keys())
    p_e_value = (p_e_value_pos / len(agreement_table.keys())) ** 2 + (p_e_value_neg / len(agreement_table.keys())) ** 2
    kappa = (p_a_value - p_e_value) / (1 - p_e_value)

    return kappa 


def get_agreement_table(review_predictions: List[Dict[int, int]]) -> Dict[int, Dict[int,int]]:
    """
    Builds an agreement table from the student predictions.

    @param review_predictions: a list of predictions for each student, the predictions are encoded as dictionaries, with the key being the review id and the value the predicted sentiment
    @return: an agreement table, which for each review contains the number of predictions that predicted each sentiment.
    """
    fir_pos, fir_neg, sec_pos, sec_neg, thr_pos, thr_neg, for_pos, for_neg = 0, 0, 0, 0, 0, 0, 0, 0

    dict_Occ = {}
    
    for pred in review_predictions:
        if pred[0] == 1:
            fir_pos += 1
        if pred[0] == -1:
            fir_neg += 1
        if pred[1] == 1:
            sec_pos += 1
        if pred[1] == -1:
            sec_neg += 1
        if pred[2] == 1:
            thr_pos += 1
        if pred[2] == -1:
            thr_neg += 1
        if pred[3] == 1:
            for_pos += 1
        if pred[3] == -1:
            for_neg += 1
    dict_Occ[0] = {1:fir_pos, -1:fir_neg}
    dict_Occ[1] = {1:sec_pos, -1:sec_neg}
    dict_Occ[2] = {1:thr_pos, -1:thr_neg}
    dict_Occ[3] = {1:for_pos, -1:for_neg}

    return dict_Occ

def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    # review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_nuanced'), include_nuance=True)
    # tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    # split_training_data = generate_random_cross_folds(tokenized_data, n=10)

    # n = len(split_training_data)
    # accuracies = []
    # for i in range(n):
    #     test = split_training_data[i]
    #     train_unflattened = split_training_data[:i] + split_training_data[i+1:]
    #     train = [item for sublist in train_unflattened for item in sublist]

    #     dev_tokens = [x['text'] for x in test]
    #     dev_sentiments = [x['sentiment'] for x in test]

    #     class_priors = nuanced_class_log_probabilities(train)
    #     nuanced_log_probabilities = nuanced_conditional_log_probabilities(train)
    #     preds_nuanced = []
    #     for review in dev_tokens:
    #         pred = predict_nuanced_sentiment_nbc(review, nuanced_log_probabilities, class_priors)
    #         preds_nuanced.append(pred)
    #     acc_nuanced = nuanced_accuracy(preds_nuanced, dev_sentiments)
    #     accuracies.append(acc_nuanced)

    # mean_accuracy = cross_validation_accuracy(accuracies)
    # print(f"Your accuracy on the nuanced dataset: {mean_accuracy}\n")

    review_predictions = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions.csv'))

    print('Agreement table for this year.')

    agreement_table = get_agreement_table(review_predictions)
    print_agreement_table(agreement_table)

    fleiss_kappa = calculate_kappa(agreement_table)

    print(f"The cohen kappa score for the review predictions is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [0, 1]})

    print(f"The cohen kappa score for the review predictions of review 1 and 2 is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [2, 3]})

    print(f"The cohen kappa score for the review predictions of review 3 and 4 is {fleiss_kappa}.\n")

    # review_predictions_four_years = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions_2019_2022.csv'))
    # agreement_table_four_years = get_agreement_table(review_predictions_four_years)

    # print('Agreement table for the years 2019 to 2022.')
    # print_agreement_table(agreement_table_four_years)

    # fleiss_kappa = calculate_kappa(agreement_table_four_years)

    # print(f"The cohen kappa score for the review predictions from 2019 to 2022 is {fleiss_kappa}.")



if __name__ == '__main__':
    main()
