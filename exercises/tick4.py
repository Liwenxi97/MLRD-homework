import os
from typing import List, Dict, Tuple
from exercises.tick1 import accuracy, predict_sentiment, read_lexicon
from exercises.tick2 import calculate_class_log_probabilities, calculate_smoothed_log_probabilities, predict_sentiment_nbc
from utils.sentiment_detection import read_tokens, load_reviews, split_data
import math

def read_lexicon_magnitude(filename: str) -> Dict[str, Tuple[int, str]]:
    """
    Read the lexicon from a given path.

    @param filename: path to file
    @return: dictionary from word to a tuple of sentiment (1, -1) and magnitude ('strong', 'weak').
    """
    with open(filename, encoding='utf-8') as f:
        txt = f.readlines()
    lexicon_Dict = {}
    for line in txt:
        if "=negative" in line:
            if "=strong" in line:
                lexicon_Dict[line[5:line.index(" ")]] = (-1,"strong")   
            if "=weak" in line:
                lexicon_Dict[line[5:line.index(" ")]] = (-1,"weak")     
        elif "=positive" in line:
            if "=strong" in line:
                lexicon_Dict[line[5:line.index(" ")]] = (1,"strong")
            if "=weak" in line:
                lexicon_Dict[line[5:line.index(" ")]] = (1,"weak")

    return lexicon_Dict



def predict_sentiment_magnitude(review: List[str], lexicon: Dict[str, Tuple[int, str]]) -> int:
    """
    Modify the simple classifier from Tick1 to include the information about the magnitude of a sentiment. Given a list
    of tokens from a tokenized review and a lexicon containing both sentiment and magnitude of a word, determine whether
    the sentiment of each review in the test set is positive or negative based on whether there are more positive or
    negative words. A word with a strong intensity should be weighted *four* times as high for the evaluator.

    @param review: list of tokens from tokenized review
    @param lexicon: dictionary from word to a tuple of sentiment (1, -1) and magnitude ('strong', 'weak').
    @return: calculated sentiment for each review (+1 or -1 for positive or negative sentiments respectively).
    """
    
    mark = 0
    for token in review:
        if token in lexicon:
            if lexicon[token][1] == "strong":
                mark += lexicon[token][0] * 4
            elif lexicon[token][1] == "weak":
                mark += lexicon[token][0]
        else:
            continue
    return 1 if mark >= 0 else -1


def sign_test(actual_sentiments: List[int], classification_a: List[int], classification_b: List[int]) -> float:
    """
    Implement the two-sided sign test algorithm to determine if one classifier is significantly better or worse than
    another. The sign for a result should be determined by which classifier is more correct and the ceiling of the least
    common sign total should be used to calculate the probability.

    @param actual_sentiments: list of correct sentiment for each review
    @param classification_a: list of sentiment prediction from classifier A
    @param classification_b: list of sentiment prediction from classifier B
    @return: p-value of the two-sided sign test.
    """
    null = 0
    plus = 0
    minus = 0
    for i in range(len(actual_sentiments)):
        if classification_a[i] == classification_b[i]:
            null += 1
        else:
            if classification_a[i] == actual_sentiments[i]:
                plus += 1
            if classification_b[i] == actual_sentiments[i]:
                minus += 1

    total = null + plus + minus
    kValue = null // 2 + min(plus, minus)
    pValue = 0
    for idx in range(kValue+1):
        coeff = math.factorial(total) / (math.factorial(idx) * math.factorial(total-idx))
        pValue += coeff * (0.5 ** idx) * (0.5 ** (total-idx))

    return pValue 


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    training_data, validation_data = split_data(review_data, seed=0)

    train_tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in training_data]
    dev_tokenized_data = [read_tokens(fn['filename']) for fn in validation_data]
    validation_sentiments = [x['sentiment'] for x in validation_data]

    lexicon_magnitude = read_lexicon_magnitude(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))
    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    preds_magnitude = []
    preds_simple = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_magnitude(review, lexicon_magnitude)
        preds_magnitude.append(pred)
        pred_simple = predict_sentiment(review, lexicon)
        preds_simple.append(pred_simple)

    acc_magnitude = accuracy(preds_magnitude, validation_sentiments)
    acc_simple = accuracy(preds_simple, validation_sentiments)

    print(f"Your accuracy using simple classifier: {acc_simple}")
    print(f"Your accuracy using magnitude classifier: {acc_magnitude}")

    class_priors = calculate_class_log_probabilities(train_tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(train_tokenized_data)

    preds_nb = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_nb.append(pred)

    acc_nb = accuracy(preds_nb, validation_sentiments)
    print(f"Your accuracy using Naive Bayes classifier: {acc_nb}\n")

    p_value_magnitude_simple = sign_test(validation_sentiments, preds_simple, preds_magnitude)
    print(f"The p-value of the two-sided sign test for classifier_a \"{'classifier simple'}\" and classifier_b \"{'classifier magnitude'}\": {p_value_magnitude_simple}")

    p_value_magnitude_nb = sign_test(validation_sentiments, preds_nb, preds_magnitude)
    print(f"The p-value of the two-sided sign test for classifier_a \"{'classifier magnitude'}\" and classifier_b \"{'naive bayes classifier'}\": {p_value_magnitude_nb}")



if __name__ == '__main__':
    main()
