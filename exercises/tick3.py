from utils.sentiment_detection import clean_plot, read_tokens, chart_plot, best_fit
from typing import List, Tuple, Callable
import os
from os import listdir
import re
import math
def estimate_zipf(token_frequencies_log: List[Tuple[float, float]], token_frequencies: List[Tuple[int, int]]) \
        -> Callable:
    """
    Use the provided least squares algorithm to estimate a line of best fit in the log-log plot of rank against
    frequency. Weight each word by its frequency to avoid distortion in favour of less common words. Use this to
    create a function which given a rank can output an expected frequency.

    @param token_frequencies_log: list of tuples of log rank and log frequency for each word
    @param token_frequencies: list of tuples of rank to frequency for each word used for weighting
    @return: a function estimating a word's frequency from its rank
    """
    param = best_fit(token_frequencies_log,token_frequencies)
    def predict_frequencies(logRank):
        logFre = param[0] * logRank + param[1]
        return math.exp(logFre)




def count_token_frequencies(dataset_path: str) -> List[Tuple[str, int]]:

    """
    For each of the words in the dataset, calculate its frequency within the dataset.

    @param dataset_path: a path to a folder with a list of  reviews
    @returns: a list of the frequency for each word in the form [(word, frequency), (word, frequency) ...], sorted by
        frequency in descending order
    """
    fileName = listdir(dataset_path)
    allToken = []
    for file in fileName:
        with open (os.path.join(dataset_path, file), encoding="utf-8") as f:
            txtList = f.read().split()
            allToken.extend([w.lower() for w in txtList])
    
    myDict = {}
    for key in allToken:
        myDict[key] = myDict.get(key,0) + 1 

    resultList = [(k, v/len(allToken)) for k, v in myDict.items()]
    resultList.sort(key=lambda tup: tup[1], reverse=True)
   
    return resultList

    
    


def draw_frequency_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the provided chart plotting program to plot the most common 10000 word ranks against their frequencies.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    myFrequencies = []
    for idx, tup in enumerate(frequencies[:10000]):
        newTuple = (float(idx),float(tup[1]))
        myFrequencies.append(newTuple)

    chart_plot(myFrequencies, "Top 10000 ranks against their frequencies", "rank", "frequencies")
    


def draw_selected_words_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot your 10 selected words' word frequencies (from Task 1) against their
    ranks. Plot the Task 1 words on the frequency-rank plot as a separate series on the same plot (i.e., tell the
    plotter to draw it as an additional graph on top of the graph from above function).

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    tokenFrequencies = []
    tokenList = ["excellent", "amazing", "disastrous", "astonishing", "terrible", "ridiculous", "horrific", "awesome", "awful", "wonderful"]
    for idx, tup in enumerate(frequencies[:10000]):
        if tup[0] in tokenList:
            newTuple = (float(idx),float(tup[1]))
            tokenFrequencies.append(newTuple)
    chart_plot(tokenFrequencies, "Top 10000 ranks against their frequencies", "rank", "frequencies")


def draw_zipf(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot the logs of your first 10000 word frequencies against the logs of their
    ranks. Also use your estimation function to plot a line of best fit.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    myLogFrequencies = []
    for idx, tup in enumerate(frequencies[:10000]):
        newLogTuple = (math.log(float(idx)+1),math.log(float(tup[1])))
        myLogFrequencies.append(newLogTuple)

    chart_plot(myLogFrequencies, "The logs of Top 10000 ranks against their logs of frequencies", "log-rank", "log-frequencies")
    

    


def compute_type_count(dataset_path: str) -> List[Tuple[int, int]]:
    """
     Go through the words in the dataset; record the number of unique words against the total number of words for total
     numbers of words that are powers of 2 (starting at 2^0, until the end of the data-set)

     @param dataset_path: a path to a folder with a list of  reviews
     @returns: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    fileName = listdir(dataset_path)
    allToken = []
    for file in fileName:
        with open (os.path.join(dataset_path, file), encoding="utf-8") as f:
            txtList = f.read().split()
            allToken.extend([w.lower() for w in txtList])
    
    start = 0 
    resultList = []
    while 2 ** start <= len(allToken):
        typeCount = len(list(set(allToken[:2 ** start])))
        tokenCount = 2 ** start
        tup = (tokenCount,typeCount)
        resultList.append(tup)
        start += 1

    allTypeCount = len(list(set(allToken)))
    resultList.append((len(allToken),allTypeCount))
    return resultList
    


def draw_heap(type_counts: List[Tuple[int, int]]) -> None:
    """
    Use the provided chart plotting program to plot the logs of the number of unique words against the logs of the
    number of total words.

    @param type_counts: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    log_type_counts = []
    for tup in type_counts:
        newTup = (math.log(tup[0]),math.log(tup[1]))
        log_type_counts.append(newTup)
    
    chart_plot(log_type_counts, "the logs of the number of types for every log-n tokens", "token-num", "type-num")


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    frequencies = count_token_frequencies(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))
    

    draw_frequency_ranks(frequencies)
    draw_selected_words_ranks(frequencies)

    clean_plot()
    draw_zipf(frequencies)

    clean_plot()
    tokens = compute_type_count(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))
    draw_heap(tokens)


if __name__ == '__main__':
    main()
