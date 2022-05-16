from utils.markov_models import load_dice_data
import os
from exercises.tick7 import estimate_hmm
import random

from typing import List, Dict, Tuple


def viterbi(observed_sequence: List[str], transition_probs: Dict[Tuple[str, str], float], emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model. Use the same symbols for the start and end observations as in tick 7 ('B' for the start observation and 'Z' for the end observation).

    @param observed_sequence: A sequence of observed die rolls
    @param: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    @param: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    @return: The most likely single sequence of hidden states
    """
    
    transitionMatrix = [[0,0,0] for _ in range(3)]
    piMatrix = [0,0,0]

    for k,v in transition_probs.items():
        if k == ('B', 'F'):
            piMatrix[0] = v
        if k == ('B', 'W'):
            piMatrix[1] = v
        if k == ('B', 'Z'):
            piMatrix[2] = v
        if k == ('F', 'F'):
            transitionMatrix[0][0] = v
        if k == ('F', 'W'):
            transitionMatrix[0][1] = v
        if k == ('F', 'Z'):
            transitionMatrix[0][2] = v
        if k == ('W', 'F'):
            transitionMatrix[1][0] = v
        if k == ('W', 'W'):
            transitionMatrix[1][1] = v
        if k == ('W', 'Z'):
            transitionMatrix[1][2] = v
        if k == ('Z', 'F'):
            transitionMatrix[2][0] = v
        if k == ('Z', 'W'):
            transitionMatrix[2][1] = v
        if k == ('Z', 'Z'):
            transitionMatrix[2][2] = v

    emissionMatrix = [[0]*7 for _ in range(3)]
    for k,v in emission_probs.items():
        if k == ('F', '1'):
            emissionMatrix[0][0] = v
        if k == ('F', '2'):
            emissionMatrix[0][1] = v
        if k == ('F', '3'):
            emissionMatrix[0][2] = v
        if k == ('F', '4'):
            emissionMatrix[0][3] = v
        if k == ('F', '5'):
            emissionMatrix[0][4] = v
        if k == ('F', '6'):
            emissionMatrix[0][5] = v
        if k == ('F', 'Z'):
            emissionMatrix[0][6] = v
        if k == ('W', '1'):
            emissionMatrix[1][0] = v
        if k == ('W', '2'):
            emissionMatrix[1][1] = v
        if k == ('W', '3'):
            emissionMatrix[1][2] = v
        if k == ('W', '4'):
            emissionMatrix[1][3] = v
        if k == ('W', '5'):
            emissionMatrix[1][4] = v
        if k == ('W', '6'):
            emissionMatrix[1][5] = v
        if k == ('W', 'Z'):
            emissionMatrix[1][6] = v
        if k == ('Z', '1'):
            emissionMatrix[2][0] = v
        if k == ('Z', '2'):
            emissionMatrix[2][1] = v
        if k == ('Z', '3'):
            emissionMatrix[2][2] = v
        if k == ('Z', '4'):
            emissionMatrix[2][3] = v
        if k == ('Z', '5'):
            emissionMatrix[2][4] = v
        if k == ('Z', '6'):
            emissionMatrix[2][5] = v
        if k == ('Z', 'Z'):
            emissionMatrix[2][6] = v
    
    observed_sequence.append("Z")
    T = len(observed_sequence)
    N = len(transitionMatrix[0])

    delta = [[0]*N for _ in range(T)]
    pathi = [[0]*N for _ in range(T)]

    #init 
    observeDict = {"1":0,"2":1,"3":2,"4":3,"5":4,"6":5,"Z":6}
    # piMatrix: F W Z [0.48055555555555557, 0.5194444444444445, 0.0]
    # transitionMatrix [F[F0.9483570573367035, W0.04995787326504368, Z0.001685069398252849], W[F0.05021061799050886, W0.9482786200522546, Z0.0015107619572365498], Z[F0, W0, Z0]]
    # emissionMatrix [[0.16519888253292536, 0.16510132588355284, 0.16730965367389472, 0.16736286639173428, 0.16788612478382334, 0.16714114673406943, 0.0], [0.5016973854931305, 0.24824485007909283, 0.06332758651333914, 0.06101700940227148, 0.06332758651333914, 0.06238558199882694, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    for i in range(N):
        delta[0][i] = piMatrix[i]*emissionMatrix[i][observeDict[observed_sequence[0]]]
        pathi[0][i] = 0

    # iter
    for t in range(1,T):
        for i in range(N):
            temp, maxIndex = 0,0
            for j in range(N):
                res = delta[t-1][j]*transitionMatrix[j][i]
                if res > temp:
                    temp = res
                    maxIndex = j
            delta[t][i] = temp*emissionMatrix[i][observeDict[observed_sequence[t]]]
            pathi[t][i] = maxIndex

    # end
    p = max(delta[-1])
    
    for i in range(N):
        if delta[-1][i] == p:
            i_T = i
    diceIndex = {0:"F",1:"W",2:"Z"}
    #step4：backtrack
    path = [0]*T
    i_t = i_T
    for t in reversed(range(T-1)):
        i_t = pathi[t+1][i_t]
        path[t] = diceIndex[i_t]
    path[-1] = diceIndex[i_T]

    return path[:-1]


def precision_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the precision of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of predicted weighted states that were actually weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The precision of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    countTP = 0
    countFP = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j] == 1 and true[i][j] == 1:
                countTP += 1
            if pred[i][j] == 1 and true[i][j] == 0:
                countFP += 1
    return countTP / (countTP + countFP)



def recall_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the recall of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of actual weighted states that were predicted weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The recall of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    countTP = 0
    countFN = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j] == 1 and true[i][j] == 1:
                countTP += 1
            if pred[i][j] == 0 and true[i][j] == 1:
                countFN += 1
    return countTP / (countTP + countFN)



def f1_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the F1 measure of the estimated sequence with respect to the positive class (weighted state), i.e. the harmonic mean of precision and recall.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The F1 measure of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    PL = precision_score(pred,true)
    RL = recall_score(pred,true)
    return 2*PL*RL / (PL+RL)



def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    dice_data_shuffled = random.sample(dice_data, len(dice_data))
    dev_size = int(len(dice_data) / 10)
    train = dice_data_shuffled[dev_size:]
    dev = dice_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm(train)
   

    for sample in dev_observed_sequences:
        prediction = viterbi(sample, transition_probs, emission_probs)
        predictions.append(prediction)


    predictions_binarized = [[1 if x == 'W' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x == 'W' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    print(f"Evaluating HMM using cross-validation with 10 folds.")




if __name__ == '__main__':
    main()
