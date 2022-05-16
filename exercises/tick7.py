from utils.markov_models import load_dice_data, print_matrices
import os
import itertools
from typing import List, Dict, Tuple

def get_transition_probs(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden dice types using maximum likelihood estimation. Counts the number of times each state sequence appears and divides it by the count of all transitions going from that state. The table must include proability values for all state-state pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
    @return: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    """
    stateName = []
    for x in hidden_sequences:
        diff = list(set(x).difference(set(stateName)))
        stateName.extend(diff)
    resultDict = {}
    for state in list(itertools.permutations(stateName,2)) + [(state,state) for state in stateName]:
        count = 0
        total = 0
        for seq in hidden_sequences:
            slow = 0 
            fast = 1
            while fast <= len(seq) - 1:
                if seq[slow] == state[0]:
                    total += 1
                    if seq[fast] == state[1]:
                        count += 1
                    slow += 1
                    fast += 1

                else:
                    slow += 1
                    fast += 1
        try:
            resultDict[state] = count / total
        except ZeroDivisionError:
            resultDict[state] = 0
        

    return resultDict







def get_emission_probs(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden dice states to observed dice rolls, using maximum likelihood estimation. Counts the number of times each dice roll appears for the given state (fair or loaded) and divides it by the count of that state. The table must include proability values for all state-observation pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
    @param observed_sequences: A list of dice roll sequences
    @return: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    """
    stateName = []
    for x in hidden_sequences:
        diff = list(set(x).difference(set(stateName)))
        stateName.extend(diff)
    # print(stateName)
    # print("###############")
    observeName = []
    for x in observed_sequences:
        diff = list(set(x).difference(set(observeName)))
        observeName.extend(diff)
    # print(observeName)
    resultDict = {}
    for pair in itertools.product(stateName,observeName):
        count = 0
        total = 0
        for idx in range(len(hidden_sequences)):
            hiddenSeq = hidden_sequences[idx]
            observeSeq = observed_sequences[idx]
            for i in range(len(hiddenSeq)):
                if hiddenSeq[i] == pair[0]:
                    total += 1
                    if observeSeq[i] == pair[1]:
                        count += 1
        
        try:
            resultDict[pair] = count / total
        except ZeroDivisionError:
            resultDict[pair] = 0
        

    return resultDict
    


def estimate_hmm(training_data: List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities. We use 'B' for the start state and 'Z' for the end state, both for emissions and transitions.

    @param training_data: The dice roll sequence data (visible dice rolls and hidden dice types), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'
    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]
    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]
    transition_probs = get_transition_probs(hidden_sequences)
    emission_probs = get_emission_probs(hidden_sequences, observed_sequences)
    
    return [transition_probs, emission_probs]
    


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))
    
    transition_probs, emission_probs = estimate_hmm(dice_data)
    print(f"The transition probabilities of the HMM:")
    print_matrices(transition_probs)
    print(f"The emission probabilities of the HMM:")
    print_matrices(emission_probs)

if __name__ == '__main__':
    main()