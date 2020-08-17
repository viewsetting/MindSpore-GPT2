from typing import Union
import numpy as np

def ngram_ppl(prob: Union[np.ndarray, list], length: int, log_softmax=False, index: float=np.e):
    """
    Calculate Perplexity(PPL) score under N-gram language model.

    Please make sure the sum of `prob` is 1.
    Otherwise, assign `normalize=True`.

    The number of N is depended by model.

    Args:
        prob (Union[list, np.ndarray]): Prediction probability
            of the sentence.
        log_softmax (bool): If sum of `prob` is not 1, please
            set normalize=True.
        index (float): Base number of log softmax.

    Returns:
        float, ppl score.
    """
    if not length:
        return np.inf
    if not isinstance(prob, (np.ndarray, list)):
        raise TypeError("`prob` must be type of list or np.ndarray.")
    if not isinstance(prob, np.ndarray):
        prob = np.array(prob)
    if prob.shape[0] == 0:
        raise ValueError("`prob` length must greater than 0.")

    print(f'length:{length}, log_prob:{prob}')

    if log_softmax:
        prob = np.sum(prob) / length
        ppl = 1. / np.power(index, prob)
        print(f'avg log prob:{prob}')
    else:
        p = 1.
        for i in range(prob.shape[0]):
            p *= (1. / prob[i])
        ppl = pow(p, 1 / length)

    print(f'ppl val:{ppl}')
    return ppl