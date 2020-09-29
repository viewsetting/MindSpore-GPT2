import math
import numpy as np
from rouge_score import get_rouge_score
class Accuracy():
    """
    calculate accuracy
    """
    def __init__(self):
        self.acc_num = 0
        self.total_num = 0

    def update(self, logits, labels):
        labels = labels.asnumpy()
        labels = labels[:, 1:]
        # print("Accuracy labels length: {}".format(len(labels[0])))
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logits_id = np.argmax(logits, axis=-1)
        self.acc_num += np.sum(labels == logits_id)
        self.total_num += len(labels)
        print("=========== accuracy is {} ===========".format(self.acc_num / self.total_num))

def postprocess(backpointers, best_tag_id):
    '''
    Do postprocess
    '''
    best_tag_id = best_tag_id.asnumpy()
    batch_size = len(best_tag_id)
    best_path = []
    for i in range(batch_size):
        best_path.append([])
        best_local_id = best_tag_id[i]
        best_path[-1].append(best_local_id)
        for bptrs_t in reversed(backpointers):
            bptrs_t = bptrs_t[0].asnumpy()
            local_idx = bptrs_t[i]
            best_local_id = local_idx[best_local_id]
            best_path[-1].append(best_local_id)
        # Pop off the start tag (we dont want to return that to the caller)
        best_path[-1].pop()
        best_path[-1].reverse()
    return best_path

class F1():
    '''
    calculate F1 score
    '''
    def __init__(self, use_crf=False, num_labels=2):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.use_crf = use_crf
        self.num_labels = num_labels

    def update(self, logits, labels):
        '''
        update F1 score
        '''
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        if self.use_crf:
            backpointers, best_tag_id = logits
            best_path = postprocess(backpointers, best_tag_id)
            logit_id = []
            for ele in best_path:
                logit_id.extend(ele)
        else:
            logits = logits.asnumpy()
            logit_id = np.argmax(logits, axis=-1)
            logit_id = np.reshape(logit_id, -1)
        pos_eva = np.isin(logit_id, [i for i in range(1, self.num_labels)])
        pos_label = np.isin(labels, [i for i in range(1, self.num_labels)])
        self.TP += np.sum(pos_eva&pos_label)
        self.FP += np.sum(pos_eva&(~pos_label))
        self.FN += np.sum((~pos_eva)&pos_label)

class Rouge():
    '''
    Get Rouge Score
    '''
    def __init__(self):
        self.Rouge1 = -1.0
        self.Rouge2 = -1.0
        self.RougeL = -1.0
        self.total_num = 0
        
    def update(self,hypothesis,targets):
        scores = get_rouge_score(hypothesis,targets)
        self.Rouge1 += scores['rouge-1']['f']*100
        self.Rouge2 += scores['rouge-2']['f']*100
        self.RougeL += scores['rouge-l']['f']*100
        total_num += 1
        