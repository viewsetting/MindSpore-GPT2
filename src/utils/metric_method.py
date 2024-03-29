import math
import numpy as np
import tempfile
import re
import string
import subprocess
from .rouge_score import get_rouge_score
from .bleu_score import sum_bleu
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

class LastTokenAccuracy():
    """
    calculate accuracy
    """
    def __init__(self):
        self.acc_num = 0
        self.total_num = 0

    def update(self, logits, labels):
        labels = labels.asnumpy()
        #labels = labels[:, 1:]
        # print("Accuracy labels length: {}".format(len(labels[0])))
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logits_id = np.argmax(logits, axis=-1)
        logits_id = np.reshape(logits_id,-1)    
#         print("After argmax: logits_id shape,label shape:",logits_id.shape,labels.shape)
#         print("logits_id: ",logits_id)
#         print("labels_id: ",labels)
        self.acc_num += np.sum(labels == logits_id)
        self.total_num += len(labels)
        print("acc_num:",self.acc_num)
        print("total_num",self.total_num)
        print("=========== Last token Accuracy is {} ===========".format(self.acc_num / self.total_num)) 
    
  
class LastWordAccuracy():
    def __init__(self,smooth=True,min_overlap=3):
        self.acc_num = 0
        self.total_num = 0
        self.smooth = smooth
        self.min_overlap = min_overlap
    def normalize(self,word):
        word = word.lstrip()
        word = word.rstrip()
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return remove_punc(lower(word))
    
    # (output_string,label_string)
    def overlap(self,a,b):
        len_b = len(b)
        max_len = 0
        for i in range(len_b-1):
            for j in range(i+self.min_overlap,len_b+1):
                b_ =b[i:j]
                if b_ in a:
                    max_len = max(max_len,len(b_))
                else:
                    break
        return max_len / len(a)

    def update(self,output,label):
        if type(output) is str and type(label) is str:
            output = [output]
            label = [label]
        for output_word,label_word in zip(output,label):
            self.total_num += 1
#             if self.normalize(output_word) == self.normalize(label_word):
#                 self.acc_num+=1
            if self.smooth is False:
                if self.normalize(output_word) == self.normalize(label_word):
                    self.acc_num+=1
            else:
                self.acc_num += self.overlap(self.normalize(label_word),self.normalize(output_word))
        print("=========== last word accuracy is {} ===========".format(self.acc_num / self.total_num))
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
        self.Rouge1 = 0.0
        self.Rouge2 = 0.0
        self.RougeL = 0.0
        self.total_num = 0
        
    def update(self,hypothesis,targets):
        #batch_size = len(hypothesis)
        #for i in range(batch_size):
        scores = get_rouge_score(hypothesis,targets)
        self.Rouge1 += scores['rouge-1']['f']*100
        self.Rouge2 += scores['rouge-2']['f']*100
        self.RougeL += scores['rouge-l']['f']*100
        self.total_num += 1

        print("========== ROUGE_so_far ==========\n    {}     \n===================".format((self.Rouge1+self.Rouge2+self.RougeL)/float(3.0*self.total_num)))
        

class BLEU():
    def __init__(self,tokenizer=None):
        self.bleu = float(0.0)
        self.total_num = int(0)
        self.tokenizer = tokenizer
    def update(self,hypotheses,references):
        
        hypo_l = []
        ref_l = []
        if tokenizer is not None:
            for hypo,ref in zip(hypotheses,references):
                hypo_l.append(tokenizer.encode(hypo))
                ref_l.append(tokenizer.encode(ref))
        hypotheses = hypo_l
        references = ref_l
        #print(hypotheses)
        
        bleu_avg,res_list = sum_bleu(references,hypotheses)
        self.bleu += bleu_avg*100
        #print(res_list)
        #self.bleu += moses_multi_bleu(np.array(hypotheses),np.array(references))
        #print(type(self.bleu))
        self.total_num += 1

"""BLEU metric implementation.
"""

def moses_multi_bleu(hypotheses, references, lowercase=False):
    """Calculate the bleu score for hypotheses and references
    using the MOSES ulti-bleu.perl script.
    Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script
    Returns:
    The BLEU score as a float32 value.
    """

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    # Get MOSES multi-bleu script
    # try:
    #     multi_bleu_path, _ = urllib.request.urlretrieve(
    #         "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
    #         "master/scripts/generic/multi-bleu.perl")
    #     os.chmod(multi_bleu_path, 0o755)
    # except: #pylint: disable=W0702
    #     print("Unable to fetch multi-bleu.perl script, using local.")
    # metrics_dir = os.path.dirname(os.path.realpath(__file__))
    # bin_dir = os.path.abspath(os.path.join(metrics_dir, "..", "..", "bin"))
    # multi_bleu_path = os.path.join(bin_dir, "tools/multi-bleu.perl")
    multi_bleu_path = "./src/utils/multi-bleu.perl"

    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

    # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                print("multi-bleu.perl script returned non-zero exit code")
                print(error.output)
                bleu_score = np.float32(0.0)

    # Close temp files
    hypothesis_file.close()
    reference_file.close()
    return bleu_score

if __name__=="__main__":
    from tokenization import Tokenizer
    tokenizer = Tokenizer(vocab_file='./src/utils/pretrain-data/gpt2-vocab.json',
        merge_file='./src/utils/pretrain-data/gpt2-merges.txt')
    b = BLEU(tokenizer)
    b.update(['I am his fathers.','You are here.'],['I am his father.','I am here.'])
    print(b.bleu,type(b.bleu))
