"""Calculate ROUGE score."""
from typing import List
from rouge import Rouge

# H_PATH = "summaries.txt"
# R_PATH = "references.txt"


def get_rouge_score(hypothesis: List[str], target: List[str]):
    """
    Calculate ROUGE score.

    Args:
        hypothesis (List[str]): Inference result.
        target (List[str]): Reference.
    """

    if not hypothesis or not target:
        raise ValueError(f"`hypothesis` and `target` can not be None.")

    _rouge = Rouge()
    scores = _rouge.get_scores(hypothesis, target, avg=True)
    print(" | ROUGE Score:")
    print(f" | RG-1(F): {scores['rouge-1']['f'] * 100:8.2f}")
    print(f" | RG-2(F): {scores['rouge-2']['f'] * 100:8.2f}")
    print(f" | RG-L(F): {scores['rouge-l']['f'] * 100:8.2f}")

    # with open(H_PATH, "w") as f:
    #     f.writelines(edited_hyp)

    # with open(R_PATH, "w") as f:
    #     f.writelines(edited_ref)
    return scores
<<<<<<< Updated upstream
=======

if __name__ == "__main__":
    # s1 = [u'\xa0 Yes minister!']
    # s2 = ["Yes minister!"]
    s1 =   ['Father-of-two Paul Doyle moved family into £820,000 home in Altrincham .He bought it after seven-year jail term for cocaine and cannabis dealing .Admitted supplying  drugs,  money laundering and benefit fraud offences .Teenage son rode quad bikes and got Asbo for terrorising local children .']
    s2 =  [" A former member of the family has been jailed for 16 years after being caught in the act of selling drugs to criminals in his home town of Altrincham, Cheshire. The father of two has been dubbed the 'father-of-two' because he has lived a life of luxury and is a member of a drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug drug"]
    s2_p = [" A former member of the family has been jailed for 16 years after being caught in the act of selling drugs to criminals in his home town of Altrincham, Cheshire. The father of two has been dubbed the 'father-of-two' because he has lived a life of luxury and is a member of a drug"]
    print(s1,s2)
    print("----")
    print(get_rouge_score(s1,s2_p))
>>>>>>> Stashed changes
