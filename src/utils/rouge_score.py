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

# if __name__ == "__main__":
#     s1 = ["This is a cat","809890"]
#     s2 = ["This is a dog.","I am Iron Man!"]
#     print("----")
#     print(get_rouge_score(s1,s2)['rouge-1']['r']*100)