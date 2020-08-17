from typing import List
from rouge import Rouge

H_PATH = "summaries.txt"
R_PATH = "rederences.txt"

def rouge(hypothesis: List[str], target: List[str]):
    """
    Calculate ROUGE score.

    Args:
        hypothesis (List[str]): Inference result.
        target (List[str]): Reference.
    """
    def cut(s):
        idx = s.find("</s>")
        if idx != -1:
            s = s[:idx]
        return s
    if not hypothesis or not target:
        raise ValueError(f"`hypothesis` and `target` can not be None.")

    edited_hyp = []
    edited_ref = []

    for h, r in zip(hypothesis, target):
        h = cut(h).replace("<s>", "").strip()
        r = cut(r).replace("<s>", "").strip()
        edited_hyp.append(h + "\n")
        edited_ref.append(r + "\n")

    _rouge = Rouge()
    scores = _rouge.get_scores(edited_hyp, edited_ref, avg=True)
    print(" | ROUGE Score:")
    print(f" | R-1(F): {scores['rouge-1']['f'] * 100:8.2f}")
    print(f" | R-2(F): {scores['rouge-2']['f'] * 100:8.2f}")
    print(f" | R-L(F): {scores['rouge-l']['f'] * 100:8.2f}")

    with open(H_PATH, "w") as f:
        f.writelines(edited_hyp)
    with open(R_PATH, "w") as f:
        f.writelines(edited_ref)