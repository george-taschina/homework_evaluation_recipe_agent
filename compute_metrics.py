"""Q5: measure alignment between human labels and the judge.

    uv run python compute_metrics.py                      # original judge (results_judged_v1.json)
    uv run python compute_metrics.py results_judged.json  # improved judge

Positive class = "bad". Prints the confusion matrix, accuracy, precision,
recall, and every disagreement with both reasonings.
"""

import csv
import json
import sys


def main() -> None:
    judged_file = sys.argv[1] if len(sys.argv) > 1 else 'results_judged_v1.json'

    with open('human_labels.csv') as f:
        human = {row['question']: row['good_or_bad'] for row in csv.DictReader(f)}

    with open(judged_file) as f:
        results = json.load(f)

    tp = fp = fn = tn = 0
    disagreements = []

    for i, row in enumerate(results, 1):
        h = human[row['question']]
        j = row['judge_label']
        if h == 'bad' and j == 'bad':
            tp += 1
        elif h == 'good' and j == 'bad':
            fp += 1
            disagreements.append((i, row, h, j))
        elif h == 'bad' and j == 'good':
            fn += 1
            disagreements.append((i, row, h, j))
        else:
            tn += 1

    total = len(results)
    print(f"labels: {judged_file}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn} (total {total})")
    print(f"accuracy  = (TP+TN)/total = ({tp}+{tn})/{total} = {(tp + tn) / total:.4f}")
    print(f"precision = TP/(TP+FP) = {tp}/{tp + fp} = {tp / (tp + fp):.4f}")
    print(f"recall    = TP/(TP+FN) = {tp}/{tp + fn} = {tp / (tp + fn):.4f}")
    print(f"disagreements = {len(disagreements)}")

    for i, row, h, j in disagreements:
        print(f"\n#{i} [{row['question']}] human={h} judge={j}")
        print(f"  judge reasoning: {row['judge_reasoning']}")


if __name__ == '__main__':
    main()
