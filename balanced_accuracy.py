import argparse

import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def bacc_to_grade_mapping(bacc):
    def print_grade(grade):
        print(grade)

    if bacc > 0.875:
        return print_grade(grade=1)
    elif bacc > 0.825:
        return print_grade(grade=2)
    elif bacc > 0.775:
        return print_grade(grade=3)
    elif bacc > 0.750:
        return print_grade(grade=4)
    else:
        return print_grade(grade=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()
    submission = pd.read_csv(args.submission)
    target = None
    if args.target is not None:
        target = pd.read_csv(args.target)
    merged = pd.merge(target, submission, on="file_id", how='left', suffixes=('_true', '_pred'))
    encode_class_labels = ["A549", "CACO-2", "HEK 293", "HeLa", "MCF7", "PC-3", "RT4", "U-2 OS", "U-251 MG"]
    merged = merged.replace(encode_class_labels, list(range(9)))
    bacc = balanced_accuracy_score(merged.cell_line_true, merged.cell_line_pred)
    bacc_to_grade_mapping(bacc)