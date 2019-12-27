"""

 model_tree.py  (author: Anson Wong / git: ankonzoid)

 Given a classification/regression model, this code builds its model tree.

"""

import argparse
import os, pickle, csv
from src.ModelTree import ModelTree
from src.utils import load_csv_data, cross_validate

# ====================
# Parse arguments
# ====================

parser = argparse.ArgumentParser()
parser.add_argument("data",
                    help = "a csv file with the target variable labeled with the header 'y'.")
parser.add_argument("model_tree",
                    help = "a pickle file to dump the model tree to.")
parser.add_argument("predictions",
                    help = "name of csv to output predictions to")
parser.add_argument("--seed",
                    help = "a seed for the random permutation",
                    type = int,
                    default = None)
args = parser.parse_args()

def main():
    # ====================
    # Settings
    # ====================
    mode = "clf"  # "clf" / "regr"
    cross_validation = True  # cross-validate model tree?

    # ====================
    # Load data
    # ====================
    data_csv_data_filename = os.path.join(args.data)
    X, y, header = load_csv_data(data_csv_data_filename, mode = mode, verbose = True)

    # ====================
    # Load model tree
    # ====================
    print("Loading model tree from %s ..." % args.model_tree)
    model_tree = pickle.load(open(args.model_tree, "rb")))
    y_pred = model_tree.predict(X)
    explanations = model_tree.explain(X, header)
    loss = model_tree.loss(X, y, y_pred)
    print(" -> Test loss={:.6f}\n".format(loss))

    # ====================
    # Save model tree results
    # ====================
    print("Saving mode tree predictions to '{}'".format(args.predictions))
    with open(predictions_csv_filename, "w") as f:
        writer = csv.writer(f)
        field_names = ["x", "y", "y_pred", "explanation"]
        writer.writerow(field_names)
        for (x_i, y_i, y_pred_i, exp_i) in zip(X, y, y_pred, explanations):
            field_values = [x_i, y_i, y_pred_i, exp_i]
            writer.writerow(field_values)

# Driver
if __name__ == "__main__":
    main()