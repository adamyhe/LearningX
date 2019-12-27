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
                    help = "name of permuted database (base output file that gets split)")
parser.add_argument("--model_tree_visual",
                    help = "a png file to save a visual of the model tree (none if not indicated)",
                    type = str,
                    default = None)
parser.add_argument("--y_header",
                    help = "the column header of the target variable in the csv dataset",
                    type = str,
                    default = "y")
parser.add_argument("--seed",
                    help = "a seed for the random permutation",
                    type = int,
                    default = None)
parser.add_argument("--cross_validate",
                    help = "should we cross validate?",
                    type = bool,
                    default = True)
args = parser.parse_args()

def main():
    # ====================
    # Settings
    # ====================
    mode = "regr"  # "clf" / "regr"
    #save_model_tree = True  # save model tree?
    #save_model_tree_predictions = True  # save model tree predictions/explanations?

    # ====================
    # Load data
    # ====================
    X, y, header = load_csv_data(args.data, 
                                 mode = mode, y_header = args.y_header, verbose = True)

    # *********************************************
    #
    # Insert your models here!
    #
    # All models must have the following class instantiations:
    #
    #   fit(X, y)
    #   predict(X)
    #   loss(X, y, y_pred)
    #
    # Below are some ready-for-use regression models:
    #
    #   mean regressor  (models/mean_regr.py)
    #   linear regressor  (models/linear_regr.py)
    #   logistic regressor  (lmodels/ogistic_regr.py)
    #   support vector machine regressor  (models/svm_regr.py)
    #   decision tree regressor (models/DT_sklearn_regr.py)
    #   neural network regressor (models/DT_sklearn_regr.py)
    #
    # as well as some classification models:
    #
    #   modal classifier (models/modal_clf.py)
    #   decision tree classifier (models/DT_sklearn_clf.py)
    #
    # *********************************************
    from models.mean_regr import mean_regr
    from models.linear_regr import linear_regr
    from models.logistic_regr import logistic_regr
    from models.svm_regr import svm_regr
    from models.DT_sklearn_regr import DT_sklearn_regr

    from models.modal_clf import modal_clf
    from models.DT_sklearn_clf import DT_sklearn_clf

    # Choose model
    model = logistic_regr

    # Build model tree
    model_tree = ModelTree(model, max_depth = 4, min_samples_leaf = 10,
                           search_type = "greedy", n_search_grid = 100)

    # ====================
    # Train model tree
    # ====================
    print("Training model tree with '{}'...".format(model.__class__.__name__))
    model_tree.fit(X, y, verbose = True)
    y_pred = model_tree.predict(X)
    explanations = model_tree.explain(X, header)
    loss = model_tree.loss(X, y, y_pred)
    print(" -> loss_train={:.6f}\n".format(loss))
    if args.model_tree_visual != None:    
        model_tree.export_graphviz(args.model_tree_visual, header,
                                   export_png = True, export_pdf = False)

    # ====================
    # Save model tree results
    # ====================
    print("Saving model tree to '{}'...".format(args.model_tree))
    pickle.dump(model, open(args.model_tree, 'wb'))

    # ====================
    # Cross-validate model tree
    # ====================
    if args.cross_validate:
        cross_validate(model_tree, X, y, kfold = 5, seed = args.seed)

# Driver
if __name__ == "__main__":
    main()