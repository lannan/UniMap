"""Data processing utilities."""

import json
import math
import torch
from texttable import Texttable

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def process_pair(path):
    """
    Reading a json file with a pair of graphs.
    :param path: Path to a JSON file.
    :return data: Dictionary with data.
    """
    data = json.load(open(path))
    return data


def calculate_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    :param prediction: Predicted log value of GED.
    :param target: Actual log value of GED.
    :return score: Log Squared Error.
    """
    # log_prediction = -math.log(prediction)
    # log_target = -math.log(target)
    # score = (log_prediction - log_target)**2
    score = torch.nn.functional.mse_loss(prediction, target)
    score = score.detach().numpy()
    return score
