import random
import pandas as pd
import numpy as np


class ConfusionMatrix:
    def __init__(self, predicted, actual):
        ## Saves the predictd and the acutal value
        self.predicted = predicted
        self.actual = actual

        ## Saves the confusion matrix
        self.matrix = self._create_matrix()

        ## Saves tp,fp, fn, and fp for each predictor
        self.hash_map = self.loop_basics()

        ### Saves advanced merics such as accuracy, sensitivty and such inside another dictioanry
        self.advanced_hash_map = self.loop_advanced()

    def _create_matrix(self):
        hash_map = {(i, j): 0 for i in set(self.actual) for j in set(self.predicted)}
        for pred_set_index, actual_index in zip(self.actual, self.predicted):
            hash_map[pred_set_index, actual_index] += 1
        df = pd.DataFrame(list(hash_map.items()))
        df[["row", "col"]] = pd.DataFrame(df[0].tolist(), index=df.index)
        df.rename(columns={1: "value"}, inplace=True)
        return df[["row", "col", "value"]]

    def loop_basics(self):
        hash_map = dict()
        for pred in set(self.predicted):
            hash_map[pred] = self._calculate_basics(pred)
        return hash_map

    def _calculate_basics(self, predictor):
        tp_m = self.matrix[
            (self.matrix["col"] == self.matrix["row"])
            & (self.matrix["row"] == predictor)
        ]
        tp = tp_m["value"].iloc[0]
        fn = sum(self.matrix[self.matrix["row"] == predictor]["value"])
        fp = sum(self.matrix[self.matrix["col"] == predictor]["value"])
        tn = self.matrix[["value"]].sum().iloc[0] - fn - fp + tp
        return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}

    def loop_advanced(self):
        hash_map = dict()
        for pred in set(self.predicted):
            hash_map[pred] = self._calculate_advanced(pred)
        return hash_map

    def _calculate_advanced(self, predictor):
        temp_hash = self.hash_map[predictor]
        ### Accuracy
        acc = (temp_hash["tp"] + temp_hash["tn"]) / sum(
            [temp_hash[key] for key in temp_hash.keys()]
        )

        ### Precision
        precision = temp_hash["tp"] / (temp_hash["tp"] + temp_hash["fp"])

        ### True Positive Rate
        tpr = temp_hash["tp"] / (temp_hash["tp"] + temp_hash["fn"])

        ### True Negative Rate
        tnr = temp_hash["tn"] / (temp_hash["tn"] + temp_hash["fp"])

        return {"accuracy": acc, "precision": precision, "tpr": tpr, "tnr": tnr}


predictions = np.random.randint(low=0, high=3, size=100)
y_train = predictions.copy()
random.shuffle(predictions)
cff = ConfusionMatrix(predictions, y_train)
