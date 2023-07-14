import lime
import lime.lime_tabular
import numpy as np

class LimeExplainer:
    def __init__(self, training_data, feature_names, class_names):
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=True
        )
        
    def explain_instance(self, instance, predict_fn, num_features=5):
        exp = self.explainer.explain_instance(
            instance, 
            predict_fn, 
            num_features=num_features
        )
        return exp.as_list()
