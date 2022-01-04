#!/usr/bin/env python
# coding: utf-8

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def shap_analyze_model(model, observations):
    all_features = observations.columns.values
    med = observations.median().values.reshape((1,observations.shape[1]))
    explainer = shap.KernelExplainer(model.predict, med)
    shap_values = explainer.shap_values(observations)
    shap.summary_plot(shap_values, features=observations, feature_names=all_features)
    shap.summary_plot(shap_values, features=observations, feature_names=all_features, plot_type='bar')
    for fea in all_features:
        shap.dependence_plot(fea, shap_values, pd.DataFrame(observations, columns=all_features))
    
    shap_values_single = explainer.shap_values(observations.iloc[0,:])
    shap.force_plot(explainer.expected_value, shap_values_single, observations.iloc[0,:], matplotlib=True)

    return

def shap_forceplots_multiple(model, observations):
    med = observations.median().values.reshape((1,observations.shape[1]))
    explainer = shap.KernelExplainer(model.predict, med)
    shap_values = explainer.shap_values(observations)
    shap.force_plot(explainer.expected_value, shap_values, observations)
    plt.show()
    return
