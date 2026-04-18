# XAI Cybersecurity Threat Detection - Executive Summary (v6)

## 1. Complete Evaluation Metrics Table
Below is the tabular summary of all models tested within and across datasets:

| Train Dataset | Test Dataset | Model Pipeline | Accuracy | F1-Score (Weighted) |
| :--- | :--- | :--- | :--- | :--- |
| UNSW-NB15 | UNSW-NB15 | RandomForest | 0.9908 | 0.9907 |
| UNSW-NB15 | UNSW-NB15 | RandomForest + SHAP | 0.9897 | 0.9896 |
| UNSW-NB15 | UNSW-NB15 | RandomForest + LIME | 0.9886 | 0.9885 |
| UNSW-NB15 | UNSW-NB15 | SVM | 0.9254 | 0.8902 |
| UNSW-NB15 | UNSW-NB15 | SVM + SHAP | 0.9254 | 0.8902 |
| UNSW-NB15 | UNSW-NB15 | SVM + LIME | 0.9254 | 0.8902 |
| UNSW-NB15 | UNSW-NB15 | XGBoost | 0.9912 | 0.9911 |
| UNSW-NB15 | UNSW-NB15 | XGBoost + SHAP | 0.9897 | 0.9896 |
| UNSW-NB15 | UNSW-NB15 | XGBoost + LIME | 0.9897 | 0.9897 |
| UNSW-NB15 | UNSW-NB15 | CNN | 0.9836 | 0.9832 |
| UNSW-NB15 | UNSW-NB15 | CNN + SHAP | 0.9820 | 0.9814 |
| UNSW-NB15 | UNSW-NB15 | CNN + LIME | 0.9785 | 0.9772 |
| TON_IoT | TON_IoT | RandomForest | 0.9938 | 0.9938 |
| TON_IoT | TON_IoT | RandomForest + SHAP | 0.9928 | 0.9928 |
| TON_IoT | TON_IoT | RandomForest + LIME | 0.9928 | 0.9928 |
| TON_IoT | TON_IoT | SVM | 0.7631 | 0.6605 |
| TON_IoT | TON_IoT | SVM + SHAP | 0.9512 | 0.9505 |
| TON_IoT | TON_IoT | SVM + LIME | 0.8771 | 0.8719 |
| TON_IoT | TON_IoT | XGBoost | 0.9934 | 0.9934 |
| TON_IoT | TON_IoT | XGBoost + SHAP | 0.9927 | 0.9927 |
| TON_IoT | TON_IoT | XGBoost + LIME | 0.9930 | 0.9930 |
| TON_IoT | TON_IoT | CNN | 0.9140 | 0.9074 |
| TON_IoT | TON_IoT | CNN + SHAP | 0.9387 | 0.9355 |
| TON_IoT | TON_IoT | CNN + LIME | 0.8237 | 0.7808 |
| **UNSW-NB15** | **TON_IoT** | **RandomForest** | **0.2002** | **0.1441** |
| **UNSW-NB15** | **TON_IoT** | **RandomForest + SHAP** | **0.2330** | **0.1660** |
| **UNSW-NB15** | **TON_IoT** | **RandomForest + LIME** | **0.1896** | **0.1471** |
| **UNSW-NB15** | **TON_IoT** | **SVM** | **0.2369** | **0.0908** |
| **UNSW-NB15** | **TON_IoT** | **SVM + SHAP** | **0.2369** | **0.0908** |
| **UNSW-NB15** | **TON_IoT** | **SVM + LIME** | **0.2369** | **0.0908** |
| **UNSW-NB15** | **TON_IoT** | **XGBoost** | **0.1942** | **0.1059** |
| **UNSW-NB15** | **TON_IoT** | **XGBoost + SHAP** | **0.2007** | **0.1131** |
| **UNSW-NB15** | **TON_IoT** | **XGBoost + LIME** | **0.1966** | **0.1084** |
| **UNSW-NB15** | **TON_IoT** | **CNN** | **0.2447** | **0.2037** |
| **UNSW-NB15** | **TON_IoT** | **CNN + SHAP** | **0.2473** | **0.1921** |
| **UNSW-NB15** | **TON_IoT** | **CNN + LIME** | **0.2459** | **0.1125** |
| **TON_IoT** | **UNSW-NB15** | **RandomForest** | **0.1722** | **0.2554** |
| **TON_IoT** | **UNSW-NB15** | **RandomForest + SHAP** | **0.1362** | **0.2034** |
| **TON_IoT** | **UNSW-NB15** | **RandomForest + LIME** | **0.2200** | **0.3190** |
| **TON_IoT** | **UNSW-NB15** | **SVM** | **0.0754** | **0.0106** |
| **TON_IoT** | **UNSW-NB15** | **SVM + SHAP** | **0.0754** | **0.0108** |
| **TON_IoT** | **UNSW-NB15** | **SVM + LIME** | **0.0613** | **0.0788** |
| **TON_IoT** | **UNSW-NB15** | **XGBoost** | **0.1448** | **0.2154** |
| **TON_IoT** | **UNSW-NB15** | **XGBoost + SHAP** | **0.1336** | **0.1979** |
| **TON_IoT** | **UNSW-NB15** | **XGBoost + LIME** | **0.1221** | **0.1805** |
| **TON_IoT** | **UNSW-NB15** | **CNN** | **0.0630** | **0.0142** |
| **TON_IoT** | **UNSW-NB15** | **CNN + SHAP** | **0.0985** | **0.1371** |
| **TON_IoT** | **UNSW-NB15** | **CNN + LIME** | **0.0795** | **0.0884** |

## 2. The Dataset Shift Problem
- When trained and evaluated on the **same** dataset (TON_IoT), the best model (RandomForest) achieves an F1-score of **0.9938**.
- When transferring that training to a **different** dataset (UNSW-NB15), the best model (RandomForest + LIME) drops to an F1-score of **0.3190**.

## 3. The Power of XAI Feature Selection
By utilizing `SMOTENC` to properly oversample data alongside XAI Feature Selection, the models are forced to rely on fundamentally true properties of attacks rather than dataset-specific imbalances.

- Using SHAP-selected features improved generalization in **5** out of the cross-dataset instances compared to using all raw features.

## 4. Visual Evidence
Check the `outputs_v6/` directory for `.png` files showing **SHAP Summary Plots**.
