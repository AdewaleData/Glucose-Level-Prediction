import numpy as np
import pandas as pd


def add_engineered_features(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    if {"Systolic_Peak", "Diastolic_Peak"}.issubset(d.columns):
        d["Pulse_Pressure_Peak"] = d["Systolic_Peak"] - d["Diastolic_Peak"]
        sys_safe = d["Systolic_Peak"].replace(0, np.nan)
        dia_safe = d["Diastolic_Peak"].replace(0, np.nan)
        d["Peak_Sys_Dia_Ratio"] = d["Systolic_Peak"] / dia_safe
        d["Augmentation_Index_Proxy"] = (d["Systolic_Peak"] - d["Diastolic_Peak"]) / sys_safe
    if {"PPG_Signal", "Heart_Rate"}.issubset(d.columns):
        hr = d["Heart_Rate"].replace(0, np.nan)
        d["PPG_per_HeartRate"] = d["PPG_Signal"] / hr
    if {"Pulse_Area", "PPG_Signal"}.issubset(d.columns):
        ppg = d["PPG_Signal"].replace(0, np.nan)
        d["PulseArea_per_PPG"] = d["Pulse_Area"] / ppg
    if "Pulse_Area" in d.columns:
        d["log1p_Pulse_Area"] = np.log1p(d["Pulse_Area"].clip(lower=0))
    for c in [x for x in d.columns if x not in raw.columns]:
        d[c] = d[c].replace([np.inf, -np.inf], np.nan)
    return d


BASE_INPUT_COLUMNS = [
    "Gender",
    "Age",
    "Height",
    "Weight",
    "BMI",
    "Peak_Diff",
    "index",
    "PPG_Signal",
    "Heart_Rate",
    "Pulse_Area",
    "Systolic_Peak",
    "Diastolic_Peak",
]
