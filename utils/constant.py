import re


BEHAVIOR_PATTERNS = {
    "top_k_cascade": re.compile(r"^top_(\d+)_cascade$"),
    "neighbor_k": re.compile(r"^neighbor_(\d+)$"),
    "random_c": re.compile(r"random_(\d+)$"),
}

IPS_ESTIMATORS_TO_BEHAVIOR = {"SIPS": "standard", "IIPS": "independent", "RIPS": "cascade"}
SNIPS_ESTIMATORS_TO_BEHAVIOR = {"snSIPS": "standard", "snIIPS": "independent", "snRIPS": "cascade"}
VANILLA_ESTIMATORS_TO_BEHAVIOR = IPS_ESTIMATORS_TO_BEHAVIOR | SNIPS_ESTIMATORS_TO_BEHAVIOR

TRUE_ADAPTIVE_IPS_ESTIMATORS_TO_BEHAVIOR = {"AIPS (true)": "adaptive"}
TRUE_ADAPTIVE_SNIPS_ESTIMATORS_TO_BEHAVIOR = {"snAIPS (true)": "adaptive"}
TRUE_ADAPTIVE_ESTIMATORS_TO_BEHAVIOR = (
    TRUE_ADAPTIVE_IPS_ESTIMATORS_TO_BEHAVIOR | TRUE_ADAPTIVE_SNIPS_ESTIMATORS_TO_BEHAVIOR
)

ESTIMATED_ADAPTIVE_IPS_ESTIMATORS_TO_BEHAVIOR = {"AIPS (w/UBT)": "adaptive"}
ESTIMATED_ADAPTIVE_SNIPS_ESTIMATORS_TO_BEHAVIOR = {"snAIPS (w/UBT)": "adaptive"}
ADAPTIVE_ESTIMATORS_WITH_UBT_TO_BEHAVIOR = (
    ESTIMATED_ADAPTIVE_IPS_ESTIMATORS_TO_BEHAVIOR | ESTIMATED_ADAPTIVE_SNIPS_ESTIMATORS_TO_BEHAVIOR
)

VANILLA_IPS_ESTIMATORS_TO_BEHAVIOR = (
    IPS_ESTIMATORS_TO_BEHAVIOR
    | TRUE_ADAPTIVE_IPS_ESTIMATORS_TO_BEHAVIOR
    | ESTIMATED_ADAPTIVE_IPS_ESTIMATORS_TO_BEHAVIOR
)
VANILLA_SNIPS_ESTIMATORS_TO_BEHAVIOR = (
    SNIPS_ESTIMATORS_TO_BEHAVIOR
    | TRUE_ADAPTIVE_SNIPS_ESTIMATORS_TO_BEHAVIOR
    | ESTIMATED_ADAPTIVE_SNIPS_ESTIMATORS_TO_BEHAVIOR
)


TRUE_MARGINALIZED_ESTIMATORS_TO_BEHAVIOR = {"MSIPS": "standard", "MIIPS": "independent", "MRIPS": "cascade"}
MARGINALIZED_ESTIMATORS_WITH_SLOPE_TO_BEHAVIOR = {
    "MSIPS (w/SLOPE)": "standard",
    "MIIPS (w/SLOPE)": "independent",
    "MRIPS (w/SLOPE)": "cascade",
}

MARGINALIZED_ESTIMATORS_TO_BEHAVIOR = (
    TRUE_MARGINALIZED_ESTIMATORS_TO_BEHAVIOR | MARGINALIZED_ESTIMATORS_WITH_SLOPE_TO_BEHAVIOR
)

ESTIMATORS_WITH_TUNE = ADAPTIVE_ESTIMATORS_WITH_UBT_TO_BEHAVIOR | MARGINALIZED_ESTIMATORS_WITH_SLOPE_TO_BEHAVIOR

PELETTE = {
    # SIPS
    "SIPS": "tab:red",
    "snSIPS": "tab:red",
    "MSIPS": "tab:red",
    # IIPS
    "IIPS": "tab:blue",
    "snIIPS": "tab:blue",
    "MIIPS": "tab:blue",
    # RIPS
    "RIPS": "tab:purple",
    "snRIPS": "tab:purple",
    "MRIPS": "tab:purple",
    "MRIPS (w/SLOPE)": "tab:pink",
    # AIPS
    "AIPS (w/UBT)": "tab:green",
    "snAIPS (w/UBT)": "tab:green",
}
LINESTYLE = {
    # SIPS
    "SIPS": "",
    "snSIPS": "",
    "MSIPS": (7, 2),
    # IIPS
    "IIPS": "",
    "snIIPS": "",
    "MIIPS": (7, 2),
    # RIPS
    "RIPS": "",
    "snRIPS": "",
    "MRIPS": (7, 2),
    "MRIPS (w/SLOPE)": (7, 2),
    # AIPS
    "AIPS (w/UBT)": "",
    "snAIPS (w/UBT)": "",
}
TITLE_FONTSIZE = 25
LABEL_FONTSIZE = 25
TICK_FONTSIZE = 18
LINEWIDTH = 5
MARKERSIZE = 18

TQDM_FORMAT = "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
