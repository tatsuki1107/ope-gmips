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

ESTIMATED_ADAPTIVE_IPS_ESTIMATORS_TO_BEHAVIOR = {r"AIPS-$\hat{c}$(UBT)": "adaptive"}
ESTIMATED_ADAPTIVE_SNIPS_ESTIMATORS_TO_BEHAVIOR = {r"snAIPS-$\hat{c}$(UBT)": "adaptive"}
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
MARGINALIZED_ESTIMATORS_HAT_TO_BEHAVIOR = {
    r"MSIPS-$\hat{w}$": "standard",
    r"MIIPS-$\hat{w}$": "independent",
    r"MRIPS-$\hat{w}$": "cascade",
}
MARGINALIZED_ESTIMATORS_WITH_SLOPE_TO_BEHAVIOR = {
    "MSIPS (SLOPE)": "standard",
    "MIIPS (SLOPE)": "independent",
    "MRIPS (SLOPE)": "cascade",
    r"MSIPS-$\hat{w}$ (SLOPE)": "standard",
    r"MIIPS-$\hat{w}$ (SLOPE)": "independent",
    r"MRIPS-$\hat{w}$ (SLOPE)": "cascade",
}

MARGINALIZED_ESTIMATORS_TO_BEHAVIOR = (
    TRUE_MARGINALIZED_ESTIMATORS_TO_BEHAVIOR
    | MARGINALIZED_ESTIMATORS_HAT_TO_BEHAVIOR
    | MARGINALIZED_ESTIMATORS_WITH_SLOPE_TO_BEHAVIOR
)


PELETTE = {
    # SIPS
    "SIPS": "tab:red",
    "snSIPS": "tab:red",
    "MSIPS": "tab:red",
    r"MSIPS-$\hat{w}$": "tab:red",
    # IIPS
    "IIPS": "tab:blue",
    "snIIPS": "tab:blue",
    "MIIPS": "tab:blue",
    r"MIIPS-$\hat{w}$": "tab:blue",
    # RIPS
    "RIPS": "tab:purple",
    "snRIPS": "tab:purple",
    "MRIPS": "tab:purple",
    r"MRIPS-$\hat{w}$": "tab:purple",
    # AIPS
    "AIPS (true)": "tab:green",
    "snAIPS (true)": "tab:green",
    r"AIPS-$\hat{c}$": "tab:green",
    r"snAIPS-$\hat{c}$": "tab:green",
}
LINESTYLE = {
    # SIPS
    "SIPS": "",
    "snSIPS": "",
    "MSIPS": (7, 2),
    r"MSIPS-$\hat{w}$": (1, 1),
    # IIPS
    "IIPS": "",
    "snIIPS": "",
    "MIIPS": (7, 2),
    r"MIIPS-$\hat{w}$": (1, 1),
    # RIPS
    "RIPS": "",
    "snRIPS": "",
    "MRIPS": (7, 2),
    r"MRIPS-$\hat{w}$": (1, 1),
    # AIPS
    "AIPS (true)": "",
    "snAIPS (true)": "",
    r"AIPS-$\hat{c}$": "",
    r"snAIPS-$\hat{c}$": "",
}
TITLE_FONTSIZE = 25
LABEL_FONTSIZE = 20
LINEWIDTH = 5
MARKERSIZE = 20

TQDM_FORMAT = "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
