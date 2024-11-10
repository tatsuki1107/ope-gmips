import pandas as pd
from pandas import DataFrame


def aggregate_simulation_results(
    simulation_result_list: list, policy_value: float, x_value: int, is_normalized: bool = True
) -> DataFrame:
    """Aggregate simulation results to calculate bias, variance, and squared error.

    Args:
        simulation_result_list: list
            list of simulation results.

        policy_value: float
            policy value.

        x_value: int
            x axis value.

        is_normalized: bool =True
            whether to normalize the squared error or not.

    Returns:
        DataFrame: aggregated simulation results included bias, variance, and squared error.
    """

    result_df = (
        DataFrame(DataFrame(simulation_result_list).stack())
        .reset_index(1)
        .rename(columns={"level_1": "estimator", 0: "value"})
    )
    result_df["x"] = x_value
    se = (result_df["value"] - policy_value) ** 2
    if is_normalized:
        se /= policy_value**2

    result_df["se"] = se
    result_df["bias"] = 0
    result_df["variance"] = 0

    expected_values = result_df.groupby("estimator").agg({"value": "mean"})["value"].to_dict()
    for estimator_name, expected_value in expected_values.items():
        row = result_df["estimator"] == estimator_name

        bias = (policy_value - expected_value) ** 2

        estimated_values = result_df[row]["value"].values
        variance = estimated_values.var()

        if is_normalized:
            bias /= policy_value**2
            variance /= policy_value**2

        result_df.loc[row, "bias"] = bias
        result_df.loc[row, "variance"] = variance

    return result_df


def reorder_result_df(result_df: DataFrame, estimator_names: list) -> DataFrame:
    """Reorder the result DataFrame.

    Args:
        result_df: DataFrame
            DataFrame of simulation results.

        estimator_names: list
            list of estimator names.

    Returns:
        DataFrame: reordered DataFrame.
    """

    grouped = result_df.groupby("estimator")

    df_reordered_list = []
    for estimator in estimator_names:
        if estimator in grouped.groups:
            df_reordered_list.append(grouped.get_group(estimator))

    df_reordered = pd.concat(df_reordered_list).reset_index(drop=True)

    return df_reordered
