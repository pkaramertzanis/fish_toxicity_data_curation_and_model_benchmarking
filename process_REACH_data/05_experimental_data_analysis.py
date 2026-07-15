"""
Analyze the ratio between EC10 and NOEC values for chronic fish toxicity data from REACH.
"""

import logging

import logger

log = logger.setup_applevel_logger(
    logger_name="05_EC10_NOEC_ratio",
    file_name="logs/05_EC10_NOEC_ratio.log",
    level_stream=logging.INFO,
    level_file=logging.DEBUG,
)

import numpy as np
import pandas as pd

from analyse_predictions.regression import orthogonal_regression

# pandas display options
# do not fold dataframes
pd.set_option("expand_frame_repr", False)
# maximum number of columns
pd.set_option("display.max_columns", 50)
# maximum number of rows
pd.set_option("display.max_rows", 500)
# precision of float numbers
pd.set_option("display.precision", 3)
# maximum column width
pd.set_option("max_colwidth", 250)

# enable pandas copy-on-write
pd.options.mode.copy_on_write = True


# read datasets
experimental_data_acute = pd.read_pickle(
    r"data/fish_acute/processed/REACH_short_term_fish_measurement.pickle"
)
experimental_data_chronic = pd.read_pickle(
    r"data/fish_chronic/processed/REACH_long_term_fish_measurement.pickle"
)
smiles = pd.read_excel(r"data/structures/smiles.xlsx")

# number of substances with acute fish toxicity data for any scenario
msk = (
    experimental_data_acute["scenario"].notnull()
    & experimental_data_acute["exclusion reasons"].isnull()
)
tmp = (
    experimental_data_acute.loc[msk]
    .assign(
        has_effect=lambda df: df["effect concentration (mg/L, lower bound)"].notna()
    )
    .pivot_table(
        index="smiles",
        columns="scenario",
        values="has_effect",
        aggfunc="any",
        fill_value=False,
    )
    .astype(int)
)
log.info(
    f"Number of substances with acute fish toxicity data in any scenario: {tmp.shape[0]}"
)


# number of substances with chronic fish toxicity data for any scenario
msk = (
    experimental_data_chronic["scenario"].notnull()
    & experimental_data_chronic["exclusion reasons"].isnull()
)
tmp = (
    experimental_data_chronic.loc[msk]
    .assign(
        has_effect=lambda df: df["effect concentration (mg/L, lower bound)"].notna()
    )
    .pivot_table(
        index="smiles",
        columns="scenario",
        values="has_effect",
        aggfunc="any",
        fill_value=False,
    )
    .astype(int)
)
log.info(
    f"Number of substances with chronic fish toxicity data in any scenario: {tmp.shape[0]}"
)


# number of substances with acute fish toxicity data for scenarios a and b
msk = (
    experimental_data_acute["scenario"].str.contains(r"(?:a|b)$")
    & experimental_data_acute["exclusion reasons"].isnull()
)
tmp = (
    experimental_data_acute.loc[msk]
    .assign(
        has_effect=lambda df: df["effect concentration (mg/L, lower bound)"].notna()
    )
    .pivot_table(
        index="smiles",
        columns="scenario",
        values="has_effect",
        aggfunc="any",
        fill_value=False,
    )
    .astype(int)
)
log.info(
    f"Number of substances with acute fish toxicity data for scenarios a and b: {tmp.shape[0]}"
)
log.info(f"Breakdown by scenario:\n{tmp.sum().sort_values(ascending=False).to_json()}")

# number of substances with chronic fish toxicity data for scenarios a and b
msk = (
    experimental_data_chronic["scenario"].str.contains(r"(?:a|b)$")
    & experimental_data_chronic["exclusion reasons"].isnull()
)
tmp = (
    experimental_data_chronic.loc[msk]
    .assign(
        has_effect=lambda df: df["effect concentration (mg/L, lower bound)"].notna()
    )
    .pivot_table(
        index="smiles",
        columns="scenario",
        values="has_effect",
        aggfunc="any",
        fill_value=False,
    )
    .astype(int)
)
log.info(
    f"Number of substances with chronic fish toxicity data for scenarios a and b: {tmp.shape[0]}"
)
log.info(f"Breakdown by scenario:\n{tmp.sum().sort_values(ascending=False).to_json()}")

# number of standardised smiles with AFT_1a and 1b used for the quantitative assessment
msk = (
    experimental_data_acute["scenario"].str.contains(r"AFT_1(:?a|b)$")
    & experimental_data_acute["exclusion reasons"].isnull()
)
tmp = (
    experimental_data_acute.loc[msk]
    .merge(
        smiles[["smiles", "smiles (standardised)"]],
        how="inner",
        left_on="smiles",
        right_on="smiles",
    )
    .assign(
        has_effect=lambda df: df["effect concentration (mg/L, lower bound)"].notna()
    )
    .pivot_table(
        index="smiles (standardised)",
        columns="scenario",
        values="has_effect",
        aggfunc="any",
        fill_value=False,
    )
    .astype(int)
)
log.info(
    f"Number of standardised smiles with AFT_1a or AFT_1b scenarios: {tmp.shape[0]}"
)

# number of standardised smiles with FLS_1a and 1b used for the quantitative assessment
msk = (
    experimental_data_chronic["scenario"].str.contains(r"FLS_1(:?a|b)$")
    & experimental_data_chronic["exclusion reasons"].isnull()
)
tmp = (
    experimental_data_chronic.loc[msk]
    .merge(
        smiles[["smiles", "smiles (standardised)"]],
        how="inner",
        left_on="smiles",
        right_on="smiles",
    )
    .assign(
        has_effect=lambda df: df["effect concentration (mg/L, lower bound)"].notna()
    )
    .pivot_table(
        index="smiles (standardised)",
        columns="scenario",
        values="has_effect",
        aggfunc="any",
        fill_value=False,
    )
    .astype(int)
)
log.info(
    f"Number of standardised smiles with FLS_1a or FLS_1b scenarios: {tmp.shape[0]}"
)

# most common species in acute fish toxicity data used for the quantitative assessment
msk = (
    experimental_data_acute["scenario"].str.contains(r"AFT_1(:?a|b)$")
    & experimental_data_acute["exclusion reasons"].isnull()
)
tmp = (
    experimental_data_acute.loc[msk]
    .merge(
        smiles[["smiles", "smiles (standardised)"]],
        how="inner",
        left_on="smiles",
        right_on="smiles",
    )
    .groupby("test organisms")["smiles (standardised)"]
    .nunique()
    .sort_values(ascending=False)
    .to_frame()
    .reset_index()
    .rename(columns={"smiles (standardised)": "number of studies"})
)
log.info(
    f"Most common species in acute fish toxicity data used for the quantitative assessment in terms of number of standardised structures:\n{tmp.head(4).to_json(orient='records')}"
)

# most common species in chronic fish toxicity data used for the quantitative assessment
msk = (
    experimental_data_chronic["scenario"].str.contains(r"FLS_1(:?a|b)$")
    & experimental_data_chronic["exclusion reasons"].isnull()
)
tmp = (
    experimental_data_chronic.loc[msk]
    .merge(
        smiles[["smiles", "smiles (standardised)"]],
        how="inner",
        left_on="smiles",
        right_on="smiles",
    )
    .groupby("test organisms")["smiles (standardised)"]
    .nunique()
    .sort_values(ascending=False)
    .to_frame()
    .reset_index()
    .rename(columns={"smiles (standardised)": "number of standardised structures"})
)
log.info(
    f"Most common species in chronic fish toxicity data used for the quantitative assessment in terms of number of standardised structures:\n{tmp.head(4).to_json(orient='records')}"
)


# most common effect levels in chronic fish toxicity data used for the quantitative assessment
msk = (
    experimental_data_chronic["scenario"].str.contains(r"FLS_1(:?a|b)$")
    & experimental_data_chronic["exclusion reasons"].isnull()
)
tmp = (
    experimental_data_chronic.loc[msk]
    .merge(
        smiles[["smiles", "smiles (standardised)"]],
        how="inner",
        left_on="smiles",
        right_on="smiles",
    )
    # keep the lowest effect level per standardised structure
    .groupby("smiles (standardised)")
    .apply(
        lambda df: df[
            [
                "smiles (standardised)",
                "effect concentration (mol/L, lower bound)",
                "basis for effect (standardised)",
            ]
        ].loc[df["effect concentration (mol/L, lower bound)"].idxmin()]
    )
    .reset_index(drop=True)
    .assign(
        **{
            "basis for effect (standardised)": lambda df: df[
                "basis for effect (standardised)"
            ].apply(lambda s: ",".join(sorted(s)))
        }
    )
    .groupby("basis for effect (standardised)")["smiles (standardised)"]
    .nunique()
    .sort_values(ascending=False)
    .to_frame()
    .reset_index()
    .rename(columns={"smiles (standardised)": "number of standardised structures"})
)
log.info(
    f"Most common effect levels in chronic fish toxicity data used for the quantitative assessment in terms of number of standardised structures:\n{tmp.head(4).to_json(orient='records')}"
)


# most common endpoint in chronic fish toxicity data used for the quantitative assessment
msk = (
    experimental_data_chronic["scenario"].str.contains(r"FLS_1(:?a|b)$")
    & experimental_data_chronic["exclusion reasons"].isnull()
)
tmp = (
    experimental_data_chronic.loc[msk]
    .merge(
        smiles[["smiles", "smiles (standardised)"]],
        how="inner",
        left_on="smiles",
        right_on="smiles",
    )
    # keep the lowest effect level per standardised structure
    .groupby("smiles (standardised)")
    .apply(
        lambda df: df[
            [
                "smiles (standardised)",
                "effect concentration (mol/L, lower bound)",
                "endpoint (standardised)",
            ]
        ].loc[df["effect concentration (mol/L, lower bound)"].idxmin()]
    )
    .reset_index(drop=True)
    .assign(
        **{
            "endpoint (standardised)": lambda df: df["endpoint (standardised)"].apply(
                lambda s: ",".join(sorted(s))
            )
        }
    )
    .groupby("endpoint (standardised)")["smiles (standardised)"]
    .nunique()
    .sort_values(ascending=False)
    .to_frame()
    .reset_index()
    .rename(columns={"smiles (standardised)": "number of standardised structures"})
)
log.info(
    f"Most common endpoints in chronic fish toxicity data used for the quantitative assessment in terms of number of standardised structures:\n{tmp.head(4).to_json(orient='records')}"
)


# analyse the slope and intercept of the Lee et al. AFT and QSAR Toolbox AW LC50 data
lee_data = pd.read_excel(r"data/other/AFT_vs_Toolbox_Lee_et_al.xlsx")
x = np.log10(lee_data["AFT LC50 (mg/L)"])
y = np.log10(lee_data["QSAR toolbox AW LC50 (mg/L)"])
orthogonal_regression_scipy(x, y)
# delete after this point


# examine the EC10/NOEC ratio for growth endpoints
EC10_NOEC_growth_ratio = (
    experimental_data_chronic.loc[
        experimental_data_chronic["basis for effect (standardised)"].apply("".join)
        == "growth"
    ]
    .loc[
        experimental_data_chronic["endpoint (standardised)"]
        .apply("".join)
        .isin(["NOEC", "EC10"])
    ]
    .assign(
        **{
            "endpoint (standardised)": lambda df: df["endpoint (standardised)"].apply(
                "".join
            )
        }
    )
    .pivot_table(
        index="UUID (endpoint study record) (echachem)",
        columns="endpoint (standardised)",
        values="effect concentration (mg/L, lower bound)",
        aggfunc="min",
    )
    .dropna(how="any", axis="index")
    .assign(**{"EC10/NOEC": lambda df: df["EC10"] / df["NOEC"]})
    .dropna(subset="EC10/NOEC")
)
pd.qcut(
    EC10_NOEC_growth_ratio["EC10/NOEC"], q=4
).value_counts().sort_index().to_frame().reset_index().rename(
    columns={"index": "EC10/NOEC ratio bin", "EC10/NOEC": "count"}
)

# examine the lowest endpoint per study and basis for effect
lowest_effect_level_per_study = (
    experimental_data_chronic.assign(
        **{
            "endpoint (standardised)": lambda df: df["endpoint (standardised)"].apply(
                lambda s: ",".join(sorted(s))
            )
        }
    )
    .assign(
        **{
            "basis for effect (standardised)": lambda df: df[
                "basis for effect (standardised)"
            ].apply(lambda s: ",".join(sorted(s)))
        }
    )
    .groupby(["UUID (endpoint study record) (echachem)"])[
        [
            "basis for effect (standardised)",
            "endpoint (standardised)",
            "effect concentration (mg/L, lower bound)",
        ]
    ]
    .apply(
        lambda df: df.sort_values(by="endpoint (standardised)", ascending=False).loc[
            df["effect concentration (mg/L, lower bound)"].idxmin()
        ]
    )  # sorting ensures that NOEC is preferred over EC10 in case of ties
    .reset_index()
)
lowest_effect_level_per_study.groupby(
    ["basis for effect (standardised)", "endpoint (standardised)"]
)["UUID (endpoint study record) (echachem)"].nunique().sort_values(
    ascending=False
).to_frame().reset_index().rename(
    columns={"UUID (endpoint study record) (echachem)": "number of studies"}
)
