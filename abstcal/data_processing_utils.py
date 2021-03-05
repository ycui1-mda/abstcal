
from datetime import timedelta
import pandas as pd
import numpy as np
from abstcal.calculator_data import CalculatorData
from abstcal.tlfb_data import TLFBData
from abstcal.visit_data import VisitData
from abstcal.calculator_error import InputArgumentError
import matplotlib.pyplot as plt


def from_wide_to_long(data_filepath, data_source_type, subject_col_name="id"):
    """
    Convert data from the wide format to the long format

    :param data_filepath: the filepath to the data

    :param data_source_type: the source type of the data: visit or TLFB

    :param subject_col_name: the column name of the subject id

    :return: DataFrame, the transformed data in the long format
    """
    wide_df = CalculatorData.read_data_from_path(data_filepath)
    if data_source_type == "tlfb":
        var_col_name, value_col_name = "date", "amount"
    else:
        var_col_name, value_col_name = "visit", "date"
    long_df = wide_df.melt(id_vars=subject_col_name, var_name=var_col_name, value_name=value_col_name)
    long_df.rename(columns={subject_col_name: "id"}, inplace=True)
    return long_df


def mask_dates(tlfb_filepath, bio_filepath, visit_filepath, reference):
    """
    Mask the dates in the visit and TLFB datasets using the anchor visit

    :param tlfb_filepath: the DataFrame object or filepath to the TLFB data (e.g., csv, xls, xlsx)

    :param visit_filepath: the DataFrame object or filepath to the visit data (e.g., csv, xls, xlsx)

    :param bio_filepath: the DataFrame object or filepath to the biochemical data (e.g., csv, xls, xlsx)

    :param reference: the anchor visit [it must be one member of visit list]
        or arbitrary date Union[str "mm/dd/yyyy", datetime.date] using which to mask the dates in these datasets

    :return: two DataFrames for visit and TLFB, respectively
    """
    visit_df = CalculatorData.read_data_from_path(visit_filepath)
    visit_df = VisitData._validated_data(visit_df, True)

    tlfb_df = CalculatorData.read_data_from_path(tlfb_filepath)
    tlfb_df = TLFBData._validated_data(tlfb_df, True)

    tlfb_dfs = list()
    for filepath in filter(lambda x: x is not None, (tlfb_filepath, bio_filepath)):
        tlfb_df = CalculatorData.read_data_from_path(filepath)
        tlfb_df = TLFBData._validated_data(tlfb_df, True)
        tlfb_dfs.append(tlfb_df)

    if reference in visit_df['visit'].unique():
        anchor_visit = reference
        anchor_dates = visit_df.loc[visit_df['visit'] == anchor_visit, ["id", "date"]]. \
            rename(columns={"date": "anchor_date"})

        tlfb_dfs_anchored = list()
        for tlfb_df in tlfb_dfs:
            tlfb_df_anchored = tlfb_df.merge(anchor_dates, on="id")
            tlfb_df_anchored['date'] = (tlfb_df_anchored['date'] - tlfb_df_anchored['anchor_date']).map(
                lambda x: x.days if pd.notnull(x) else np.nan
            )
            tlfb_dfs_anchored.append(tlfb_df_anchored.drop("anchor_date", axis=1))

        visit_df_anchored = visit_df.merge(anchor_dates, on="id")
        visit_df_anchored['date'] = (visit_df_anchored['date'] - visit_df_anchored['anchor_date']).map(
            lambda x: x.days if pd.notnull(x) else np.nan
        )
        return *tlfb_dfs_anchored, visit_df_anchored.drop("anchor_date", axis=1)
    else:
        try:
            reference_date = pd.to_datetime(reference)
        except TypeError:
            raise InputArgumentError("You're expecting to pass a date object or string as a reference.")
        else:
            for tlfb_df in tlfb_dfs:
                tlfb_df['date'] = (tlfb_df['date'] - reference_date).map(lambda x: x.days)
            visit_df['date'] = (visit_df['date'] - reference_date).map(lambda x: x.days)
        return *tlfb_df, visit_df


def add_additional_visit_dates(visit_filepath, visit_dates, use_raw_date):
    visit_df = CalculatorData.read_data_from_path(visit_filepath)
    visit_df = VisitData._validated_data(visit_df, True)
    visit_wide = visit_df.pivot(index="id", columns="visit", values="date").reset_index()
    for new_visit, reference_visit, days in visit_dates:
        visit_wide[new_visit] = visit_wide[reference_visit] + (timedelta(days=days) if use_raw_date else days)
    return visit_wide.melt(id_vars="id", var_name="visit", value_name="date")\
        .sort_values(by=['id', 'visit', 'date'], ignore_index=True)


show_figure = plt.show
