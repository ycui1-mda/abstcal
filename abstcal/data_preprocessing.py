
from datetime import timedelta
import pandas as pd
import numpy as np
from abstcal.calculator_data import CalculatorData
from abstcal.tlfb_data import TLFBData
from abstcal.visit_data import VisitData
from abstcal.calculator_error import InputArgumentError, _show_warning


def from_wide_to_long(data_filepath, data_source_type, subject_col_name="id"):
    """
    Convert data from the wide format to the long format

    :param data_filepath: the filepath to the data

    :param data_source_type: the source type of the data: visit or TLFB

    :param subject_col_name: the column name of the subject id

    :return: DataFrame, the transformed data in the long format
    """
    wide_df = data_filepath if isinstance(data_filepath, pd.DataFrame) \
        else CalculatorData.read_data_from_path(data_filepath)
    if data_source_type == "tlfb":
        var_col_name, value_col_name = "date", "amount"
    else:
        var_col_name, value_col_name = "visit", "date"
    long_df = wide_df.melt(id_vars=subject_col_name, var_name=var_col_name, value_name=value_col_name)
    long_df.rename(columns={subject_col_name: "id"}, inplace=True)
    return long_df


def mask_dates(tlfb_filepath, visit_filepath, reference, using_anchor_visit=True):
    """
    Mask the dates in the visit and TLFB datasets using the anchor visit

    :param tlfb_filepath: the DataFrame object or filepath to the TLFB data (e.g., csv, xls, xlsx)

    :param visit_filepath: the DataFrame object or filepath to the visit data (e.g., csv, xls, xlsx)

    :param reference: the anchor visit or arbitrary date using which to mask the dates in these two datasets

    :param using_anchor_visit: using the anchor visit (each subject uses his/her own date for the anchor visit
        as a reference) or a shared reference date for masking

    :return: two DataFrames for visit and TLFB, respectively
    """
    visit_df = visit_filepath if isinstance(visit_filepath, pd.DataFrame) \
        else CalculatorData.read_data_from_path(visit_filepath)
    visit_df = VisitData._validated_data(visit_df, True)

    tlfb_df = tlfb_filepath if isinstance(tlfb_filepath, pd.DataFrame) \
        else CalculatorData.read_data_from_path(tlfb_filepath)
    tlfb_df = TLFBData._validated_data(tlfb_df, True)

    if using_anchor_visit:
        anchor_visit = reference
        anchor_dates = visit_df.loc[visit_df['visit'] == anchor_visit, ["id", "date"]]. \
            rename(columns={"date": "anchor_date"})

        tlfb_df_anchored = tlfb_df.merge(anchor_dates, on="id")
        tlfb_df_anchored['date'] = (tlfb_df_anchored['date'] - tlfb_df_anchored['anchor_date']).map(
            lambda x: x.days if pd.notnull(x) else np.nan
        )

        visit_df_anchored = visit_df.merge(anchor_dates, on="id")
        visit_df_anchored['date'] = (visit_df_anchored['date'] - visit_df_anchored['anchor_date']).map(
            lambda x: x.days if pd.notnull(x) else np.nan
        )
        return tlfb_df_anchored.drop("anchor_date", axis=1), visit_df_anchored.drop("anchor_date", axis=1)
    else:
        tlfb_df['date'] = (tlfb_df['date'] - reference).map(lambda x: x.days)
        visit_df['date'] = (visit_df['date'] - reference).map(lambda x: x.days)
        return tlfb_df, visit_df


def add_additional_visit_dates(visit_filepath, visit_dates, use_raw_date):
    visit_df = visit_filepath if isinstance(visit_filepath, pd.DataFrame) \
        else CalculatorData.read_data_from_path(visit_filepath)
    visit_df = VisitData._validated_data(visit_df, True)
    visit_wide = visit_df.pivot(index="id", columns="visit", values="date").reset_index()
    for new_visit, reference_visit, days in visit_dates:
        visit_wide[new_visit] = visit_wide[reference_visit] + (timedelta(days=days) if use_raw_date else days)
    return visit_wide.melt(id_vars="id", var_name="visit", value_name="date")\
        .sort_values(by=['id', 'visit', 'date'], ignore_index=True)
