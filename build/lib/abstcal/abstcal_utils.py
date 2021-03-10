"""
Provides utility functions used in the abstinence calculation
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abstcal.calculator_data import CalculatorData
from abstcal.calculator_error import InputArgumentError, FileExtensionError

# Use the following session to track data
# Reference: https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
import streamlit as st
from streamlit.report_thread import get_report_ctx


def read_data_from_path(filepath):
    """
    Read data from the specified path

    :param filepath: Union[str, Path, DataFrame], the path to the file to be read
        Supported file types: comma-separated, tab-separated, and Excel spreadsheet

    :return: a DataFrame
    """
    if isinstance(filepath, pd.DataFrame):
        return filepath

    path = Path(filepath)
    file_extension = path.suffix.lower()
    if file_extension == ".csv":
        df = pd.read_csv(filepath, infer_datetime_format=True)
    elif file_extension in (".xls", ".xlsx", ".xlsm", ".xlsb"):
        df = pd.read_excel(filepath, infer_datetime_format=True)
    elif file_extension == ".txt":
        df = pd.read_csv(filepath, sep='\t', infer_datetime_format=True)
    else:
        raise FileExtensionError(filepath)
    return df


def from_wide_to_long(data_filepath, data_source_type, subject_col_name="id"):
    """
    Convert data from the wide format to the long format

    :param data_filepath: the filepath to the data

    :param data_source_type: the source type of the data: visit or TLFB

    :param subject_col_name: the column name of the subject id

    :return: DataFrame, the transformed data in the long format
    """
    wide_df = read_data_from_path(data_filepath)
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
    visit_df = read_data_from_path(visit_filepath)

    tlfb_df = read_data_from_path(tlfb_filepath)

    tlfb_dfs = list()
    for filepath in filter(lambda x: x is not None, (tlfb_filepath, bio_filepath)):
        tlfb_df = read_data_from_path(filepath)
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


def write_data_to_path(df, filepath, index=False):
    """
    Write data to the specified path

    :param df: DataFrame
    :param filepath: Union[str, Path, None], the path to the file to be read
        Supported file types: comma-separated, tab-separated, and Excel spreadsheet, if None, no writing will be
        performed

    :param index: bool, whether the index of the DataFrame will be written to the output file

    :return: a DataFrame
    """
    if filepath is None:
        return

    path = Path(filepath)
    file_extension = path.suffix.lower()
    if file_extension == ".csv":
        pd.DataFrame.to_csv(df, path, index=index)
    elif file_extension in (".xls", ".xlsx", ".xlsm", ".xlsb"):
        pd.DataFrame.to_excel(df, path, index=index)
    elif file_extension == ".txt":
        pd.DataFrame.to_csv(df, path, sep='\t', index=index)
    else:
        raise FileExtensionError(filepath)


show_figure = plt.show


class SessionState(object):
    def __init__(self, **kwargs):
        """A new SessionState object.

        Parameters
        ----------
        **kwargs : any
            Default values for the session state.

        Example
        -------
        >>> session_state = SessionState(user_name='', favorite_color='black')
        >>> session_state.user_name = 'Mary'
        ''
        >>> session_state.favorite_color
        'black'

        """
        for key, val in kwargs.items():
            setattr(self, key, val)


@st.cache(allow_output_mutation=True)
def get_session(id, **kwargs):
    return SessionState(**kwargs)


def get_saved_session(**kwargs):
    """Gets a SessionState object for the current session.

    Creates a new object if necessary.

    Parameters
    ----------
    **kwargs : any
        Default values you want to add to the session state, if we're creating a
        new one.

    Example
    -------
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    ''
    >>> session_state.user_name = 'Mary'
    >>> session_state.favorite_color
    'black'

    Since you set user_name above, next time your script runs this will be the
    result:
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    'Mary'

    """
    ctx = get_report_ctx()
    id = ctx.session_id
    return get_session(id, **kwargs)


__all__ = ["get_saved_session"]
