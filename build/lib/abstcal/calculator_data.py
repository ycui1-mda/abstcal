"""
CalculatorData
---------
A data model for creating subclasses to process timeline followback and visit data in the abstinence calculation
"""

from enum import Enum
import pandas as pd
from abstcal.calculator_error import InputArgumentError, FileFormatError


class DataImputationCode(Enum):
    RAW = 0
    IMPUTED = 1
    OVERRIDDEN = 2


class CalculatorData:
    data: pd.DataFrame
    duplicates: pd.DataFrame = None
    subject_ids: set
    _index_keys: list
    _value_key: str
    use_raw_date: bool

    @property
    def col_names(self):
        return [*self._index_keys, self._value_key]

    def check_duplicates(self, duplicate_kept="mean"):
        """
        Check duplicate records, which are identified using id and date for the TLFB data and using id and visit for
        the visit data

        :param duplicate_kept: Union["min", "max", "mean", False, None], specify the action with duplicate records
            "max": default option, keep the records with the maximal value
            "min": keep the records with the minimal value
            "mean": drop duplicate records and replace them with mean values
            False: remove all duplicate records
            None: no actions on duplicate records
        :return: a DataFrame of duplicate records
        """
        if duplicate_kept not in ("min", "max", "mean", False, None):
            raise InputArgumentError("Please specify how you deal with the duplicates, min, max, mean, False, or None.")

        duplicated_indices = self.data.duplicated(self._index_keys, keep=False)
        duplicates = self.data.loc[duplicated_indices, :].sort_values(by=self._index_keys, ignore_index=True)

        if duplicate_kept is None or duplicates.empty:
            return 0

        if duplicate_kept in ("min", "max"):
            self.data.sort_values(by=[*self._index_keys, self._value_key], inplace=True)
            self.data.drop_duplicates(self._index_keys, keep="first" if duplicate_kept == "min" else "last",
                                      inplace=True, ignore_index=True)
        elif duplicate_kept == "mean":
            df = self.data.drop_duplicates(self._index_keys, keep=False)
            duplicates_means = duplicates.groupby(self._index_keys)[self._value_key].mean().reset_index()
            self.data = pd.concat([df, duplicates_means], axis=0)
            self.data.sort_values(by=self._index_keys, inplace=True, ignore_index=True)
        else:
            self.data.drop_duplicates(self._index_keys, keep=False, inplace=True)

        return duplicates

    def drop_na_records(self):
        """
        Drop the records with any na values

        :return: int, the number of NA records that are dropped
        """
        missing_records = self.data.isnull().any(axis=1).sum()
        if missing_records:
            self.data.dropna(axis=0, inplace=True)
            self.data.reset_index(drop=True)
        return missing_records

    # @staticmethod
    # def read_data_from_path(filepath):
    #     """
    #     Read data from the specified path
    #
    #     :param filepath: Union[str, Path, DataFrame], the path to the file to be read
    #         Supported file types: comma-separated, tab-separated, and Excel spreadsheet
    #
    #     :return: a DataFrame
    #     """
    #     if isinstance(filepath, pd.DataFrame):
    #         return filepath
    #
    #     path = Path(filepath)
    #     file_extension = path.suffix.lower()
    #     if file_extension == ".csv":
    #         df = pd.read_csv(filepath, infer_datetime_format=True)
    #     elif file_extension in (".xls", ".xlsx", ".xlsm", ".xlsb"):
    #         df = pd.read_excel(filepath, infer_datetime_format=True)
    #     elif file_extension == ".txt":
    #         df = pd.read_csv(filepath, sep='\t', infer_datetime_format=True)
    #     else:
    #         raise FileExtensionError(filepath)
    #     return df

    def validate_data(self, df):
        needed_cols_unordered = set(self.col_names)
        current_cols_unordered = set(df.columns)
        if needed_cols_unordered.issubset(current_cols_unordered):
            if self.use_raw_date:
                df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
            if self._value_key == 'amount':
                df['amount'] = df['amount'].astype(float)
            return df.loc[:, self.col_names]
        else:
            raise FileFormatError(f'Required Columns of {self.__class__.__name__}: {self.col_names}')

    # @staticmethod
    # def from_wide_to_long(data_filepath, data_source_type, subject_col_name="id"):
    #     """
    #     Convert data from the wide format to the long format
    #
    #     :param data_filepath: the filepath to the data
    #
    #     :param data_source_type: the source type of the data: visit or TLFB
    #
    #     :param subject_col_name: the column name of the subject id
    #
    #     :return: DataFrame, the transformed data in the long format
    #     """
    #     wide_df = CalculatorData.read_data_from_path(data_filepath)
    #     if data_source_type == "tlfb":
    #         var_col_name, value_col_name = "date", "amount"
    #     else:
    #         var_col_name, value_col_name = "visit", "date"
    #     long_df = wide_df.melt(id_vars=subject_col_name, var_name=var_col_name, value_name=value_col_name)
    #     long_df.rename(columns={subject_col_name: "id"}, inplace=True)
    #     return long_df

    # @staticmethod
    # def mask_dates(tlfb_filepath, bio_filepath, visit_filepath, reference):
    #     """
    #     Mask the dates in the visit and TLFB datasets using the anchor visit
    #
    #     :param tlfb_filepath: the DataFrame object or filepath to the TLFB data (e.g., csv, xls, xlsx)
    #
    #     :param visit_filepath: the DataFrame object or filepath to the visit data (e.g., csv, xls, xlsx)
    #
    #     :param bio_filepath: the DataFrame object or filepath to the biochemical data (e.g., csv, xls, xlsx)
    #
    #     :param reference: the anchor visit [it must be one member of visit list]
    #         or arbitrary date Union[str "mm/dd/yyyy", datetime.date] using which to mask the dates in these datasets
    #
    #     :return: two DataFrames for visit and TLFB, respectively
    #     """
    #     visit_df = CalculatorData.read_data_from_path(visit_filepath)
    #
    #     tlfb_df = CalculatorData.read_data_from_path(tlfb_filepath)
    #
    #     tlfb_dfs = list()
    #     for filepath in filter(lambda x: x is not None, (tlfb_filepath, bio_filepath)):
    #         tlfb_df = CalculatorData.read_data_from_path(filepath)
    #         tlfb_dfs.append(tlfb_df)
    #
    #     if reference in visit_df['visit'].unique():
    #         anchor_visit = reference
    #         anchor_dates = visit_df.loc[visit_df['visit'] == anchor_visit, ["id", "date"]]. \
    #             rename(columns={"date": "anchor_date"})
    #
    #         tlfb_dfs_anchored = list()
    #         for tlfb_df in tlfb_dfs:
    #             tlfb_df_anchored = tlfb_df.merge(anchor_dates, on="id")
    #             tlfb_df_anchored['date'] = (tlfb_df_anchored['date'] - tlfb_df_anchored['anchor_date']).map(
    #                 lambda x: x.days if pd.notnull(x) else np.nan
    #             )
    #             tlfb_dfs_anchored.append(tlfb_df_anchored.drop("anchor_date", axis=1))
    #
    #         visit_df_anchored = visit_df.merge(anchor_dates, on="id")
    #         visit_df_anchored['date'] = (visit_df_anchored['date'] - visit_df_anchored['anchor_date']).map(
    #             lambda x: x.days if pd.notnull(x) else np.nan
    #         )
    #         return *tlfb_dfs_anchored, visit_df_anchored.drop("anchor_date", axis=1)
    #     else:
    #         try:
    #             reference_date = pd.to_datetime(reference)
    #         except TypeError:
    #             raise InputArgumentError("You're expecting to pass a date object or string as a reference.")
    #         else:
    #             for tlfb_df in tlfb_dfs:
    #                 tlfb_df['date'] = (tlfb_df['date'] - reference_date).map(lambda x: x.days)
    #             visit_df['date'] = (visit_df['date'] - reference_date).map(lambda x: x.days)
    #         return *tlfb_df, visit_df

    # @staticmethod
    # def write_data_to_path(df, filepath, index=False):
    #     """
    #     Write data to the specified path
    #
    #     :param df: DataFrame
    #     :param filepath: Union[str, Path, None], the path to the file to be read
    #         Supported file types: comma-separated, tab-separated, and Excel spreadsheet, if None, no writing will be
    #         performed
    #
    #     :param index: bool, whether the index of the DataFrame will be written to the output file
    #
    #     :return: a DataFrame
    #     """
    #     if filepath is None:
    #         return
    #
    #     path = Path(filepath)
    #     file_extension = path.suffix.lower()
    #     if file_extension == ".csv":
    #         pd.DataFrame.to_csv(df, path, index=index)
    #     elif file_extension in (".xls", ".xlsx", ".xlsm", ".xlsb"):
    #         pd.DataFrame.to_excel(df, path, index=index)
    #     elif file_extension == ".txt":
    #         pd.DataFrame.to_csv(df, path, sep='\t', index=index)
    #     else:
    #         raise FileExtensionError(filepath)

