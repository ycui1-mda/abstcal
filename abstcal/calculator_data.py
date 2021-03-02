from enum import Enum
from pathlib import Path
import pandas as pd
from calculator_error import InputArgumentError, FileFormatError, FileExtensionError, _show_warning


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

    @staticmethod
    def read_data_from_path(filepath):
        """
        Read data from the specified path

        :param filepath: Union[str, Path, BytesIO], the path to the file to be read
            Supported file types: comma-separated, tab-separated, and Excel spreadsheet

        :return: a DataFrame
        """
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

    @staticmethod
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

    @staticmethod
    def _validate_columns(df, needed_cols_ordered, data_name, col_names):
        needed_cols_unordered = set(needed_cols_ordered)
        current_cols_unordered = set(df.columns)
        if len(current_cols_unordered) != len(needed_cols_unordered):
            raise FileFormatError(f'The {data_name} data should have only {col_names} columns.')
        if needed_cols_unordered != current_cols_unordered:
            warning_message = f"The {data_name} data does not appear to have the needed columns: {col_names}. " \
                              f"It has been renamed in such order for calculation purposes. If your columns are not " \
                              f"in this order, and you'll encounter errors in later calculation steps."
            _show_warning(warning_message)
            df.columns = needed_cols_ordered
