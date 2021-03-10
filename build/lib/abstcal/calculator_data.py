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
