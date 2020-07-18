# %%
import pandas as pd
import os
import warnings
from pathlib import Path
from datetime import datetime, timedelta


# %%
class AbstinenceCalculatorError(Exception):
    pass


class FileExtensionError(AbstinenceCalculatorError):
    def __init__(self, file_path):
        self.file_path = file_path

    def __str__(self):
        return f"The file {self.file_path} isn't a tab-delimited text file, CSV or excel file."


class DataSourceMissingError(AbstinenceCalculatorError):
    def __str__(self):
        return "You have to set your timeline follow back data and visit data."


class FileFormatError(AbstinenceCalculatorError):
    pass


# %%
# tlfb_data: pd.DataFrame = pd.read_sas("tbltimelinefb.sas7bdat")
# tlfb_data.drop('SubstanceId', axis=1, inplace=True)
# %%
# tlfb_data.rename({"SubjectID": "id", "NumCig": "amount"}, axis=1, inplace=True)
# %%
# Validate TLFB data
# needed_cols_ordered = ('id', 'date', 'amount')
# needed_cols_unordered = set(needed_cols_ordered)
# current_cols_unordered = {'id', 'date', 'amount'}
#
# if len(tlfb_data.columns) != len(needed_cols_ordered):
#     raise FileFormatError('The TLFB data should have id, date, and amount columns.')
#
# if needed_cols_unordered != current_cols_unordered:
#     message = """The TLFB data don't appear to have the needed columns: id, date, and amount. It has been rename in
#     such order. If your columns aren't in this order, and you'll encounter error in a later calculation step."""
#     warnings.warn(message)
#     tlfb_data.columns = needed_cols_ordered
# %%
# Remove duplicate rows
# if duplicate_counts := len(tlfb_data.duplicated(['id', 'date'], keep='first')):
#     message = f"The TLFB have {duplicate_counts} duplicates based on id and date."
#     warnings.warn(message)
#     tlfb_data.drop_duplicates(["id", "date"], inplace=True)


# %%
class AbstinenceCalculator:
    """
    An Abstinence Calculator based on Timeline follow back (TLFB) and visit data

    :param tlfb_filepath: filepath for the TLFB data in the tabular format
    :param visit_filepath: filepath for the visit data in the tabular format
    :type tlfb_filepath: str, pathlib.Path
    :type visit_filepath: str, pathlib.Path

    """
    def __init__(self, tlfb_filepath, visit_filepath):
        self._tlfb_data = read_data_from_path(tlfb_filepath)
        self._visit_data = read_data_from_path(visit_filepath)
        self._tlfb_duplicates = None
        self._visit_duplicates = None

    @classmethod
    def read_data(cls, tlfb_filepath, visit_filepath):
        calculator = cls(tlfb_filepath, visit_filepath)
        return calculator

    def prepare_tlfb_data(self, to_impute=True):
        self.validate_tlfb_data()
        self.remove_tlfb_duplicates()
        self.sort_tlfb_data()
        if to_impute:
            self.impute_tlfb_data()

    def validate_tlfb_data(self):
        needed_cols_ordered = ('id', 'date', 'amount')
        needed_cols_unordered = set(needed_cols_ordered)
        current_cols_unordered = set(self._tlfb_data.columns)
        if len(current_cols_unordered) != len(needed_cols_unordered):
            raise FileFormatError('The TLFB data should have only id, date, and amount columns.')
        if needed_cols_unordered != current_cols_unordered:
            warning_message = """The TLFB data don't appear to have the needed columns: id, date, and amount. It has 
            been renamed in such order. If your columns aren't in this order, and you'll encounter error in a later 
            calculation step."""
            show_warning(warning_message)
            self._tlfb_data.columns = needed_cols_ordered
        self._tlfb_data['date'] = pd.to_datetime(self._tlfb_data['date'], errors='coerce', infer_datetime_format=True)
        self._tlfb_data['amount'] = self._tlfb_data['amount'].astype(float)

    def remove_tlfb_duplicates(self):
        self._tlfb_duplicates = self._tlfb_data.duplicated(['id', 'date'], keep=False)
        if self._tlfb_duplicates is not None:
            message = f"""The TLFB data have {len(self._tlfb_duplicates)} duplicates based on id and date. Extra 
            duplicates will be removed in the calculation. To get the duplicates, use get_TLFB_duplicates()."""
            show_warning(message)
            self._tlfb_data.drop_duplicates(["id", "date"], inplace=True)

    def sort_tlfb_data(self):
        self._tlfb_data = self._tlfb_data.sort_values(by=["id", "date"]).reset_index(drop=True)

    def get_tlfb_duplicates(self):
        return self._tlfb_duplicates

    def get_data_summary(self, outfile_name="TLFB_data_summary"):
        summary_df = self._tlfb_data.groupby("id").agg({"date": ["min", "max", "count"], "amount": ["mean"]})
        summary_df.columns = ['date_min', 'date_max', 'record_count', 'amount_mean']
        summary_df.reset_index()
        summary_df.to_csv(outfile_name + ".csv")

    def impute_tlfb_data(self):
        tlfb_df = self._tlfb_data
        tlfb_df['diff_days'] = tlfb_df.groupby(['id'])['date'].diff().map(
            lambda x: x.days if pd.notnull(x) else 1)
        tlfb_df['imputation_code'] = 0
        missing_data = tlfb_df[tlfb_df['diff_days'] > 1.0]
        ids, dates, amounts, imputation_codes = [], [], [], []
        for x in missing_data.itertuples():
            start_data = tlfb_df.iloc[x.Index - 1]
            days = x.diff_days
            start_amount = start_data.amount
            start_date = start_data.date
            end_amount = x.amount
            for i in range(1, days):
                ids.append(x.id)
                dates.append(start_date + timedelta(days=i))
                amount, imputation_code = impute_tlfb_block(start_amount, end_amount)
                amounts.append(amount)
                imputation_codes.append(imputation_code)
        tlfb_df.drop(['diff_days'], inplace=True)
        imputed_tlfb_data = pd.DataFrame.from_dict({
            'id': ids,
            'date': dates,
            'amount': amounts,
            'imputation_code': imputation_codes
        })
        self._tlfb_data = pd.concat([self._tlfb_data, imputed_tlfb_data]).sort_values(['id', 'date']).reset_index(drop=True)
        message = f"The number of imputed records: {len(ids)}"
        show_warning(message)

    def prepare_visit_data(self, data_format, how_impute_visit=None):
        """
        Prepare the visit data ready for calculation.
        :param data_format: the data format, "wide"=each row stores one subject with all his/her visit dates as columns,
                            so each subject has only one row
                            or "long"=each row stores one subject one visit, so each subject should have multiple rows
        :param how_impute_visit: how the missing visits are imputed, None=no imputation, "freq"=use the most frequent,
                                 "mean"=use the average.
                                  Both imputation methods assume that the visit structure is the same among all
                                  subjects. It will first find the earliest visit as the anchor date, impute any missing
                                  visit dates either using the average or the most frequent interval.
        """
        self.validate_visit_data()
        self.remove_visit_duplicates_if_applicable()
        self.impute_visit_data(how_impute_visit)

    def validate_visit_data(self):
        if 'id' not in self._visit_data.columns:
            show_warning("Your visit data doesn't have the id column. The calculator will use the first column as id.")
            self._visit_data.rename({self._visit_data.columns[0]: 'id'}, inplace=True)

        unique_TLFB_ids = set(self._tlfb_data['id'].unique())
        unique_visit_ids = set(self._visit_data['id'].unique())
        if not unique_TLFB_ids.issuperset(unique_visit_ids):
            message = "The visit data don't have the visit information for all subjects. Those subjects who don't" \
                      "visit data can't be scored."
            show_warning(message)

    def remove_visit_duplicates_if_applicable(self):
        self._visit_duplicates = self._visit_data.duplicated(['id'], keep=False)
        if self._visit_duplicates is not None:
            message = f"""The visits data have {len(self._visit_duplicates)} duplicates based on id. Extra 
            duplicates will be removed in the calculation. To get the duplicates, use get_visit_duplicates()."""
            show_warning(message)
            self._visit_data.drop_duplicates(["id"], inplace=True)

    def get_visit_duplicates(self):
        return self._visit_duplicates

    def impute_visit_data(self, how_impute_visit):
        if how_impute_visit is None:
            return
        dates = self._visit_data.drop(["id"], axis=1)
        reference_date_index = dates.idxmin(axis=1).value_counts().idxmax()


def show_warning(warning_message):
    warnings.warn(warning_message)


def is_prepared(self):
    return True


def impute_tlfb_block(start_amount, end_amount):
    imputation_code = 9
    amount = (start_amount + end_amount) / 2.0
    if start_amount == 0 and end_amount == 0:
        imputation_code = 1
    elif start_amount > 0 and end_amount == 0:
        imputation_code = 2
    elif start_amount > 0 and end_amount > 0:
        imputation_code = 3
    elif start_amount > 0 and end_amount > 0:
        imputation_code = 4
    return amount, imputation_code


def read_tlfb_data(file_path):
    raw_tlfb = read_data_from_path(file_path)
    validated_tlfb = validate_tlfb_data(raw_tlfb)
    return validated_tlfb


def validate_tlfb_data(tlfb_df):
    verify_tlfb_col_count(tlfb_df)
    formatted_tlfb = format_tlfb_cols(tlfb_df)
    return formatted_tlfb


def verify_tlfb_col_count(tlfb_df):
    _, col_count = tlfb_df.shape
    if col_count != 3:
        raise ValueError("Your timeline follow-back data should have only three columns: id, date, and amount.")


def format_tlfb_cols(tlfb_df):
    tlfb_df.columns = ['id', 'date', 'amount']
    tlfb_df['date'] = pd.to_datetime(tlfb_df['date'], errors='coerce', infer_datetime_format=True)
    tlfb_df['amount'] = tlfb_df['amount'].astype(float)
    return tlfb_df


def read_data_from_path(file_path):
    file_extension = os.path.splitext(file_path)
    if file_extension == "csv":
        df = pd.read_csv(file_path, infer_datetime_format=True)
    elif file_extension in ("xls", "xlsx", "xlsm", "xlsb"):
        df = pd.read_excel(file_path, infer_datetime_format=True)
    elif file_extension == "txt":
        df = pd.read_csv(file_path, sep='\t', infer_datetime_format=True)
    else:
        raise FileExtensionError(file_path)
    return df


def read_visit_data(file_path):
    visit_df = read_data_from_path(file_path)
    validate_visit_data(visit_df)
    return visit_df


def validate_visit_data(visit_df):
    return True


class AbstinenceData:
    def __init__(self, tlfb_df, visit_df):
        self.tlfb_df = tlfb_df
        self.visit_df = visit_df

    def is_valid(self):
        return self.is_tlfb_valid() and self.is_visit_valid()

    def is_tlfb_valid(self):
        return bool(self.tlfb_df)

    def is_visit_valid(self):
        return bool(self.visit_df)

    def prepare_data(self, impute_tlfb, impute_visit):
        if not self.is_valid():
            raise DataSourceMissingError
        prepare_tlfb_data(impute_tlfb)
        prepare_visit_data(impute_visit)

    def prepare_tlfb_data(self, to_impute):
        pass

    def prepare_visit_data(self, to_impute):
        pass


# %%
# abstinence_calculator = AbstinenceCalculator(tlfb_df)
# tlfb_df = pd.read_csv("/Users/ycui/PycharmProjects/abstinence_calculator/venv/bin/tlfb_edited.csv")
# abstinence_calculator.prepare_data()
# abstinence_calculator.get_data_summary()
