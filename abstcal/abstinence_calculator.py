# %%
import warnings
from enum import Enum
from statistics import mean
from collections import namedtuple
from datetime import timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %%
class AbstinenceCalculatorError(Exception):
    pass


class FileExtensionError(AbstinenceCalculatorError):
    def __init__(self, file_path):
        self.file_path = file_path

    def __str__(self):
        return f"The file {self.file_path} isn't a tab-delimited text file, CSV or Excel file."


class FileFormatError(AbstinenceCalculatorError):
    pass


class InputArgumentError(AbstinenceCalculatorError):
    pass


# %%
def _show_warning(warning_message):
    warnings.warn(warning_message)


def read_subject_ids(filepath, has_header=True):
    """

    :param filepath: Union[str, path], the path to the file, which has the list of subjects that you want to
        use in later steps. Note that the list of subjects should be in the long format - just one column

    :param has_header: bool, whether the file has a header or not, by default, True

    :return: list, the list of subjects
    """
    return pd.read_csv(filepath, header='infer' if has_header else None).iloc[:, 0].to_list()


# %%
TLFBRecord = namedtuple("TLFBRecord", "id date amount imputation_code")


class DataImputationCode(Enum):
    RAW = 0
    IMPUTED = 1


# %%
class CalculatorData:
    data: pd.DataFrame
    duplicates: pd.DataFrame = None
    subject_ids: set
    _index_keys: list
    _value_key: str

    def check_duplicates(self, duplicate_kept="max"):
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

        :param filepath: Union[str, Path], the path to the file to be read
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


# %%
class TLFBData(CalculatorData):
    def __init__(self, filepath, included_subjects="all", abst_cutoff=0):
        """
        Create the instance object for the TLFB data

        :param filepath: Union[str, path], the file path to the TLFB data

        :param included_subjects: Union[list, tuple], the list of subject ids that are included in the dataset,
            the default option "all" means that all subjects in the dataset will be used

        :param abst_cutoff: Union[float, int], the cutoff used to determine if a TLFB record is abstinent or not

        """
        df = super().read_data_from_path(filepath)
        self.data = self._validated_data(df)
        if included_subjects and included_subjects != "all":
            self.data = self.data.loc[self.data["id"].isin(included_subjects), :].reset_index(drop=True)

        self._index_keys = ['id', 'date']
        self._value_key = 'amount'
        self.subject_ids = set(self.data['id'].unique())
        self.abst_cutoff = abst_cutoff

    @staticmethod
    def _validated_data(df):
        CalculatorData._validate_columns(df, ('id', 'date', 'amount'), "TLFB", "id, date, and amount")
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        df['amount'] = df['amount'].astype(float)
        return df

    def profile_data(self, min_amount_cutoff=None, max_amount_cutoff=None):
        """
        Profile the TLFB data

        :param min_amount_cutoff: Union[None, int, float], default None
            The minimal amount allowed for the consumption, lower than that is considered to be an outlier
            when it's set None, outlier detection won't consider the lower bound

        :param max_amount_cutoff: Union[None, int, float], default None
            The maximal amount allowed for the consumption, higher than that is considered to be an outlier
            when it's set None, outlier detection won't consider the higher bound

        :return: Tuple, summaries for the TLFB data at the sample and subject levels
        """
        tlfb_summary_series = self._get_tlfb_sample_summary(min_amount_cutoff, max_amount_cutoff)
        tlfb_subject_summary = self._get_tlfb_subject_summary(min_amount_cutoff, max_amount_cutoff)
        sns.distplot(self.data['amount'])
        plt.show()
        return tlfb_summary_series, tlfb_subject_summary

    def _get_tlfb_sample_summary(self, min_amount_cutoff, max_amount_cutoff):
        record_count = self.data.shape[0]
        subject_count = self.data['id'].nunique()
        record_counts = self.data.groupby('id').size()
        record_count_min, record_count_max = record_counts.min(), record_counts.max()
        tlfb_summary = {'record_count': record_count,
                        'subject_count': subject_count,
                        'records_per_subject_mean': int(record_count / subject_count),
                        'records_per_subject_range': f"{record_count_min} - {record_count_max}"}
        key_names = 'min_date max_date min_amount max_amount'.split()
        col_names = 'date date amount amount'.split()
        func_names = 'min max min max'.split()
        for key_name, col_name, func_name in zip(key_names, col_names, func_names):
            func = getattr(pd.Series, func_name)
            tlfb_summary[key_name] = func(self.data[col_name])

        max_sorted_tlfb_data = \
            self.data.sort_values(by=['amount'], ascending=False).reset_index(drop=True).iloc[0:5, :].copy()
        max_sorted_tlfb_data['date'] = max_sorted_tlfb_data['date'].dt.strftime('%m/%d/%Y')
        tlfb_summary['max_amount_5_records'] = [tuple(x) for x in max_sorted_tlfb_data.values]

        min_sorted_tlfb_data = \
            self.data.dropna().sort_values(by=['amount']).reset_index(drop=True).iloc[0:5, :].copy()
        min_sorted_tlfb_data['date'] = min_sorted_tlfb_data['date'].dt.strftime('%m/%d/%Y')
        tlfb_summary['min_amount_5_records'] = [tuple(x) for x in min_sorted_tlfb_data.values]

        count_key_names = 'missing_subject_count missing_date_count missing_amount_count duplicate_count'.split()
        columns = [self.data['id'].isna(),
                   self.data['date'].isna(),
                   self.data['amount'].isna(),
                   self.data.duplicated(['id', 'date'], keep=False)]
        if min_amount_cutoff is not None:
            count_key_names.append('outlier_amount_low_count')
            columns.append(pd.Series(self.data['amount'] < min_amount_cutoff))
        if max_amount_cutoff is not None:
            count_key_names.append('outlier_amount_high_count')
            columns.append(pd.Series(self.data['amount'] > max_amount_cutoff))
        for key_name, column in zip(count_key_names, columns):
            tlfb_summary[key_name] = column.sum()
        return pd.Series(tlfb_summary)

    def _get_tlfb_subject_summary(self, min_amount_cutoff, max_amount_cutoff):
        tlfb_dates_amounts = self.data.groupby('id').agg({
            "date": ["count", "min", "max"],
            "amount": ["min", "max", "mean"]
        })
        summary_dfs = [tlfb_dates_amounts]
        col_names = ['record_count', 'date_min', 'date_max', 'amount_min', 'amount_max',
                     'amount_mean', 'duplicates_count']
        summary_dfs.append(self.data.loc[self.data.duplicated(['id', 'date'], keep=False), :].groupby(
            'id')['amount'].agg('count'))
        if min_amount_cutoff is not None:
            summary_dfs.append(self.data.loc[self.data['amount'] < min_amount_cutoff, :].groupby(
                'id')['amount'].agg('count'))
            col_names.append('outliers_low_count')
        if max_amount_cutoff is not None:
            summary_dfs.append(self.data.loc[self.data['amount'] > max_amount_cutoff, :].groupby(
                'id')['amount'].agg('count'))
            col_names.append('outliers_high_count')
        summary_df = pd.concat(summary_dfs, axis=1)
        summary_df.columns = col_names
        return summary_df.fillna(0)

    def interpolate_biochemical_data(self, half_life_in_days, maximum_days_to_interpolate):
        """
        Interpolate the biochemical data

        :param half_life_in_days: Union[float, int]
            The half-life of the biochemical measure to interpolate the days preceding a non-abstinent day

        :param maximum_days_to_interpolate: int, The maximum number of days to interpolate

        :return: None
        """
        if half_life_in_days <= 0 or maximum_days_to_interpolate < 1 or \
                not isinstance(maximum_days_to_interpolate, int):
            InputArgumentError("The half life of the biochemical should be greater than 0, and the maximum number of "
                               "days to interpolate should be a positive integer")

        if 'imputation_code' in self.data.columns:
            self.data = self.data[self.data['imputation_code'] == 0]

        self.data.sort_values(by=self._index_keys, inplace=True, ignore_index=True)
        self.data['imputation_code'] = DataImputationCode.RAW.value

        interpolated_records = []

        for subject_id in sorted(self.subject_ids):
            subject_data = self.data[self.data['id'] == subject_id]
            interpolated_dates = set()
            for row in subject_data.itertuples():
                current_amount = row.amount
                current_date = row.date
                for day in range(1, maximum_days_to_interpolate+1):
                    interpolated_date = current_date + timedelta(days=-day)
                    if interpolated_date in interpolated_dates or current_amount < self.abst_cutoff:
                        continue
                    interpolated_amount = pow(2, day / half_life_in_days) * current_amount
                    interpolated_record = \
                        [subject_id, interpolated_date, interpolated_amount, DataImputationCode.IMPUTED.value]
                    interpolated_dates.add(interpolated_date)
                    interpolated_records.append(interpolated_record)

        interpolated_df = pd.DataFrame(interpolated_records, columns=['id', 'date', 'amount', 'imputation_code'])
        self.data = pd.concat([self.data, interpolated_df]).sort_values(by=self._index_keys, ignore_index=True)

    def recode_data(self, floor_amount=None, ceil_amount=None):
        """
        Recode the abnormal data of the TLFB dataset

        :param floor_amount: Union[None, int, float], default None
            Recode values lower than the floor amount to the floor amount.
            When None, no replacement will be performed

        :param ceil_amount: Union[None, int, float], default None
            Recode values higher than the ceil amount to the ceil amount.
            When None, no replacement will be performed

        :return: Summary of the recoding
        """
        recode_summary = dict()
        if floor_amount is not None:
            outlier_count = pd.Series(self.data['amount'] < floor_amount).sum()
            recode_summary[f'Number of outliers (< {floor_amount})'] = outlier_count
            self.data['amount'] = self.data['amount'].map(lambda x: max(x, floor_amount))
        if ceil_amount is not None:
            outlier_count = pd.Series(self.data['amount'] > ceil_amount).sum()
            recode_summary[f'Number of outliers (> {ceil_amount})'] = outlier_count
            self.data['amount'] = self.data['amount'].map(lambda x: min(x, ceil_amount))
        if recode_summary:
            return recode_summary

    def impute_data(self, impute="linear", biochemical_data=None, trigger_amount=0):
        """
        Impute the TLFB data

        :param impute: Union["uniform", "linear", None, int, float], how the missing TLFB data are imputed
            None: no imputation
            "uniform": impute the missing TLFB data using the mean value of the amounts before and after
            the missing interval
            "linear" (the default): impute the missing TLFB data by interpolating a linear trend based on the amounts
            before and after the missing interval
            Numeric value (int or float): impute the missing TLFB data using the specified value

        :type biochemical_data: TLFBData
        :param biochemical_data: The biochemical data that is used to impute the self-reported data

        :param trigger_amount: Union[float, int]
            The amount, below which the TLFB data interpolation will be triggered based on the biochemical data

        :return: Summary of the imputation
        """
        if impute is None or str(impute).lower() == "none":
            return
        if not (impute in ("uniform", "linear") or impute.isnumeric()):
            raise InputArgumentError("The imputation mode can only be None, 'uniform', 'linear', "
                                     "or a numeric value.")
        if 'imputation_code' in self.data.columns:
            self.data = self.data[self.data['imputation_code'] == 0]

        self.data.sort_values(by=self._index_keys, inplace=True, ignore_index=True)
        self.data['imputation_code'] = DataImputationCode.RAW.value

        if biochemical_data is not None:
            _biochemical_data = biochemical_data.data[["id", "amount"]].rename(columns={"amount": "chemical_amount"})
            _merged = self.data.merge(_biochemical_data, on="id")

            def interpolate_tlfb(row):
                amount = row['amount']
                imputation_code = row['imputation_code']
                if row['amount'] <= trigger_amount and row['chemical_amount'] > biochemical_data.biochemical_cutoff:
                    amount = trigger_amount
                    imputation_code = DataImputationCode.IMPUTED.value
                return amount, imputation_code

            _merged[['amount', 'imputation_code']] = _merged.apply(interpolate_tlfb, axis=1)
            self.data = _merged

        self.data['diff_days'] = self.data.groupby(['id'])['date'].diff().map(
            lambda x: x.days if pd.notnull(x) else 1)
        missing_data = self.data[self.data['diff_days'] > 1.0]
        imputed_records = []
        for row in missing_data.itertuples():
            start_data = self.data.iloc[row.Index - 1]
            start_record = TLFBRecord(start_data.id, start_data.date, start_data.amount, start_data.imputation_code)
            end_record = TLFBRecord(row.id, row.date, row.amount, start_data.imputation_code)
            imputed_records.extend(self._impute_missing_block(start_record, end_record, impute=impute))
        self.data.drop(['diff_days'], axis=1, inplace=True)
        imputed_tlfb_data = pd.DataFrame(imputed_records)
        self.data = pd.concat([self.data, imputed_tlfb_data]).sort_values(self._index_keys, ignore_index=True)
        impute_summary = self.data.groupby(['imputation_code']).size().reset_index(). \
            rename({0: "record_count"}, axis=1)
        impute_summary['imputation_code'] = impute_summary['imputation_code'].map(
            lambda x: DataImputationCode(x).name
        )
        return impute_summary

    def _get_missing_data(self):
        self.data['diff_days'] = self.data.groupby(['id'])['date'].diff().map(
            lambda x: x.days if pd.notnull(x) else 1)
        return self.data[self.data['diff_days'] > 1.0]

    @staticmethod
    def _impute_missing_block(start_record: TLFBRecord, end_record: TLFBRecord, impute):
        subject_id, start_date, start_amount, _ = start_record
        subject_id, end_date, end_amount, _ = end_record
        imputation_code = DataImputationCode.IMPUTED.value
        day_number = (end_date - start_date).days
        imputed_records = []
        if impute == "linear":
            m = (end_amount - start_amount) / day_number
            for i in range(1, day_number):
                imputed_date = start_date + timedelta(days=i)
                imputed_amount = m * i + start_amount
                imputed_records.append(TLFBRecord(subject_id, imputed_date, imputed_amount, imputation_code))
        elif impute == "uniform":
            imputed_amount = mean([start_amount, end_amount])
            for i in range(1, day_number):
                imputed_date = start_date + timedelta(days=i)
                imputed_records.append(TLFBRecord(subject_id, imputed_date, imputed_amount, imputation_code))
        else:
            imputed_amount = float(impute)
            for i in range(1, day_number):
                imputed_date = start_date + timedelta(days=i)
                imputed_records.append(TLFBRecord(subject_id, imputed_date, imputed_amount, imputation_code))
        return imputed_records

    def get_subject_data(self, subject_id, start_date, end_date, mode):
        df = self.data
        if mode != "itt" and 'imputation_code' in self.data.columns:
            df = self.data.loc[self.data['imputation_code'] == DataImputationCode.RAW.value, :]
        subject_data = df[(df['id'] == subject_id) &
                          (start_date <= df['date']) &
                          (df['date'] < end_date)]
        return subject_data


# %%
class VisitData(CalculatorData):
    def __init__(self, filepath, data_format="long", included_visits="all", included_subjects="all"):
        """
        Create the instance object for the visit data

        :param filepath: Union[str, path], the file path to the visit data

        :param data_format: Union["long", "wide"], how the visit data are organized
            long: the data should have three columns, id, visit, and date
            wide: the data have multiple columns and one row per subject, the first column has to be named 'id'
                and the remaining columns are visits with values being the dates

        :type included_visits: Union[str, list, tuple]
        :param included_visits: Union[str, list, tuple], the list of visits that are included in the dataset, the
            default option "all" means that all visits in the dataset will be used

        :param included_subjects: Union[list, tuple], the list of subject ids that are included in the dataset,
            the default option "all" means that all subjects in the dataset will be used

        """
        if data_format == "wide":
            df_wide = super().read_data_from_path(filepath)
            df_long = df_wide.melt(id_vars="id")
        else:
            df_long = super().read_data_from_path(filepath)
        self.data = self._validated_data(df_long)
        if included_visits and included_visits != "all":
            self.data = self.data.loc[self.data["visit"].isin(included_visits), :]
        if included_subjects and included_subjects != "all":
            self.data = self.data.loc[self.data["id"].isin(included_subjects), :]
        self.data.reset_index(drop=True, inplace=True)
        self._index_keys = ['id', 'visit']
        self._value_key = 'date'
        self.visits = set(self.data['visit'].unique())
        self.subject_ids = set(self.data['id'].unique())

    @staticmethod
    def _validated_data(df):
        CalculatorData._validate_columns(df, ('id', 'visit', 'date'), "visit", "id, visit, and date")
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        return df

    def profile_data(self, min_date_cutoff=None, max_date_cutoff=None, expected_visit_order="infer"):
        """
        Profile the visit data

        :param min_date_cutoff: Union[None, datetime, str], default None
            The minimal date allowed for the visit's date, lower than that is considered to be an outlier
            When it's set None, outlier detection won't consider the lower bound
            When it's str, it should be able to be casted to a datetime object, mm/dd/yyyy, such as '10/23/2020'

        :param max_date_cutoff: Union[None, datetime, str], default None
            The maximal amount allowed for the consumption, higher than that is considered to be an outlier
            When it's set None, outlier detection won't consider the higher bound
            When it's str, it should be able to be casted to a datetime object, such as '10/23/2020'

        :type expected_visit_order: object
        :param expected_visit_order: Union["infer", list, None], default "infer"
            The expected order of the visits, such that when dates are out of order can be detected
            The list of visit names should match the actual visits in the dataset
            When None, no checking will be performed
            The default is infer, and it means that the order of the visits is sorted based on its numeric or
            alphabetic order

        :return: Tuple, summaries of the visit data at the sample and subject level
        """
        casted_min_date_cutoff = pd.to_datetime(min_date_cutoff, infer_datetime_format=True)
        casted_max_date_cutoff = pd.to_datetime(max_date_cutoff, infer_datetime_format=True)
        visit_summary_series = self._get_visit_sample_summary(casted_min_date_cutoff, casted_max_date_cutoff)
        visit_subject_summary = self._get_visit_subject_summary(casted_min_date_cutoff, casted_max_date_cutoff,
                                                                expected_visit_order)
        return visit_summary_series, visit_subject_summary

    def get_out_of_order_visit_data(self, expected_visit_order='infer'):
        """
        :type expected_visit_order: object
        :param expected_visit_order: Union["infer", list, None], default "infer"
            The expected order of the visits, such that when dates are out of order can be detected
            The list of visit names should match the actual visits in the dataset
            When None, no checking will be performed
            The default is infer, and it means that the order of the visits is sorted based on its numeric or
            alphabetic order

        :return: DataFrame or None
        """
        subject_summary = self._get_visit_subject_summary(expected_visit_order=expected_visit_order).reset_index()
        out_of_order_ids = \
            subject_summary.loc[subject_summary['visit_dates_out_of_order'], "id"].to_list()
        out_of_order_df = self.data.loc[self.data['id'].isin(out_of_order_ids), :].sort_values(by=['id', 'visit'])
        if not out_of_order_df.empty:
            return out_of_order_df

    def _get_visit_sample_summary(self, min_date_cutoff, max_date_cutoff):
        visit_summary = {'record_count': self.data.shape[0]}
        key_names = 'subject_count visit_count distinct_visits min_date max_date'.split()
        col_names = 'id visit visit date date'.split()
        func_names = 'nunique nunique unique min max'.split()
        for key_name, col_name, func_name in zip(key_names, col_names, func_names):
            func = getattr(pd.Series, func_name)
            visit_summary[key_name] = func(self.data[col_name])

        count_key_names = 'missing_subject_count missing_visit_count missing_date_count duplicate_count'.split()
        columns = [self.data['id'].isna(),
                   self.data['visit'].isna(),
                   self.data['date'].isna(),
                   self.data.duplicated(['id', 'visit'], keep=False)]
        if min_date_cutoff is not None:
            count_key_names.append('outlier_date_low_count')
            columns.append(pd.Series(self.data['date'] < min_date_cutoff))
        if max_date_cutoff is not None:
            count_key_names.append('outlier_date_high_count')
            columns.append(pd.Series(self.data['date'] > max_date_cutoff))
        for key_name, column in zip(count_key_names, columns):
            visit_summary[key_name] = column.sum()

        min_date_indices = self.data.groupby(['id'])['date'].idxmin()
        anchor_visit = self.data.loc[min_date_indices, 'visit'].value_counts().idxmax()
        visit_summary['anchor_visit'] = anchor_visit
        anchor_dates = self.data.loc[self.data['visit'] == anchor_visit, ['id', 'date']].copy().\
            rename({'date': 'anchor_date'}, axis=1)
        visit_data_anchored = pd.merge(self.data, anchor_dates, on='id')
        visit_data_anchored['interval_to_anchor'] = (visit_data_anchored['date'] - visit_data_anchored['anchor_date']).\
            map(lambda x: x.days)
        self._visit_data = visit_data_anchored.drop(['anchor_date'], axis=1)

        grid = sns.FacetGrid(self._visit_data.loc[self._visit_data['visit'] != anchor_visit, :], col="visit")
        grid.map(plt.hist, "interval_to_anchor")
        plt.show()

        for visit in self.visits - {anchor_visit}:
            visit_dates = self._visit_data.dropna().loc[self.data['visit'] == visit, :].\
                sort_values(by=['interval_to_anchor', 'id'], ignore_index=True)
            visit_summary[f"v{visit}_v{anchor_visit}_interval_days_range"] = \
                f"{visit_dates['interval_to_anchor'].min()} - {visit_dates['interval_to_anchor'].max()}"
            visit_summary[f"v{visit}_v{anchor_visit}_interval_days_min5"] = \
                [tuple(x) for x in visit_dates.loc[0:4, ['id', 'interval_to_anchor']].values]
            rows = visit_dates.shape[0]
            visit_summary[f"v{visit}_v{anchor_visit}_interval_days_max5"] = \
                [tuple(x) for x in reversed(visit_dates.loc[rows - 5:rows - 1, ['id', 'interval_to_anchor']].values)]

        visit_record_counts = self.data.groupby(['visit'])['date'].count().to_dict()
        subject_count = len(self.subject_ids)
        visit_summary.update(
            {f"visit_{visit}_retention": f"{count} ({count / subject_count:.2%})"
             for visit, count in visit_record_counts.items()}
        )
        return pd.Series(visit_summary)

    def _get_visit_subject_summary(self, min_date_cutoff=None, max_date_cutoff=None, expected_visit_order='infer'):
        visit_dates_amounts = self.data.groupby('id').agg({
            "date": ["count", "min", "max"]
        })
        visit_dates_amounts['date_interval'] = \
            (visit_dates_amounts[('date', 'max')] - visit_dates_amounts[('date', 'min')]).map(lambda x: x.days)
        summary_dfs = [visit_dates_amounts]
        col_names = ['record_count', 'date_min', 'date_max', 'date_interval', 'duplicates_count']
        summary_dfs.append(self.data.loc[self.data.duplicated(['id', 'visit'], keep=False), :].groupby(
            'id')['date'].agg('count'))
        if min_date_cutoff is not None:
            summary_dfs.append(self.data.loc[self.data['date'] < min_date_cutoff, :].groupby(
                'id')['date'].agg('count'))
            col_names.append('outliers_date_low_count')
        if max_date_cutoff is not None:
            summary_dfs.append(self.data.loc[self.data['date'] > max_date_cutoff, :].groupby(
                'id')['date'].agg('count'))
            col_names.append('outliers_date_high_count')
        visit_dates_out_of_order = self._check_visit_order(expected_visit_order)
        summary_dfs.append(visit_dates_out_of_order)
        col_names.append('visit_dates_out_of_order')
        summary_df = pd.concat(summary_dfs, axis=1)
        summary_df.columns = col_names
        return summary_df.fillna(0)

    def _check_visit_order(self, expected_visit_order):
        if expected_visit_order is not None:
            if isinstance(expected_visit_order, list):
                self.data['visit'] = self.data['visit'].map(
                    {visit: i for i, visit in enumerate(expected_visit_order)})
            elif isinstance(expected_visit_order, str) and expected_visit_order != 'infer':
                _show_warning("Supported options for expected_visit_order are list of visits, None, and infer. "
                              "The expected visit order is inferred to check if the dates are in the correct order.")
            sorted_visit_data = self.data.sort_values(by=['id', 'visit'], ignore_index=True)
            sorted_visit_data['ascending'] = sorted_visit_data.groupby(['id'])['date'].diff().map(
                lambda x: True if x is pd.NaT or x.days >= 0 else False
            )
            visits_out_of_order = sorted_visit_data.groupby(['id'])['ascending'].all().map(lambda x: not x)
            if visits_out_of_order.sum() > 0:
                _show_warning(f"Please note that some subjects (n={visits_out_of_order.sum()}) appear to have their "
                             f"visit dates out of the correct order. You can find out who they are in the visit data "
                             f"summary by subject. Please fix them if applicable.")
            return visits_out_of_order

    def recode_data(self, floor_date=None, ceil_date=None):
        """
        Recode the abnormal data of the TLFB dataset

        :param floor_date: Union[None, int, float], default None
            Recode values lower than the floor amount to the floor amount.
            When None, no replacement will be performed

        :param ceil_date: Union[None, int, float], default None
            Recode values higher than the ceil amount to the ceil amount.
            When None, no replacement will be performed

        :return: summary of the recoding
        """
        recode_summary = dict()
        if floor_date is not None:
            casted_floor_date = pd.to_datetime(floor_date, infer_datetime_format=True)
            outlier_count = pd.Series(self.data['date'] < casted_floor_date).sum()
            recode_summary[f"Number of outliers (< {casted_floor_date.strftime('%m/%d/%Y')})"] = outlier_count
            self.data['date'] = self.data['date'].map(lambda x: max(x, casted_floor_date))
        if ceil_date is not None:
            casted_ceil_date = pd.to_datetime(ceil_date, infer_datetime_format=True)
            outlier_count = pd.Series(self.data['date'] > casted_ceil_date).sum()
            recode_summary[f"Number of outliers (> {casted_ceil_date.strftime('%m/%d/%Y')})"] = outlier_count
            self.data['date'] = self.data['date'].map(lambda x: min(x, casted_ceil_date))
        return recode_summary

    # noinspection PyTypeChecker
    def impute_data(self, anchor_visit='infer', impute='freq'):
        """
        Impute any missing visit data.

        :type anchor_visit: object
        :param anchor_visit: The anchor visit, which needs to exist for all subjects and the date of which will be
            used to impute the missing visit dates

        :param impute: Union["freq", "mean", None], how the missing visit data are imputed
        None: no imputation
        "freq": use the most frequent, see below for further clarification
        "mean": use the average, see below for further clarification
            Both imputation methods assume that the visit structure is the same among all subjects. It will first find
            the earliest visit as the anchor date, impute any missing visit dates either using the average or the most
            frequent interval. Please note the anchor dates can't be missing.

        :return DataFrame
        """

        if impute is None or str(impute).lower() == "none":
            return
        if impute not in ('freq', 'mean'):
            raise InputArgumentError('You can only specify the imputation method to be "freq" or "mean".')

        if 'imputation_code' in self.data.columns:
            self.data = self.data[self.data['imputation_code'] == 0]

        self.data = self.data.sort_values(by=self._index_keys, ignore_index=True)
        if anchor_visit == 'infer':
            min_date_indices = self.data.groupby(['id'])['date'].idxmin()
            anchor_visit = self.data.loc[min_date_indices, 'visit'].value_counts().idxmax()
        visit_ids = sorted(self.subject_ids)
        anchor_ids = set(self.data.loc[self.data['visit'] == anchor_visit, 'id'].unique())
        missing_anchor_ids = set(visit_ids) - anchor_ids
        if missing_anchor_ids:
            message = f"Subjects {missing_anchor_ids} are missing anchor visit {anchor_visit}. " \
                      f"There might be problems calculating abstinence data for these subjects."
            _show_warning(message)
        ids_s = pd.Series(visit_ids, name='id')
        anchor_dates = self.data.loc[self.data['visit'] == anchor_visit, ['id', 'date']].\
            rename({'date': 'anchor_date'}, axis=1)
        anchor_df = pd.merge(ids_s, anchor_dates, how='outer', on='id')

        df_anchor = anchor_df.copy()
        df_anchor['visit'] = anchor_visit
        df_anchor['imputed_date'] = df_anchor['date'] = df_anchor['anchor_date']
        imputed_visit_dfs = [df_anchor]

        for visit in self.visits - {anchor_visit}:
            visit_dates = self.data.loc[self.data['visit'] == visit, ['id', 'date']]
            df_visit = pd.merge(anchor_df, visit_dates, how='outer', on='id')
            days_diff = (df_visit['date'] - df_visit['anchor_date']).map(
                lambda day_diff: day_diff.days if pd.notnull(day_diff) else np.nan)
            used_days_diff = days_diff.value_counts().idxmax() if impute == 'freq' else int(days_diff.mean())

            def impute_date(x):
                if pd.notnull(x['date']):
                    return x['date']
                imputed_date = x['anchor_date'] + timedelta(days=used_days_diff)
                return imputed_date

            df_visit['imputed_date'] = df_visit.apply(impute_date, axis=1)
            df_visit['visit'] = visit
            imputed_visit_dfs.append(df_visit)

        visit_data_imputed = pd.concat(imputed_visit_dfs)
        visit_data_imputed['imputation_code'] = \
            visit_data_imputed['date'].isnull().map(int)
        self.data = visit_data_imputed.drop(['date', 'anchor_date'], axis=1). \
            rename({'imputed_date': 'date'}, axis=1).sort_values(by=self._index_keys, ignore_index=True)
        impute_summary = self.data.groupby(['imputation_code']).size().reset_index(). \
            rename({0: "record_count"}, axis=1)
        impute_summary['imputation_code'] = impute_summary['imputation_code'].map(
            lambda x: DataImputationCode(x).name
        )
        return impute_summary

    def validate_visits(self, visits_to_check):
        if not set(visits_to_check).issubset(self.visits):
            raise InputArgumentError(f"Some visits are not in the visit list {self.visits}. "
                                     f"Please check your arguments.")

    def get_visit_dates(self, subject_id, visit_names, increment_days=0, mode='itt'):
        used_data = self.data
        if mode != 'itt':
            used_data = self.data.loc[self.data['imputation_code'] == DataImputationCode.RAW.value, :]
        if not isinstance(visit_names, list):
            visit_names = [visit_names]
        visit_dates_cond = (used_data['id'] == subject_id) & (used_data['visit'].isin(visit_names))
        dates = list(used_data.loc[visit_dates_cond, 'date'])
        if increment_days:
            dates = [date + timedelta(days=increment_days) for date in dates]
        return dates[0] if len(dates) == 1 else dates


# %%
class AbstinenceCalculator:
    def __init__(self, tlfb_data: TLFBData, visit_data: VisitData, biochemical_data: TLFBData = None):
        """
        Create an instance object to calculate abstinence

        :param tlfb_data: TLFBData, the TLFB data

        :param visit_data: VisitData, the visit data

        :param biochemical_data: TLFBData, the biochemical data

        :return: None
        """
        self.tlfb_data = tlfb_data
        self.visit_data = visit_data
        self.biochemical_data = biochemical_data
        self.exceeded_cutoff = None
        if biochemical_data is not None:
            condition = biochemical_data.data['amount'] > biochemical_data.abst_cutoff
            self.exceeded_cutoff = biochemical_data.data[condition].sort_values(by=['id', 'date'], ignore_index=True)
        self.subject_ids = sorted(tlfb_data.subject_ids & tlfb_data.subject_ids)

    def check_data_availability(self):
        tlfb_ids = pd.Series({subject_id: 'Yes' for subject_id in self.tlfb_data.subject_ids}, name='TLFB Data')
        visit_ids = pd.Series({subject_id: 'Yes' for subject_id in self.visit_data.subject_ids}, name='Visit Data')
        crossed_data = pd.concat([tlfb_ids, visit_ids], axis=1).fillna('No')
        freq_data = crossed_data.groupby(['TLFB Data', 'Visit Data']).size().reset_index().\
            rename({0: "subject_count"}, axis=1)
        return freq_data

    def abstinence_cont(self, start_visit, end_visits, abst_var_names='infer', including_end=False, mode="itt"):
        """
        Calculates the continuous abstinence for the time window.

        :param start_visit: the visit where the time window starts, it should be one of the visits in the visit dataset

        :param end_visits: Union[list, tuple, or Any] the visits for the end(s) of the time window,
            a list of the visits or a single visit, which should all belong to the visits in the visit dataset

        :param abst_var_names: Union[str, list], the name(s) of the abstinence variable(s)
            "infer": the default option, the name(s) will be inferred,
            Note that when specified, the number of abstinence variable names should match the number of end visits

        :param including_end: bool, whether you want to include the end visit or not, default=False

        :param mode: Union["itt", "ro"], how to calculate the abstinence, "itt"=intention-to-treat (the default) or
            "ro"=responders-only

        :return: Pandas DataFrame, subject id and abstinence results
        """
        end_visits = AbstinenceCalculator._listize_args(end_visits)
        abst_names = \
            AbstinenceCalculator._infer_abst_var_names(end_visits, abst_var_names, f'{mode}_cont_v{start_visit}')
        self.visit_data.validate_visits([start_visit, *end_visits])
        if len(end_visits) != len(abst_names):
            raise InputArgumentError("The number of abstinence variable names should match the number of visits.")

        results = []
        lapses = []
        for subject_id in self.subject_ids:
            result = [subject_id]

            start_date = self.visit_data.get_visit_dates(subject_id, start_visit, mode=mode)
            end_dates = self.visit_data.get_visit_dates(subject_id, end_visits, including_end, mode=mode)

            for end_date_i, end_date in enumerate(end_dates):
                abstinent, lapse = self._score_continuous(subject_id, start_date, end_date, mode)
                result.append(abstinent)
                AbstinenceCalculator._append_lapse_as_needed(lapses, lapse, abst_names[end_date_i])

            results.append(result)

        abstinence_df = pd.DataFrame(results, columns=['id', *abst_names]).set_index("id")
        lapses_df = pd.DataFrame(lapses, columns=['id', 'date', 'amount', 'abst_name']).\
            sort_values(by=['abst_name', 'id', 'date'])

        return abstinence_df, lapses_df

    def abstinence_pp(self, end_visits, days, abst_var_names='infer', including_anchor=False, mode="itt"):
        """
        Calculate the point-prevalence abstinence using the end visit's date.

        :param end_visits: The reference visit(s) on which the abstinence is to be calculated

        :param days: The number of days preceding the end visit(s), it can be a single integer, or a list/tuple of days

        :param abst_var_names: The name(s) of the abstinence variable(s), by default, the name(s) will be inferred,
            if not inferred, the number of abstinence variable names should match the number of end visits

        :param including_anchor: Whether you want to include the anchor visit or not, default=False

        :param mode: How you want to calculate the abstinence, "itt"=intention-to-treat (the default) or
            "ro"=responders-only

        :return: Pandas DataFrame, subject id and abstinence results
        """
        days = AbstinenceCalculator._listize_args(days)
        if any(day < 1 for day in days):
            InputArgumentError("The number of days has to be a positive integer.")
        end_visits = AbstinenceCalculator._listize_args(end_visits)
        self.visit_data.validate_visits(end_visits)

        all_abst_names = list()
        for day in days:
            abst_names = AbstinenceCalculator._infer_abst_var_names(end_visits, abst_var_names, f'{mode}_pp{day}')
            all_abst_names.extend(abst_names)

        if len(end_visits) * len(days) != len(all_abst_names):
            raise InputArgumentError("The number of abstinence variable names should be equal to the multiplication of"
                                     "the number of day conditions and the number of visits.")

        results = []
        lapses = []

        for subject_id in self.subject_ids:
            result = [subject_id]
            end_dates = self.visit_data.get_visit_dates(subject_id, end_visits, including_anchor, mode=mode)

            for day_i, day in enumerate(days):
                for end_date_i, end_date in enumerate(end_dates):
                    start_date = end_date + timedelta(days=-day)
                    abstinent, lapse = self._score_continuous(subject_id, start_date, end_date, mode)
                    result.append(abstinent)
                    abst_name = all_abst_names[len(end_dates) * day_i + end_date_i]
                    AbstinenceCalculator._append_lapse_as_needed(lapses, lapse, abst_name)

            results.append(result)

        abstinence_df = pd.DataFrame(results, columns=['id', *all_abst_names]).set_index("id")
        lapses_df = pd.DataFrame(lapses, columns=['id', 'date', 'amount', 'abst_name']).\
            sort_values(by=['abst_name', 'id', 'date'])

        return abstinence_df, lapses_df

    def abstinence_prolonged(self, quit_visit, end_visits, lapse_criterion, grace_days=14, abst_var_names='infer',
                             including_anchor=False, mode="itt"):
        """
        Calculate the prolonged abstinence using the time window

        :param quit_visit: The visit when the subjects are scheduled to quit smoking

        :param end_visits: The end visits for which the abstinence data are to be calculated

        :param lapse_criterion: Union[False, str, list], the criterion for finding a lapse, which is only examined
            in the time window between the date when the grace period is over and the end date
            False: lapse is not allowed
            Use amounts: it must start with a numeric value, and not end with days, such as "5 cigs", "4 drinks"
            Use days: it must end with days, such as "5 days", "7 days"
            Use amounts over time: such as "5 cigs/7 days", the time intervals are rolling windows
            Use days over time: such as "5 days/7 days", the time intervals are rolling windows
            Use multiple criteria: Combinations of any of these in a tuple or list, such as ("5 cigs", "2 days",
                "2 days/7 days", False)

        :param grace_days: The number of days for the grace period following the quit date

        :param abst_var_names: The names of the abstinence variable, by default, the name will be inferred

        :param including_anchor: Whether you want to include the anchor visit or not, default=False

        :param mode: How you want to calculate the abstinence, "itt"=intention-to-treat (the default) or
            "ro"=responders-only

        :return: Pandas DataFrame with two columns, subject id and abstinence result
        """
        end_visits = AbstinenceCalculator._listize_args(end_visits)
        self.visit_data.validate_visits([quit_visit, *end_visits])

        all_abst_names = list()
        criteria = AbstinenceCalculator._listize_args(lapse_criterion)
        for criterion in criteria:
            if criterion:
                parsed_criterion_len = len([x for parts in criterion.split("/") for x in parts.split()])
                if parsed_criterion_len not in (2, 4):
                    raise InputArgumentError("When lapse is allowed, you have to specify the criterion for lapses in "
                                             "strings, such as '5 cigs', '5 drinks'. To see the full list of supported"
                                             "criteria, please refer to the help menu.")
            else:
                assert criterion in (False,)
            formatted_criterion = criterion.replace(" ", "_") if isinstance(criterion, str) else criterion
            abst_names = AbstinenceCalculator.\
                _infer_abst_var_names(end_visits, abst_var_names, f'{mode}_prolonged_{formatted_criterion}')
            all_abst_names.extend(abst_names)

        if len(end_visits) * len(criteria) != len(all_abst_names):
            raise InputArgumentError("The number of abstinence variable names should be equal to the multiplication of"
                                     "the number of lapse criteria and the number of visits.")

        results = []
        lapses = []

        for subject_id in self.subject_ids:
            result = [subject_id]
            start_date = self.visit_data.get_visit_dates(subject_id, quit_visit, grace_days, mode=mode)
            end_dates = self.visit_data.get_visit_dates(subject_id, end_visits, including_anchor, mode=mode)

            for criterion_i, criterion in enumerate(criteria):
                for end_date_i, end_date in enumerate(end_dates):
                    abstinent = 0 if mode == 'itt' else np.nan
                    lapse = None

                    if AbstinenceCalculator._validate_dates(subject_id, start_date, end_date):
                        subject_data = self.tlfb_data.get_subject_data(subject_id, start_date, end_date, mode)
                        if not criterion:
                            abstinent, lapse = self._continuous_abst(subject_id, start_date, end_date, mode)
                        else:
                            parsed_criterion = [x for parts in criterion.split("/") for x in parts.split()]
                            if len(parsed_criterion) == 2:
                                lapse_threshold = float(parsed_criterion[0])
                                lapse_tracking = 0
                                for record in subject_data.itertuples():
                                    if parsed_criterion[-1] != "days":
                                        lapse_tracking += record.amount
                                    else:
                                        lapse_tracking += (record.amount > self.tlfb_data.abst_cutoff)
                                    if lapse_tracking >= lapse_threshold:
                                        lapse = record
                                        break
                                else:
                                    abstinent = 1
                            else:
                                assert len(parsed_criterion) == 4
                                cutoff_amount, cutoff_unit, window_amount, _ = parsed_criterion
                                cutoff_amount = float(cutoff_amount)
                                window_amount = int(float(window_amount))
                                index_list = \
                                    subject_data.index[subject_data['amount'] > self.tlfb_data.abst_cutoff].tolist()
                                for j in index_list:
                                    one_window = range(j, j + window_amount)
                                    lapse_tracking = 0
                                    for elem_i, elem in enumerate(one_window):
                                        if cutoff_unit == "days":
                                            lapse_tracking += elem in index_list
                                        else:
                                            lapse_tracking += subject_data.loc[elem_i, "amount"]
                                        if lapse_tracking > cutoff_amount:
                                            lapse = subject_data.iloc[elem_i, :]
                                            break
                                else:
                                    abstinent = 1

                    result.append(abstinent)
                    abst_name = all_abst_names[len(end_dates) * criterion_i + end_date_i]
                    AbstinenceCalculator._append_lapse_as_needed(lapses, lapse, abst_name)

            results.append(result)

        abstinence_df = pd.DataFrame(results, columns=['id', *all_abst_names]).set_index("id")
        lapses_df = pd.DataFrame(lapses, columns=['id', 'date', 'amount', 'abst_name']).\
            sort_values(by=['abst_name', 'id', 'date'])

        return abstinence_df, lapses_df

    @staticmethod
    def _listize_args(raw_args):
        if not isinstance(raw_args, (tuple, list)):
            raw_args = [raw_args]
        return raw_args

    @staticmethod
    def _infer_abst_var_names(end_visits, abst_var_names, prefix=''):
        if abst_var_names == 'infer':
            abst_names = [f"{prefix}_v{end_visit}" for end_visit in end_visits]
        else:
            abst_names = AbstinenceCalculator._listize_args(abst_var_names)
        if len(end_visits) != len(abst_names):
            raise InputArgumentError("The number of abstinence variable names should match the number of visits.")
        return abst_names

    @staticmethod
    def _validate_dates(subject_id, start_date, end_date):
        validated = False
        if start_date >= end_date:
            _show_warning(f"The end date of the time window for the subject {subject_id} {end_date} isn't later "
                         f"than the start date {start_date}. Please verify that the visit dates are correct.")
        elif pd.NaT in (start_date, end_date):
            _show_warning(f"Subject {subject_id} is missing some of the date information.")
        else:
            validated = True
        return validated

    def _validate_abstinent_biochemically(self, subject_id, start_date, end_date, abstinent):
        if abstinent:
            if self.exceeded_cutoff is not None:
                print(f"Try to correct Abstinence data get overridden for {subject_id}, {start_date!r}, {end_date!r}")
                id_check = self.exceeded_cutoff['id'] == subject_id
                start_check = self.exceeded_cutoff['date'] >= start_date
                end_check = self.exceeded_cutoff['date'] < end_date
                combined = id_check & start_check & end_check
                print(id_check.sum(), start_check.sum(), end_check.sum(), combined.sum())
                subject_exceeded = self.exceeded_cutoff[(self.exceeded_cutoff['id'] == subject_id) &
                                                        (self.exceeded_cutoff['date'] >= start_date) &
                                                        (self.exceeded_cutoff['date'] < end_date)]
                print(subject_exceeded)
                if not subject_exceeded.empty:
                    abstinent = 0
        return abstinent

    def _score_continuous(self, subject_id, start_date, end_date, mode):
        abstinent = 0 if mode == 'itt' else np.nan
        lapse = None
        if AbstinenceCalculator._validate_dates(subject_id, start_date, end_date):
            abstinent, lapse = self._continuous_abst(subject_id, start_date, end_date, mode)
        return abstinent, lapse

    def _continuous_abst(self, subject_id, start_date, end_date, mode):
        subject_data = self.tlfb_data.get_subject_data(subject_id, start_date, end_date, mode)
        lapse_record = None
        days_to_check = int((end_date - start_date).days)
        no_missing_data = subject_data['amount'].count() == days_to_check
        for record in subject_data.itertuples():
            if record.amount > self.tlfb_data.abst_cutoff:
                lapse_record = record
                break
        abstinent = int(no_missing_data and (lapse_record is None))
        if abstinent and self.exceeded_cutoff is not None:
            subject_exceeded = self.exceeded_cutoff[(self.exceeded_cutoff['id'] == subject_id) &
                                                    (self.exceeded_cutoff['date'] >= start_date) &
                                                    (self.exceeded_cutoff['date'] < end_date)].reset_index()
            if not subject_exceeded.empty:
                abstinent = 0
                lapse_record = subject_data[subject_data['date'] == subject_exceeded.loc[0, "date"]]
        return abstinent, lapse_record

    @staticmethod
    def _append_lapse_as_needed(lapses, lapse, abst_name):
        if lapse:
            lapse_id, lapse_date, lapse_amount = lapse.id, lapse.date, lapse.amount
            lapses.append((lapse_id, lapse_date, lapse_amount, abst_name))

    @staticmethod
    def merge_abst_data_to_file(dfs, filepath):
        """
        Merge abstinence data and write the merged DataFrame to a file

        :param dfs: Union[list, tuple], the list of abstinence data results (DataFrame)

        :param filepath: Union[str, path], the output file name with a proper extension

        :return: None
        """
        pd.concat(dfs, axis=1).to_csv(filepath)

    @staticmethod
    def merge_lapse_data_to_file(dfs, filepath):
        """
        Merge lapses data and write the merged DataFrame to a file

        :param dfs: Union[list, tuple], the list of lapse data results (DataFrame)

        :param filepath: Union[str, path], the output file name with a proper extension

        :return: None
        """
        pd.concat(dfs, axis=0).to_csv(filepath, index=False)


# %%
beam_sas_abst = pd.read_sas("beam_abst_wide_v2mac.sas7bdat")
beam_sas_abst = beam_sas_abst.rename(columns={'subjectid': 'id'}).set_index('id')

# beam_co = pd.read_sas("scored_co.sas7bdat")
# beam_co.rename(columns={
#     "SubjectID": 'id',
#     'CO': 'amount',
#     'EventNo': 'visit'}, inplace=True)
# beam_co = beam_co.merge(visit_data.data, on=['id', 'visit'])
# beam_co.drop(['visit'], axis=1, inplace=True)
# beam_co.to_csv("beam_co.csv", index=False)


# %%
included_subjects = pd.read_csv("beam_subjects.csv", usecols=['SubjectID']).iloc[:, 0].to_list()

tlfb_data = TLFBData("beam_tlfb.csv", included_subjects=included_subjects)
tlfb_sample_summary, tlfb_subject_summary = tlfb_data.profile_data()
tlfb_sample_summary.to_csv("beam_tlfb_sample_summary.csv")
tlfb_subject_summary.to_csv("beam_tlfb_subject_summary.csv")
tlfb_data.drop_na_records()
tlfb_data.check_duplicates()
tlfb_data.recode_data()
tlfb_data.impute_data()

biochemical_data = TLFBData("beam_co.csv", included_subjects=included_subjects, abst_cutoff=4)
biochemical_data.profile_data()
biochemical_data.interpolate_biochemical_data(0.5, 1)

visit_data = VisitData("beam_visit.csv", included_visits=list(range(11)), included_subjects=included_subjects)
visit_data.profile_data()
visit_data.get_out_of_order_visit_data()
visit_data.drop_na_records()
visit_data.check_duplicates()
visit_data.recode_data()
visit_data.impute_data(anchor_visit=1)
visit_data.get_out_of_order_visit_data()

abst_cal = AbstinenceCalculator(tlfb_data, visit_data)
abst_cal.check_data_availability()

# abst_cal.abstinence_prolonged(3, [5, 6], '5 cigs')
# abst_cal.abstinence_cont(4, [5, 6, 7, 8, 9, 10])
abst_cal.abstinence_pp([9, 10], 7)

abst_pp, lapses_pp = abst_cal.abstinence_pp([9, 10], 7)
abst_pros, lapses_pros = abst_cal.abstinence_prolonged(4, [9, 10], '6 cigs')
abst_prol, lapses_prol = abst_cal.abstinence_prolonged(4, [9, 10], False)


# abst_pp.head()
comparison = pd.concat([beam_sas_abst, abst_pp, abst_pros, abst_prol], axis=1)

comparison['pp7_v9_same'] = comparison['itt_pp7_v9'] == comparison['abst_EOT_7dpp']
comparison['pp7_v10_same'] = comparison['itt_pp7_v10'] == comparison['abst_M3_7dpp']
comparison['pros_v9_same'] = comparison['itt_prolonged_6_cigs_v9'] == comparison['abst_EOT_pros']
comparison['pros_v10_same'] = comparison['itt_prolonged_6_cigs_v10'] == comparison['abst_M3_pros']
comparison['prol_v9_same'] = comparison['itt_prolonged_False_v9'] == comparison['abst_EOT_prol']
comparison['prol_v10_same'] = comparison['itt_prolonged_False_v10'] == comparison['abst_M3_prol']

comparison['pp7_v9_same'].value_counts()
comparison['pp7_v10_same'].value_counts()
comparison['pros_v9_same'].value_counts()
comparison['pros_v10_same'].value_counts()
comparison['prol_v9_same'].value_counts()
comparison['prol_v10_same'].value_counts()


# lapses_pp.to_csv('beam_lapses.csv')

df = biochemical_data.data
min_date = df['date'].min()

mask = (df['id'] == 430005) & (df['date'] >= min_date)
mask.sum()
df[mask]