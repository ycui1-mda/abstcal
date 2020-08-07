# %%
import warnings
from abc import abstractmethod
from enum import Enum
from statistics import mean
from collections import namedtuple
from datetime import timedelta
from pathlib import Path
from functools import partial
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %%
def show_warning(warning_message):
    warnings.warn(warning_message)


# %%
TLFBRecord = namedtuple("TLFBRecord", "id date amount imputation_code")
VisitRecord = namedtuple("VisitRecord", "id visit date")


# %%
class CalculatorData:
    data: pd.DataFrame
    duplicates: pd.DataFrame = None
    subject_ids: set
    _index_keys: list
    _value_key: str

    def __init__(self, filepath):
        df = CalculatorData.read_data_from_path(filepath)
        self.data = self._validated_data(df)
        self.subject_ids = set(self.data['id'].unique())

    @staticmethod
    @abstractmethod
    def _validated_data(df):
        raise NotImplementedError

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

        if duplicate_kept is None or duplicates.shape[0] < 1:
            return duplicates

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
            warning_message = f"The {data_name} data doesn't appear to have the needed columns: {col_names}. " \
                              f"It has been renamed in such order for calculation purposes. If your columns aren't " \
                              f"in this order, and you'll encounter errors in later calculation steps."
            show_warning(warning_message)
            df.columns = needed_cols_ordered

    @staticmethod
    def _recode_value(floor_value, ceil_value, x):
        recoded_x = x
        if floor_value is not None:
            recoded_x = max(x, floor_value)
        if ceil_value is not None:
            recoded_x = min(x, ceil_value)
        return recoded_x


# %%
class TLFBData(CalculatorData):
    def __init__(self, filepath):
        super().__init__(filepath)
        self._index_keys = ['id', 'date']
        self._value_key = 'amount'

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

        :return: None
        """
        tlfb_summary_series = self._get_tlfb_sample_summary(min_amount_cutoff, max_amount_cutoff)
        tlfb_subject_summary = self._get_tlfb_subject_summary(min_amount_cutoff, max_amount_cutoff)
        sns.distplot(self.data['amount'])
        plt.show()
        return tlfb_summary_series, tlfb_subject_summary

    def _get_tlfb_sample_summary(self, min_amount_cutoff, max_amount_cutoff):
        tlfb_summary = {'record_count': self.data.shape[0]}
        key_names = 'subject_count min_date max_date min_amount max_amount'.split()
        col_names = 'id date date amount amount'.split()
        func_names = 'nunique min max min max'.split()
        for key_name, col_name, func_name in zip(key_names, col_names, func_names):
            func = getattr(pd.Series, func_name)
            tlfb_summary[key_name] = func(self.data[col_name])

        sorted_tlfb_data = self.data.sort_values(by=['amount'], ascending=False). \
                               reset_index(drop=True).iloc[0:5, :].copy()
        sorted_tlfb_data['date'] = sorted_tlfb_data['date'].dt.strftime('%m/%d/%Y')
        tlfb_summary['max_amount_5_records'] = \
            [tuple(x) for x in sorted_tlfb_data.iloc[0:5, :].values]
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

    def recode_data(self, floor_amount=None, ceil_amount=None):
        """
        Recode the abnormal data of the TLFB dataset

        :param floor_amount: Union[None, int, float], default None
            Recode values lower than the floor amount to the floor amount.
            When None, no replacement will be performed
        :param ceil_amount: Union[None, int, float], default None
            Recode values higher than the ceil amount to the ceil amount.
            When None, no replacement will be performed
        :return: TLFB data sorted by id and date
        """
        self.data['amount'] = self.data['amount'].map(
            partial(CalculatorData._recode_value, floor_amount, ceil_amount))

    def impute_data(self, impute="linear"):
        """
        Impute the TLFB data

        :param impute: Union["uniform", "linear", None, int, float], how the missing TLFB data are imputed
            None: no imputation
            "uniform": impute the missing TLFB data using the mean value of the amounts before and after
            the missing interval
            "linear" (the default): impute the missing TLFB data by interpolating a linear trend based on the amounts before and after
            the missing interval
            Numeric value (int or float): impute the missing TLFB data using the specified value

        :return: None
        """
        if impute is None or str(impute).lower() == "none":
            return
        if not (impute in ("uniform", "linear") or impute.isnumeric()):
            raise InputArgumentError("The imputation mode can only be None, 'uniform', 'linear', "
                                     "or a numeric value.")
        self.data.sort_values(by=self._index_keys, inplace=True, ignore_index=True)
        self.data['imputation_code'] = TLFBImputationCode.RAW.value
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
            lambda x: TLFBImputationCode(x).name
        )
        return impute_summary

    def _get_missing_data(self):
        self.data['diff_days'] = self.data.groupby(['id'])['date'].diff().map(
            lambda x: x.days if pd.notnull(x) else 1)
        return self.data[self.data['diff_days'] > 1.0]

    def _impute_missing_block(self, start_record: TLFBRecord, end_record: TLFBRecord, impute):
        subject_id, start_date, start_amount, _ = start_record
        subject_id, end_date, end_amount, _ = end_record
        imputation_code = TLFBImputationCode.IMPUTED.value
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


# %%
class VisitData(CalculatorData):
    def __init__(self, filepath, data_format="long"):
        """
        Create the instance object for the visit data

        :param filepath: Union[str, path], the file path to the visit data
        :param data_format: Union["long", "wide"], how the visit data are organized
            long: the data should have three columns, id, visit, and date
            wide: the data have multiple columns and one row per subject, the first column has to be named 'id'
                and the remaining columns are visits with values being the dates
        """
        if data_format == "wide":
            df_wide = CalculatorData.read_data_from_path(filepath)
            df_long = df_wide.pivot(index="id", columns="visit", values="date")
            self.data = CalculatorData._validated_data(df_long)
            self.subject_ids = set(self.data['id'].unique())
        else:
            super().__init__(filepath)
        self._index_keys = ['id', 'visit']
        self._value_key = 'date'
        self.visits = set(self.data['visit'].unique())

    @staticmethod
    def _validated_data(df):
        CalculatorData._validate_columns(df, ('id', 'visit', 'date'), "visit", "id, visit, and date")
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        return df

    def profile_data(self, min_date_cutoff=None, max_date_cutoff=None, expected_visit_order="inferred"):
        """
        Profile the visit data

        :param min_date_cutoff: Union[None, datetime, str], default None
            The minimal date allowed for the visit's date, lower than that is considered to be an outlier
            When it's set None, outlier detection won't consider the lower bound
            When it's str, it should be able to be casted to a datetime object, such as '10/23/2020'

        :param max_date_cutoff: Union[None, datetime, str], default None
            The maximal amount allowed for the consumption, higher than that is considered to be an outlier
            When it's set None, outlier detection won't consider the higher bound
            When it's str, it should be able to be casted to a datetime object, such as '10/23/2020'

        :param expected_visit_order: Union["inferred", list, None], default "inferred"
            The expected order of the visits, such that when dates are out of order can be detected
            The list of visit names should match the actual visits in the dataset
            When None, no checking will be performed
            The default is inferred, and it means that the order of the visits is sorted based on its numeric or
            alphabetic order

        :return: None
        """
        casted_min_date_cutoff = pd.to_datetime(min_date_cutoff, infer_datetime_format=True)
        casted_max_date_cutoff = pd.to_datetime(max_date_cutoff, infer_datetime_format=True)
        visit_summary_series = self._get_visit_sample_summary(casted_min_date_cutoff, casted_max_date_cutoff)
        visit_subject_summary = self._get_visit_subject_summary(casted_min_date_cutoff, casted_max_date_cutoff,
                                                                expected_visit_order)
        return visit_summary_series, visit_subject_summary

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
            visit_dates = self._visit_data.loc[self.data['visit'] == visit, :].\
                sort_values(by=['interval_to_anchor', 'id']).dropna().reset_index(drop=True)
            visit_summary[f"v{visit}_v{anchor_visit}_interval_days_range"] = \
                f"{visit_dates['interval_to_anchor'].min()} - {visit_dates['interval_to_anchor'].max()}"
            visit_summary[f"v{visit}_v{anchor_visit}_interval_days_min5"] = \
                [tuple(x) for x in visit_dates.loc[0:4, ['id', 'interval_to_anchor']].values]
            rows = visit_dates.shape[0]
            visit_summary[f"v{visit}_v{anchor_visit}_interval_days_max5"] = \
                [tuple(x) for x in reversed(visit_dates.loc[rows - 5:rows - 1, ['id', 'interval_to_anchor']].values)]

        visit_record_counts = self.data.groupby(['visit'])['date'].count().to_dict()
        visit_summary.update({f"visit_{visit}_record_count": count for visit, count in visit_record_counts.items()})
        return pd.Series(visit_summary)

    def _get_visit_subject_summary(self, min_date_cutoff, max_date_cutoff, expected_visit_order):
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
            elif isinstance(expected_visit_order, str) and expected_visit_order != 'inferred':
                show_warning("Supported options for expected_visit_order are list of visits, None, and inferred. "
                             "The expected visit order is inferred to check if the dates are in the correct order.")
            sorted_visit_data = self.data.sort_values(by=['id', 'visit']).reset_index()
            sorted_visit_data['ascending'] = sorted_visit_data.groupby(['id'])['date'].diff().map(
                lambda x: True if x is pd.NaT or x.days > 0 else False
            )
            visits_out_of_order = sorted_visit_data.groupby(['id'])['ascending'].all().map(lambda x: not x)
            if visits_out_of_order.sum() > 0:
                show_warning(f"Please note that some subjects (n={visits_out_of_order.sum()}) appear to have their "
                             f"visit dates out of the correct order. You can find out who they are in the visit data"
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

        :return: None
        """
        casted_floor_date = pd.to_datetime(floor_date, infer_datetime_format=True)
        casted_ceil_date = pd.to_datetime(ceil_date, infer_datetime_format=True)
        self.data['date'] = self.data['date'].map(
            partial(CalculatorData._recode_value, casted_floor_date, casted_ceil_date))

    def impute_data(self, impute='freq'):
        """
        Impute any missing visit data.

        :param impute: Union["freq", "mean", None], how the missing visit data are imputed
        None: no imputation
        "freq": use the most frequent, see below for further clarification
        "mean": use the average, see below for further clarification
            Both imputation methods assume that the visit structure is the same among all subjects. It will first find
            the earliest visit as the anchor date, impute any missing visit dates either using the average or the most
            frequent interval. Please note the anchor dates can't be missing.
        """
        if impute is None or str(impute).lower() == "none":
            return
        if impute not in ('freq', 'mean'):
            raise InputArgumentError('You can only specify the imputation method to be "freq" or "mean".')

        self.data = self.data.sort_values(by=self._index_keys, ignore_index=True)
        min_date_indices = self.data.groupby(['id'])['date'].idxmin()
        anchor_visit = self.data.loc[min_date_indices, 'visit'].value_counts().idxmax()
        visit_ids = sorted(self.subject_ids)
        anchor_ids = set(self.data.loc[self.data['visit'] == anchor_visit, 'id'].unique())
        missing_anchor_ids = set(visit_ids) - anchor_ids
        if missing_anchor_ids:
            message = f"Subjects {missing_anchor_ids} are missing anchor visit {anchor_visit}. " \
                      f"There might be problems calculating abstinence data for these subjects."
            show_warning(message)
        ids_s = pd.Series(visit_ids, name='id')
        anchor_dates = self.data.loc[self.data['visit'] == anchor_visit, ['id', 'date']]
        anchor_df = pd.merge(ids_s, anchor_dates, how='outer', on='id')
        df_anchor = anchor_df.copy()
        df_anchor['visit'] = anchor_visit
        df_anchor['imputed_date'] = df_anchor['date']
        df_anchor['anchor_date'] = df_anchor['date']
        imputed_visit_dfs = [df_anchor]
        anchor_df.rename({'date': 'anchor_date'}, axis=1, inplace=True)
        visits = self.visits - {anchor_visit}
        for visit in visits:
            visit_dates = self.data.loc[self.data['visit'] == visit, ['id', 'date']]
            df_visit = pd.merge(anchor_df, visit_dates, how='outer', on='id')
            days_diff = (df_visit['date'] - df_visit['anchor_date']).map(
                lambda day_diff: day_diff.days if pd.notnull(day_diff) else np.nan)
            used_days_diff = days_diff.value_counts().idxmax() if impute == 'freq' else int(days_diff.mean())

            def impute_date(x):
                if pd.notnull(x['date']):
                    return x['date']
                imputed_date = x['anchor_date'] + timedelta(days=used_days_diff)
                print(f"Imputed Date: {imputed_date} for subject: {x['id']} visit {visit}")
                return imputed_date

            df_visit['imputed_date'] = df_visit.apply(impute_date, axis=1)
            df_visit['visit'] = visit
            imputed_visit_dfs.append(df_visit)
        visit_data_imputed = pd.concat(imputed_visit_dfs)
        visit_data_imputed['imputation_code'] = visit_data_imputed['date'].map(
            lambda x: 0 if x is not pd.NaT else 1)
        self.visit_data_imputed = visit_data_imputed.drop(['date', 'anchor_date'], axis=1). \
            rename({'imputed_date': 'date'}, axis=1).sort_values(by=['id', 'visit']).reset_index(drop=True)
        impute_summary = self.visit_data_imputed.groupby(['imputation_code']).size().reset_index(). \
            rename({0: "record_count"}, axis=1)
        impute_summary['imputation_code'] = impute_summary['imputation_code'].map({0: 'Raw', 1: 'Imputed'})

# %%
tlfb_data = TLFBData("smartmod_tlfb_data.csv")
tlfb_sample_profile, tlfb_subject_profile = tlfb_data.profile_data()
tlfb_data.drop_na_records()
tlfb_data.check_duplicates()
tlfb_data.recode_data()
tlfb_data.impute_data()

# visit_data = VisitData("smartmod_visit_data.csv")
# visit_sample_profile, visit_subject_profile = visit_data.profile_data()
# visit_data.drop_na_records()
# visit_data.check_duplicates()
# visit_data.recode_data()


# %%
class TLFBImputationCode(Enum):
    RAW = 0  # raw data, no imputation
    IMPUTED = 1 # imputed value

    # @classmethod
    # def code_for_missing_interval(cls, start_amount, end_amount, abst_cutoff):
    #     imputation_code = cls.UNSPECIFIED
    #     if start_amount <= abst_cutoff and end_amount <= abst_cutoff:
    #         imputation_code = cls.AB_AB
    #     elif start_amount > abst_cutoff and end_amount > abst_cutoff:
    #         imputation_code = cls.NONAB_NONAB
    #     elif start_amount > abst_cutoff >= end_amount:
    #         imputation_code = cls.NONAB_AB
    #     elif start_amount <= abst_cutoff < end_amount:
    #         imputation_code = cls.AB_NONAB
    #     return imputation_code


# %%
class AbstinenceCalculatorError(Exception):
    pass


class FileExtensionError(AbstinenceCalculatorError):
    def __init__(self, file_path):
        self.file_path = file_path

    def __str__(self):
        return f"The file {self.file_path} isn't a tab-delimited text file, CSV or excel file."


class DataSourceQualityError(AbstinenceCalculatorError):
    pass


class FileFormatError(AbstinenceCalculatorError):
    pass


class InputArgumentError(AbstinenceCalculatorError):
    pass


# %%
class AbstinenceCalculator:
    """
    Abstinence calculator for addiction research based on the TLFB and visit data

    Attributes:
        tlfb_filepath (str, pathlib.Path): The file path for the TLFB data
        visit_filepath (str, pathlib.Path): The file path for the visit data
    """
    def __init__(self, tlfb_data: TLFBData, visit_data: VisitData, abst_cutoff=0):
        """
        Create an instance object to calculate abstinence

        :param tlfb_data: TLFBData, the TLFB data

        :param visit_data: VisitData, the visit data

        :param abst_cutoff: Union[int, float], default 0
            The cutoff for abstinence, below or equal to which is considered to be abstinent

        :return: None
        """
        self.tlfb_data = tlfb_data
        self.visit_data = visit_data
        self.abst_cutoff = abst_cutoff

    def __init__(self, tlfb_filepath, visit_filepath, abst_cutoff=0):
        """
        Create an instance object to calculate abstinence

        :param tlfb_filepath: Union[str, Path]
            The filepath for the TLFB dataset, the dataset should have id, date, and amount columns

        :param visit_filepath: Union[str, Path]
            The filepath for the visit dataset, the dataset should have id, visit, and date columns

        :param abst_cutoff: Union[int, float], default 0
            The cutoff for abstinence, below or equal to which is considered to be abstinent

        :return: None
        """
        self.tlfb_data = AbstinenceCalculator._validated_tlfb_data_from_source(tlfb_filepath)
        self._visit_data = AbstinenceCalculator._validated_visit_data_from_source(visit_filepath)
        self.visits = set(self.visit_data['visit'].unique())
        self.tlfb_ids = set(self.tlfb_data['id'].unique())
        self.visit_ids = set(self.visit_data['id'].unique())
        self.abst_cutoff = abst_cutoff
        self.tlfb_data_imputed = None
        self.visit_data_imputed = None
        self._tlfb_duplicates = None
        self._visit_duplicates = None
        self._expected_visit_order = None

    # @property
    # def visit_data(self):
    #     return self._visit_data.loc[:, ['id', 'visit', 'date']]

    @staticmethod
    def _validated_tlfb_data_from_source(tlfb_filepath):
        tlfb_data = CalculatorData.read_data_from_path(tlfb_filepath)
        _validate_columns(tlfb_data, ('id', 'date', 'amount'), "TLFB", "id, date, and amount")
        tlfb_data['date'] = pd.to_datetime(tlfb_data['date'], infer_datetime_format=True)
        tlfb_data['amount'] = tlfb_data['amount'].astype(float)
        return tlfb_data

    @staticmethod
    def _validated_visit_data_from_source(visit_filepath):
        visit_data = CalculatorData.read_data_from_path(visit_filepath)
        _validate_columns(visit_data, ('id', 'visit', 'date'), "visit", "id, visit, and date")
        visit_data['date'] = pd.to_datetime(visit_data['date'], infer_datetime_format=True)
        return visit_data

    def profile_tlfb_data(self, min_amount_cutoff=None, max_amount_cutoff=None):
        """
        Profile the TLFB data
        
        :param min_amount_cutoff: Union[None, int, float], default None
            The minimal amount allowed for the consumption, lower than that is considered to be an outlier
            when it's set None, outlier detection won't consider the lower bound
            
        :param max_amount_cutoff: Union[None, int, float], default None
            The maximal amount allowed for the consumption, higher than that is considered to be an outlier
            when it's set None, outlier detection won't consider the higher bound
            
        :return: None
        """
        tlfb_summary_series = self._get_tlfb_sample_summary(min_amount_cutoff, max_amount_cutoff)
        tlfb_subject_summary = self._get_tlfb_subject_summary(min_amount_cutoff, max_amount_cutoff)
        sns.distplot(self.tlfb_data['amount'])
        plt.show()
        return tlfb_summary_series, tlfb_subject_summary

    def _get_tlfb_sample_summary(self, min_amount_cutoff, max_amount_cutoff):
        tlfb_summary = {'record_count': self.tlfb_data.shape[0]}
        key_names = 'subject_count min_date max_date min_amount max_amount'.split()
        col_names = 'id date date amount amount'.split()
        func_names = 'nunique min max min max'.split()
        for key_name, col_name, func_name in zip(key_names, col_names, func_names):
            func = getattr(pd.Series, func_name)
            tlfb_summary[key_name] = func(self.tlfb_data[col_name])

        sorted_tlfb_data = self.tlfb_data.sort_values(by=['amount'], ascending=False).\
                               reset_index(drop=True).iloc[0:5, :].copy()
        sorted_tlfb_data['date'] = sorted_tlfb_data['date'].dt.strftime('%m/%d/%Y')
        tlfb_summary['max_amount_5_records'] = \
            [tuple(x) for x in sorted_tlfb_data.iloc[0:5, :].values]
        count_key_names = 'missing_subject_count missing_date_count missing_amount_count duplicate_count'.split()
        columns = [self.tlfb_data['id'].isna(),
                   self.tlfb_data['date'].isna(),
                   self.tlfb_data['amount'].isna(),
                   self.tlfb_data.duplicated(['id', 'date'], keep=False)]
        if min_amount_cutoff is not None:
            count_key_names.append('outlier_amount_low_count')
            columns.append(pd.Series(self.tlfb_data['amount'] < min_amount_cutoff))
        if max_amount_cutoff is not None:
            count_key_names.append('outlier_amount_high_count')
            columns.append(pd.Series(self.tlfb_data['amount'] > max_amount_cutoff))
        for key_name, column in zip(count_key_names, columns):
            tlfb_summary[key_name] = column.sum()
        return pd.Series(tlfb_summary)

    def _get_tlfb_subject_summary(self, min_amount_cutoff, max_amount_cutoff):
        tlfb_dates_amounts = self.tlfb_data.groupby('id').agg({
            "date": ["count", "min", "max"],
            "amount": ["min", "max", "mean"]
        })
        summary_dfs = [tlfb_dates_amounts]
        col_names = ['record_count', 'date_min', 'date_max', 'amount_min', 'amount_max',
                     'amount_mean', 'duplicates_count']
        summary_dfs.append(self.tlfb_data.loc[self.tlfb_data.duplicated(['id', 'date'], keep=False), :].groupby(
            'id')['amount'].agg('count'))
        if min_amount_cutoff is not None:
            summary_dfs.append(self.tlfb_data.loc[self.tlfb_data['amount'] < min_amount_cutoff, :].groupby(
                'id')['amount'].agg('count'))
            col_names.append('outliers_low_count')
        if max_amount_cutoff is not None:
            summary_dfs.append(self.tlfb_data.loc[self.tlfb_data['amount'] > max_amount_cutoff, :].groupby(
            'id')['amount'].agg('count'))
            col_names.append('outliers_high_count')
        summary_df = pd.concat(summary_dfs, axis=1)
        summary_df.columns = col_names
        return summary_df.fillna(0)

    def recode_tlfb_abnormality(self, floor_amount=None, ceil_amount=None, duplicate_kept="min"):
        """
        Recode the abnormal data of the TLFB dataset
        :param floor_amount: Union[None, int, float], default None
            Recode values lower than the floor amount to the floor amount.
            When None, no replacement will be performed
        :param ceil_amount: Union[None, int, float], default None
            Recode values higher than the ceil amount to the ceil amount.
            When None, no replacement will be performed
        :param duplicate_kept: str, supported options: "min", "max", "mean", False
            Specify which duplicate record to keep
            When False, no duplicates will be kept
        :return: TLFB data sorted by id and date
        """
        _validate_duplicate_arg_options(duplicate_kept)
        self.tlfb_data = CalculatorData.drop_na_records(self.tlfb_data)
        self.tlfb_data['amount'] = self.tlfb_data['amount'].map(partial(_recode_value, floor_amount, ceil_amount))
        self._tlfb_duplicates, self.tlfb_data = \
            AbstinenceCalculator._drop_duplicates(self.tlfb_data, ['id', 'date'], 'amount',
                                                  duplicate_kept, 'get_tlfb_duplicates')

    def profile_visit_data(self, min_date_cutoff=None, max_date_cutoff=None, expected_visit_order="inferred"):
        """
        Profile the visit data

        :param min_date_cutoff: Union[None, datetime, str], default None
            The minimal date allowed for the visit's date, lower than that is considered to be an outlier
            When it's set None, outlier detection won't consider the lower bound
            When it's str, it should be able to be casted to a datetime object, such as '10/23/2020'

        :param max_date_cutoff: Union[None, datetime, str], default None
            The maximal amount allowed for the consumption, higher than that is considered to be an outlier
            When it's set None, outlier detection won't consider the higher bound
            When it's str, it should be able to be casted to a datetime object, such as '10/23/2020'

        :param expected_visit_order: Union["inferred", list, None], default "inferred"
            The expected order of the visits, such that when dates are out of order can be detected
            The list of visit names should match the actual visits in the dataset
            When None, no checking will be performed
            The default is inferred, and it means that the order of the visits is sorted based on its numeric or
            alphabetic order

        :return: None
        """
        casted_min_date_cutoff = pd.to_datetime(min_date_cutoff, infer_datetime_format=True)
        casted_max_date_cutoff = pd.to_datetime(max_date_cutoff, infer_datetime_format=True)
        visit_summary_series = self._get_visit_sample_summary(casted_min_date_cutoff, casted_max_date_cutoff)
        visit_subject_summary = self._get_visit_subject_summary(casted_min_date_cutoff, casted_max_date_cutoff,
                                                                expected_visit_order)
        return visit_summary_series, visit_subject_summary

    def _get_visit_sample_summary(self, min_date_cutoff, max_date_cutoff):
        visit_summary = {'record_count': self.visit_data.shape[0]}
        key_names = 'subject_count visit_count distinct_visits min_date max_date'.split()
        col_names = 'id visit visit date date'.split()
        func_names = 'nunique nunique unique min max'.split()
        for key_name, col_name, func_name in zip(key_names, col_names, func_names):
            func = getattr(pd.Series, func_name)
            visit_summary[key_name] = func(self.visit_data[col_name])

        count_key_names = 'missing_subject_count missing_visit_count missing_date_count duplicate_count'.split()
        columns = [self.visit_data['id'].isna(),
                   self.visit_data['visit'].isna(),
                   self.visit_data['date'].isna(),
                   self.visit_data.duplicated(['id', 'visit'], keep=False)]
        if min_date_cutoff is not None:
            count_key_names.append('outlier_date_low_count')
            columns.append(pd.Series(self.visit_data['date'] < min_date_cutoff))
        if max_date_cutoff is not None:
            count_key_names.append('outlier_date_high_count')
            columns.append(pd.Series(self.visit_data['date'] > max_date_cutoff))
        for key_name, column in zip(count_key_names, columns):
            visit_summary[key_name] = column.sum()

        min_date_indices = self.visit_data.groupby(['id'])['date'].idxmin()
        anchor_visit = self.visit_data.loc[min_date_indices, 'visit'].value_counts().idxmax()
        visit_summary['anchor_visit'] = anchor_visit
        anchor_dates = self.visit_data.loc[self.visit_data['visit'] == anchor_visit, ['id', 'date']].copy().\
            rename({'date': 'anchor_date'}, axis=1)
        visit_data_anchored = pd.merge(self.visit_data, anchor_dates, on='id')
        visit_data_anchored['interval_to_anchor'] = (visit_data_anchored['date'] - visit_data_anchored['anchor_date']).\
            map(lambda x: x.days)
        self._visit_data = visit_data_anchored.drop(['anchor_date'], axis=1)

        grid = sns.FacetGrid(self._visit_data.loc[self._visit_data['visit'] != anchor_visit, :], col="visit")
        grid.map(plt.hist, "interval_to_anchor")
        plt.show()

        for visit in self.visits - {anchor_visit}:
            visit_dates = self._visit_data.loc[self.visit_data['visit'] == visit, :].\
                sort_values(by=['interval_to_anchor', 'id']).dropna().reset_index(drop=True)
            visit_summary[f"v{visit}_v{anchor_visit}_interval_days_range"] = \
                f"{visit_dates['interval_to_anchor'].min()} - {visit_dates['interval_to_anchor'].max()}"
            visit_summary[f"v{visit}_v{anchor_visit}_interval_days_min5"] = \
                [tuple(x) for x in visit_dates.loc[0:4, ['id', 'interval_to_anchor']].values]
            rows = visit_dates.shape[0]
            visit_summary[f"v{visit}_v{anchor_visit}_interval_days_max5"] = \
                [tuple(x) for x in reversed(visit_dates.loc[rows - 5:rows - 1, ['id', 'interval_to_anchor']].values)]

        visit_record_counts = self.visit_data.groupby(['visit'])['date'].count().to_dict()
        visit_summary.update({f"visit_{visit}_record_count": count for visit, count in visit_record_counts.items()})
        return pd.Series(visit_summary)

    def _get_visit_subject_summary(self, min_date_cutoff, max_date_cutoff, expected_visit_order):
        visit_dates_amounts = self.visit_data.groupby('id').agg({
            "date": ["count", "min", "max"]
        })
        visit_dates_amounts['date_interval'] = \
            (visit_dates_amounts[('date', 'max')] - visit_dates_amounts[('date', 'min')]).map(lambda x: x.days)
        summary_dfs = [visit_dates_amounts]
        col_names = ['record_count', 'date_min', 'date_max', 'date_interval', 'duplicates_count']
        summary_dfs.append(self.visit_data.loc[self.visit_data.duplicated(['id', 'visit'], keep=False), :].groupby(
            'id')['date'].agg('count'))
        if min_date_cutoff is not None:
            summary_dfs.append(self.visit_data.loc[self.visit_data['date'] < min_date_cutoff, :].groupby(
                'id')['date'].agg('count'))
            col_names.append('outliers_date_low_count')
        if max_date_cutoff is not None:
            summary_dfs.append(self.visit_data.loc[self.visit_data['date'] > max_date_cutoff, :].groupby(
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
                self.visit_data['visit'] = self.visit_data['visit'].map(
                    {visit: i for i, visit in enumerate(expected_visit_order)})
            elif isinstance(expected_visit_order, str) and expected_visit_order != 'inferred':
                show_warning("Supported options for expected_visit_order are list of visits, None, and inferred. "
                             "The expected visit order is inferred to check if the dates are in the correct order.")
            sorted_visit_data = self.visit_data.sort_values(by=['id', 'visit']).reset_index()
            sorted_visit_data['ascending'] = sorted_visit_data.groupby(['id'])['date'].diff().map(
                lambda x: True if x is pd.NaT or x.days > 0 else False
            )
            visits_out_of_order = sorted_visit_data.groupby(['id'])['ascending'].all().map(lambda x: not x)
            if visits_out_of_order.sum() > 0:
                show_warning(f"Please note that some subjects (n={visits_out_of_order.sum()}) appear to have their "
                             f"visit dates out of the correct order. You can find out who they are in the visit data"
                             f"summary by subject. Please fix them if applicable.")
            return visits_out_of_order

    def recode_visit_abnormality(self, floor_date=None, ceil_date=None, duplicate_kept="min"):
        """
        Recode the abnormal data of the TLFB dataset
        :param floor_date: Union[None, int, float], default None
            Recode values lower than the floor amount to the floor amount.
            When None, no replacement will be performed
        :param ceil_date: Union[None, int, float], default None
            Recode values higher than the ceil amount to the ceil amount.
            When None, no replacement will be performed
        :param duplicate_kept: str, supported options: "min", "max", "mean", False
            Specify which duplicate record to keep
            When False, no duplicates will be kept
        :return: None
        """
        _validate_duplicate_arg_options(duplicate_kept)
        self._visit_data = _drop_na_records(self._visit_data)

        casted_floor_date = pd.to_datetime(floor_date, infer_datetime_format=True)
        casted_ceil_date = pd.to_datetime(ceil_date, infer_datetime_format=True)
        self.visit_data['date'] = self.visit_data['date'].map(
            partial(_recode_value, casted_floor_date, casted_ceil_date)
        )
        self._visit_duplicates, self._visit_data = \
            AbstinenceCalculator._drop_duplicates(self.visit_data, ['id', 'visit'], 'date',
                                                  duplicate_kept, 'get_visit_duplicates')



    def cross_validate_tlfb_visit_data(self):
        tlfb_ids = pd.Series({subject_id: 'Yes' for subject_id in self.tlfb_data['id'].unique()}, name='TLFB Data')
        visit_ids = pd.Series({subject_id: 'Yes' for subject_id in self.visit_data['id'].unique()}, name='Visit Data')
        crossed_data = pd.concat([tlfb_ids, visit_ids], axis=1).fillna('No')
        freq_data = crossed_data.groupby(['TLFB Data', 'Visit Data']).size().reset_index().\
            rename({0: "subject_count"}, axis=1)
        return freq_data

    def impute_tlfb_data(self, impute="linear", show_impute_summary=True):
        """
        Impute the TLFB data

        :param impute: Union["uniform", "linear", None, int, float], how the missing TLFB data are imputed
            1. None: no imputation
            2. "uniform": impute the missing TLFB data using the mean value of the amounts before and after
            the missing interval
            3. "linear" (the default): impute the missing TLFB data by interpolating a linear trend based on the amounts before and after
            the missing interval
            4. Numeric value (int or float): impute the missing TLFB data using the specified value
        :param show_impute_summary: bool, whether show the imputation summary report
        """
        if impute is None or str(impute).lower() == "none":
            return
        if not (impute in ("uniform", "linear") or impute.isnumeric()):
            raise InputArgumentError("The imputation mode can only be None, 'uniform', 'linear', "
                                     "or a numeric value.")
        self.tlfb_data = self.tlfb_data.sort_values(by=['id', 'date']).reset_index(drop=True)
        tlfb_df = self.tlfb_data.copy()
        tlfb_df['imputation_code'] = TLFBImputationCode.RAW.value
        missing_data = AbstinenceCalculator._get_tlfb_missing_data(tlfb_df)
        if not len(missing_data):
            self.tlfb_data_imputed = tlfb_df
            show_warning("There is no missing TLFB data, and thus no imputation was performed.")
            return self.tlfb_data_imputed
        imputed_records = []
        for row in missing_data.itertuples():
            start_data = tlfb_df.iloc[row.Index - 1]
            start_record = TLFBRecord(start_data.id, start_data.date, start_data.amount, start_data.imputation_code)
            end_record = TLFBRecord(row.id, row.date, row.amount, start_data.imputation_code)
            imputed_records.extend(self._impute_tlfb_missing_block(start_record, end_record, impute=impute))
        tlfb_df.drop(['diff_days'], axis=1, inplace=True)
        imputed_tlfb_data = pd.DataFrame(imputed_records)
        self.tlfb_data_imputed = pd.concat([tlfb_df, imputed_tlfb_data]). \
            sort_values(['id', 'date']).reset_index(drop=True)
        if show_impute_summary:
            impute_summary = self.tlfb_data_imputed.groupby(['imputation_code']).size().reset_index().\
                rename({0: "record_count"}, axis=1)
            impute_summary['imputation_code'] = impute_summary['imputation_code'].map(
                lambda x: TLFBImputationCode(x).name
            )
        else:
            message = f"The number of imputed records: {len(imputed_records)}"
            show_warning(message)

    @staticmethod
    def _get_tlfb_missing_data(tlfb_df):
        tlfb_df['diff_days'] = tlfb_df.groupby(['id'])['date'].diff().map(
            lambda x: x.days if pd.notnull(x) else 1)
        return tlfb_df[tlfb_df['diff_days'] > 1.0]

    def _impute_tlfb_missing_block(self, start_record: TLFBRecord, end_record: TLFBRecord, impute):
        subject_id, start_date, start_amount, _ = start_record
        subject_id, end_date, end_amount, _ = end_record
        imputation_code = TLFBImputationCode.code_for_missing_interval(start_amount, end_amount, self.abst_cutoff).value
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

    # noinspection PyTypeChecker
    def impute_visit_data(self, impute='freq', show_impute_summary=True):
        """
        Impute any missing visit data.
        :param impute: Union["freq", "mean", None], how the missing visit data are imputed
        1. None: no imputation
        2. "freq"=use the most frequent, see below for further clarification
        3. "mean"=use the average, see below for further clarification
            Both imputation methods assume that the visit structure is the same among all subjects. It will first find
            the earliest visit as the anchor date, impute any missing visit dates either using the average or the most
            frequent interval. Please note the anchor dates can't be missing.
        :param show_impute_summary: bool, whether show the imputation summary report, default=True
        """
        if impute is None or str(impute).lower() == "none":
            return
        if impute not in ('freq', 'mean'):
            raise InputArgumentError('You can only specify the imputation method to be "freq" or "mean".')
        self.visit_data = self.visit_data.sort_values(by=['id', 'visit']).reset_index(drop=True)
        min_date_indices = self.visit_data.groupby(['id'])['date'].idxmin()
        anchor_visit = self.visit_data.loc[min_date_indices, 'visit'].value_counts().idxmax()
        visit_ids = sorted(self.visit_ids)
        anchor_ids = set(self.visit_data.loc[self.visit_data['visit'] == anchor_visit, 'id'].unique())
        missing_anchor_ids = set(visit_ids) - anchor_ids
        if missing_anchor_ids:
            message = f"Subjects {missing_anchor_ids} are missing anchor visit {anchor_visit}. " \
                      f"There might be problems calculating abstinence data for these subjects."
            show_warning(message)
        ids_s = pd.Series(visit_ids, name='id')
        anchor_dates = self.visit_data.loc[self.visit_data['visit'] == anchor_visit, ['id', 'date']]
        anchor_df = pd.merge(ids_s, anchor_dates, how='outer', on='id')
        df_anchor = anchor_df.copy()
        df_anchor['visit'] = anchor_visit
        df_anchor['imputed_date'] = df_anchor['date']
        df_anchor['anchor_date'] = df_anchor['date']
        imputed_visit_dfs = [df_anchor]
        anchor_df.rename({'date': 'anchor_date'}, axis=1, inplace=True)
        visits = self.visits - {anchor_visit}
        for visit in visits:
            visit_dates = self.visit_data.loc[self.visit_data['visit'] == visit, ['id', 'date']]
            df_visit = pd.merge(anchor_df, visit_dates, how='outer', on='id')
            days_diff = (df_visit['date'] - df_visit['anchor_date']).map(
                lambda day_diff: day_diff.days if pd.notnull(day_diff) else np.nan)
            used_days_diff = days_diff.value_counts().idxmax() if impute == 'freq' else int(days_diff.mean())

            def impute_date(x):
                if pd.notnull(x['date']):
                    return x['date']
                imputed_date = x['anchor_date'] + timedelta(days=used_days_diff)
                print(f"Imputed Date: {imputed_date} for subject: {x['id']} visit {visit}")
                return imputed_date

            df_visit['imputed_date'] = df_visit.apply(impute_date, axis=1)
            df_visit['visit'] = visit
            imputed_visit_dfs.append(df_visit)
        visit_data_imputed = pd.concat(imputed_visit_dfs)
        visit_data_imputed['imputation_code'] = visit_data_imputed['date'].map(
            lambda x: 0 if x is not pd.NaT else 1)
        self.visit_data_imputed = visit_data_imputed.drop(['date', 'anchor_date'], axis=1). \
            rename({'imputed_date': 'date'}, axis=1).sort_values(by=['id', 'visit']).reset_index(drop=True)
        print(self.visit_data_imputed.loc[self.visit_data_imputed['id'] == 421244, :])
        if show_impute_summary:
            impute_summary = self.visit_data_imputed.groupby(['imputation_code']).size().reset_index(). \
                rename({0: "record_count"}, axis=1)
            impute_summary['imputation_code'] = impute_summary['imputation_code'].map({0: 'Raw', 1: 'Imputed'})
        else:
            message = f"The number of imputed records: {visit_data_imputed['imputation_code'].sum()}"
            show_warning(message)

    @staticmethod
    def _format_visits_names(end_visits, abst_var_names, prefix=''):
        if not isinstance(end_visits, list):
            end_visits = [end_visits]
        if abst_var_names == 'inferred':
            abst_names = [f"{prefix}_v{end_visit}" for end_visit in end_visits]
        elif isinstance(abst_var_names, list):
            abst_names = abst_var_names
        else:
            abst_names = [abst_var_names]
        if len(end_visits) != len(abst_names):
            raise InputArgumentError("The number of abstinence variable names should match the number of visits.")
        return end_visits, abst_names

    def _validate_visits(self, visits_to_check):
        print('visits_to_check:', visits_to_check)
        print('self.visits:', self.visits)
        if not set(visits_to_check).issubset(self.visits):
            raise InputArgumentError(f"Some visits are not in the visit list {self.visits}. "
                                     f"Please check your arguments.")

    def abstinence_cont(self, start_visit, end_visits, abst_var_names='inferred', including_end=False, mode="itt"):
        """
        Calculates the continuous abstinence for the time window.
        :param start_visit: The name of the visit where the time window starts
        :param end_visits: The name(s) of the visits for the end(s) of the time window, a list of the visits or
        a single visit
        :param abst_var_names: The name(s) of the abstinence variable(s), by default, the name(s) will be inferred,
        if not inferred, the number of abstinence variable names should match the number of end visits
        :param including_end: Whether you want to include the end visit or not, default=False
        :param mode: How you want to calculate the abstinence, "itt"=intention-to-treat (the default) or
        "ro"=responders-only
        :return: Pandas DataFrame, subject id and abstinence results
        """
        end_visits, abst_names = AbstinenceCalculator._format_visits_names(end_visits, abst_var_names,
                                                                           f'abst_cont_v{start_visit}')
        self._validate_visits([start_visit, *end_visits])
        results = []
        for subject_id in sorted(self.tlfb_ids):
            result = [subject_id] + [np.nan for _ in end_visits]
            if subject_id not in self.visit_ids:
                results.append(result)
                continue
            start_date = self._get_visit_dates(subject_id, start_visit, mode=mode)
            end_dates = self._get_visit_dates(subject_id, end_visits, including_end, mode=mode)
            for date_i, end_date in enumerate(end_dates, 1):
                if start_date >= end_date:
                    show_warning(f"The end date of the time window for the subject {subject_id} {end_date} isn't later "
                                 f"than the start date {start_date}. Please verify that the visit dates are correct.")
                    abstinent = np.nan
                elif pd.NaT in (start_date, end_date):
                    show_warning(f"Subject {subject_id} is missing some of the date information.")
                    abstinent = np.nan
                else:
                    subject_data = self._get_subject_data(subject_id, start_date, end_date, mode)
                    abstinent = AbstinenceCalculator._continuous_abst(subject_data, start_date, end_date,
                                                                      self.abst_cutoff)
                result[date_i] = abstinent
            results.append(result)
        col_names = ['id', *abst_names]
        return pd.DataFrame(results, columns=col_names)

    def abstinence_pp(self, end_visits, days, abst_var_names='inferred', including_anchor=False, mode="itt"):
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
        if not isinstance(days, (list, tuple)):
            days = [days]
        all_abst_names = []
        for i, day in enumerate(days, 0):
            end_visits, abst_names = AbstinenceCalculator._format_visits_names(end_visits, abst_var_names,
                                                                               f'abst_pp{day}')
            all_abst_names += abst_names
        results = []
        for i, subject_id in enumerate(self.tlfb_ids):
            end_dates = self._get_visit_dates(subject_id, end_visits, including_anchor, mode=mode)
            result = [subject_id]
            for end_date in end_dates:
                for day in days:
                    start_date = end_date + timedelta(days=-day)
                    subject_data = self._get_subject_data(subject_id, start_date, end_date, mode)
                    abstinent = AbstinenceCalculator._continuous_abst(subject_data, start_date, end_date, self.abst_cutoff)
                    result.append(abstinent)
            results.append(result)
        col_names = all_abst_names.insert(0, 'id')
        return pd.DataFrame(results, columns=col_names)

    def abstinence_prolonged(self, quit_visit, end_visits, lapse_allowed, lapse_criterion, grace_days=14,
                             abst_name='inferred', including_anchor=False, mode="itt"):
        """
        Calculate the prolonged abstinence using the time window
        :param quit_visit: The visit when the subjects are scheduled to quit smoking
        :param end_visits: The reference visit on which the abstinence is to be calculated
        :param lapse_allowed: Whether a lapse is allowed
        :param lapse_criterion: The criterion for finding a lapse; a lapse is only examined in the itme window between
        the date when the grace period is over and the end date
        Supported criteria:
            1). Use amounts (e.g., "5 cigs", "4 drinks")
            2). Use days (e.g., "5 days", "7 days"), you have to end your argument with "days"
            3). Use amounts over a period of time (e.g., "5 cigs/7 days", the time interval is rolling)
            4). Use days over a period of time (e.g., "5 days/7 days", the time interval is rolling)
            5). Combinations of any of these in a tuple, such as ("5 cigs", "2 days/7 days")
        :param grace_days: The number of days for the grace period following the quit date
        :param abst_name: The name of the abstinence variable, by default, the name will be inferred
        :param including_anchor: Whether you want to include the anchor visit or not, default=False
        :param mode: How you want to calculate the abstinence, "itt"=intention-to-treat (the default) or
        "ro"=responders-only
        :return: Pandas DataFrame with two columns, subject id and abstinence result
        """
        criteria = None
        if lapse_allowed:
            criteria = lapse_criterion if isinstance(lapse_criterion, tuple) else (lapse_criterion,)
            for criterion in criteria:
                parsed_criterion_len = len([x for parts in criterion.split("/") for x in parts.split()])
                if parsed_criterion_len not in (2, 4):
                    raise InputArgumentError("When lapse is allowed, you have to specify the criterion for lapses in "
                                             "strings, such as '5 cigs', '5 drinks'. To see the full list of supported"
                                             "criteria, please refer to the help menu.")

        results = []
        for i, subject_id in enumerate(self.tlfb_ids):
            result = [subject_id]
            if subject_id not in self.visit_ids:
                results.append()
            start_date = self._get_visit_dates(subject_id, quit_visit, grace_days, mode=mode)
            end_dates = self._get_visit_dates(subject_id, end_visits, including_anchor, mode=mode)
            max_end_date = max(end_dates)
            _subject_data = self._get_subject_data(subject_id, start_date, max_end_date, mode)
            for end_date in end_dates:
                subject_data = _subject_data[(start_date <= _subject_data['date']) &
                                             (_subject_data['date'] < end_date)]
                days_to_check = int((end_date - start_date).days)
                abstinent = None
                if subject_data['amount'].count() != days_to_check:
                    show_warning(f"There are fewer data records for subject {subject_id} than the days examined.")
                else:
                    if not lapse_allowed:
                        abstinent = int((subject_data['amount'] <= self.abst_cutoff).all())
                    else:
                        for criterion in criteria:
                            parsed_criterion = [x for parts in criterion.split("/") for x in parts.split()]
                            if len(parsed_criterion) == 2 and parsed_criterion[-1] != "days":
                                abstinent = sum(subject_data["amount"]) < float(parsed_criterion[0])
                            elif len(parsed_criterion) == 2 and parsed_criterion[-1] == "days":
                                abstinent = sum(subject_data["amount"] > self.abst_cutoff) < float(parsed_criterion[0])
                            else:
                                assert len(parsed_criterion) == 4
                                cutoff_amount, cutoff_unit, window_amount, _ = parsed_criterion
                                cutoff_amount = int(float(cutoff_amount))
                                window_amount = int(float(window_amount))
                                index_list = subject_data.index[subject_data['amount'] > self.abst_cutoff].tolist()
                                if not index_list:
                                    for j in index_list:
                                        one_week = range(j, j + window_amount)
                                        if cutoff_unit == "days":
                                            lapsed = sum(elem in index_list for elem in one_week) > cutoff_amount
                                        else:
                                            lapsed = sum(subject_data.loc[one_week, "amount"]) > cutoff_amount
                                        if lapsed:
                                            abstinent = 0
                                            break
                                else:
                                    abstinent = 1
                result.append(abstinent)
            results.append(result)
        return pd.DataFrame.from_dict({'id': self.tlfb_ids, abst_name: results})

    def _get_subject_data(self, subject_id, start_date, end_date, mode):
        df = self.tlfb_data if mode == "itt" else self.tlfb_data_imputed
        subject_data = df[(df['id'] == subject_id) &
                          (start_date <= df['date']) &
                          (df['date'] < end_date)]
        return subject_data

    def _get_visit_dates(self, subject_id, visit_names, increment_days=0, mode='itt'):
        visit_data = self.visit_data_imputed if mode == 'itt' else self.visit_data
        if not isinstance(visit_names, list):
            visit_names = [visit_names]
        visit_dates_cond = (visit_data['id'] == subject_id) & (visit_data['visit'].isin(visit_names))
        dates = list(visit_data.loc[visit_dates_cond, 'date'])
        if increment_days:
            dates = [date + timedelta(days=increment_days) for date in dates]
        return dates[0] if len(dates) == 1 else dates

    @staticmethod
    def _continuous_abst(subject_data, start_date, end_date, abst_cutoff):
        days_to_check = int((end_date - start_date).days)
        subset = subject_data[(start_date <= subject_data['date']) &
                              (subject_data['date'] < end_date)]
        return int((subset['amount'].count() == days_to_check) & (subset['amount'] <= abst_cutoff).all())

    @staticmethod
    def _increment_days(dates, days=1):
        return [date + timedelta(days=days) for date in dates]

    @classmethod
    def read_data(cls, tlfb_filepath, visit_filepath, impute_tlfb="linear", impute_visit="mean", abst_cutoff=0):
        """
        Create an instance object by reading and processing data in a single combined step

        :param tlfb_filepath: filepath for the TLFB data in the tabular format, str or pathlib.Path
        :param visit_filepath: filepath for the visit data in the tabular format, str or pathlib.Path
        :param impute_tlfb:
        :param impute_visit:
        :param abst_cutoff: The cutoff of abstinence, default=0, inclusive, for example if the cutoff is 0.1 then 0.1
        and below is considered abstinent
        :return: an instance object for AbstinenceCalculator class
        """
        calculator = cls(tlfb_filepath, visit_filepath)
        calculator.abst_cutoff = abst_cutoff
        calculator.prepare_tlfb_data(impute_tlfb)
        calculator.prepare_visit_data(impute_visit)
        return calculator

    def prepare_tlfb_data(self, impute="uniform"):
        """
        Prepare the TLFB data to make them ready for abstinence calculation
        :param impute: How the missing TLFB data are imputed, None means no imputations.
        Supported options:
        1. None: no imputation
        2. "uniform" (the default): impute the missing TLFB data using the mean value of the amounts before and after
        the missing interval
        3. "linear": impute the missing TLFB data by interpolating a linear trend based on the amounts before and after
        the missing interval
        4. Numeric value: impute the missing TLFB data using a fixed value
        """
        self.tlfb_ids = self.tlfb_data['id'].unique()
        self.remove_tlfb_duplicates()
        self.sort_tlfb_data()
        self.impute_tlfb_data(impute)

    def remove_tlfb_duplicates(self):
        self._tlfb_duplicates = self.tlfb_data.loc[self.tlfb_data.duplicated(['id', 'date'], keep=False), :].copy()
        self._tlfb_duplicates.sort_values(by=['id', 'date'], inplace=True)
        if len(self._tlfb_duplicates) > 0:
            message = f"""The TLFB data have {len(self._tlfb_duplicates)} duplicates based on id and date. Extra 
            duplicates will be removed in the calculation. To get the duplicates, use get_TLFB_duplicates()."""
            show_warning(message)
            self.tlfb_data.drop_duplicates(["id", "date"], inplace=True)

    def sort_tlfb_data(self):
        self.tlfb_data = self.tlfb_data.sort_values(by=["id", "date"]).reset_index(drop=True)

    def get_tlfb_duplicates(self):
        return self._tlfb_duplicates

    def prepare_visit_data(self, impute="freq"):
        """
        Prepare the visit data ready for calculation.
        :param impute: how the missing visits are imputed, None=no imputation, "freq"=use the most frequent,
                        "mean"=use the average. Both imputation methods assume that the visit structure is the same
                        among all subjects. It will first find the earliest visit as the anchor date, impute any missing
                        visit dates either using the average or the most frequent interval. Please note the anchor dates
                        can't be missing.
        """
        self.remove_visit_duplicates_if_applicable()
        self.impute_visit_data(impute)

    def remove_visit_duplicates_if_applicable(self):
        self._visit_duplicates = self.visit_data.loc[self.visit_data.duplicated(['id', 'visit'], keep=False), :].copy()
        self._visit_duplicates.sort_values(by=['id', 'visit'], inplace=True)
        if len(self._visit_duplicates) > 0:
            message = f"""The visits data have {len(self._visit_duplicates)} duplicates based on id and visit. Extra 
            duplicates will be removed in the calculation. To get the duplicates, use get_visit_duplicates()."""
            show_warning(message)
            self.visit_data.drop_duplicates(['id', 'visit'], inplace=True)

    def get_visit_duplicates(self):
        return self._visit_duplicates

    def sort_visit_data(self):
        self.visit_data = self.visit_data.sort_values(by=["id", "visit"]).reset_index(drop=True)

    @staticmethod
    def show_report(titles, dfs):
        """
        Show the report in the console

        :param titles: Union[str, list]
            The list of titles for each of the DataFrames
            When you want to print just one DataFrame, the titles can be a string
        :param dfs: Union[str, list]
            The list of DataFrames for each of the titles
            When you want to print just one DataFrame, the dfs can be a single DataFrame
        :return: None
        """
        if isinstance(titles, list):
            titles = [titles]
        if isinstance(dfs, list):
            dfs = [dfs]
        if len(titles) != len(dfs):
            raise InputArgumentError("The number of titles need to match the number of DataFrames.")

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        for title, df in zip(titles, dfs):
            print(f"{'*' * 80}\n{title:^80}\n{'*' * 80}")
            print(df)
        pd.reset_option('max_columns')
        pd.reset_option('max_rows')







# %%
abst_cal = AbstinenceCalculator("smartmod_tlfb_data.csv", "smartmod_visit_data.csv")
abst_cal.profile_visit_data()
abst_cal.recode_visit_abnormality()
abst_cal.profile_visit_data()




# abst_cal.profile_tlfb_data(0, 50)

####
# If apply ceil and floor date, special handling should be considered
####
# abst_cal.profile_visit_data("12/01/2000", None,
#                             sample_summary_filename='visit_sample_summary.csv',
#                             subject_summary_filename='visit_subject_summary.csv')
# abst_cal.recode_tlfb_abnormality(10, 50, "mean")
# abst_cal.recode_visit_abnormality("12/01/2000", None, "min")
# abst_cal.impute_tlfb_data(show_impute_summary=False)
# abst_cal.impute_visit_data(show_impute_summary=False)
# abst_cal.abstinence_cont(3, [4, 5]).to_csv('continuous.csv')
# abst_cal.visit_data_imputed.to_csv('visit_data_imputed.csv')

# df = pd.read_csv('smartmod_visit_data.csv')
# df.loc[df['visit'] != 0, :].to_csv('smartmod_visit_data.csv', index=False)