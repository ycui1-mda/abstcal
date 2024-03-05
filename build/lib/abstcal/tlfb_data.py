"""
TLFBData
---------
A data model for processing timeline followback data in the abstinence calculation
"""

from collections import namedtuple
from datetime import timedelta
from abstcal.calculator_error import InputArgumentError
from abstcal.calculator_data import CalculatorData, DataImputationCode
from abstcal.abstcal_utils import read_data_from_path
import pandas as pd
import numpy as np
import seaborn as sns

TLFBRecord = namedtuple("TLFBRecord", "id date amount imputation_code")


class TLFBData(CalculatorData):
    _index_keys = ['id', 'date']
    _value_key = 'amount'

    def __init__(self, filepath, abst_cutoff=0, included_subjects="all", use_raw_date=True):
        """
        Create the instance object for the TLFB data

        :param filepath: Union[str, path, DataFrame], the file path to the TLFB data or the created DataFrame

        :param abst_cutoff: Union[float, int], the cutoff equal to or below which is abstinent

        :param included_subjects: Union[list, tuple], the list of subject ids that are included in the dataset,
            the default option "all" means that all subjects in the dataset will be used

        :param use_raw_date: bool, whether the raw date is used in the date column (default: True)

        """
        self.use_raw_date = use_raw_date
        df = read_data_from_path(filepath)
        self.data = self.validate_data(df)
        if included_subjects and included_subjects != "all":
            self.data = self.data.loc[self.data["id"].isin(included_subjects), :].reset_index(drop=True)

        self.subject_ids = set(self.data['id'].unique())
        self.abst_cutoff = abst_cutoff

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
        grid = sns.displot(self.data['amount'])
        return tlfb_summary_series, tlfb_subject_summary, grid

    def _get_tlfb_sample_summary(self, min_amount_cutoff, max_amount_cutoff):
        record_count = self.data.shape[0]
        subject_count = self.data['id'].nunique()
        record_counts = self.data.groupby('id').size()
        record_count_min, record_count_max = record_counts.min(), record_counts.max()
        tlfb_summary = {'record_count': record_count,
                        'subject_count': subject_count,
                        'records_per_subject_mean': round(record_count / subject_count, 2),
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
                   self.data.duplicated(self._index_keys, keep=False)]
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
        summary_dfs.append(self.data.loc[self.data.duplicated(self._index_keys, keep=False), :].groupby(
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

        :return: The number of interpolated records
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
                    if interpolated_date in interpolated_dates or current_amount <= self.abst_cutoff:
                        continue
                    interpolated_amount = pow(2, day / half_life_in_days) * current_amount
                    interpolated_record = \
                        [subject_id, interpolated_date, interpolated_amount, DataImputationCode.IMPUTED.value]
                    interpolated_dates.add(interpolated_date)
                    interpolated_records.append(interpolated_record)

        interpolated_df = pd.DataFrame(interpolated_records, columns=[*self._index_keys, 'amount', 'imputation_code'])
        self.data = pd.concat([self.data, interpolated_df]).sort_values(by=self._index_keys, ignore_index=True)
        return len(interpolated_df)

    def recode_outliers(self, floor_amount, ceil_amount, drop_outliers=True):
        """
        Recode the abnormal data of the TLFB dataset

        :param floor_amount: Union[int, float]
            Drop records when their values are lower than the floor amount if drop_outliers is True (the default),
            otherwise outliers will be replaced with the floor amount (i.e., drop_outliers=False)

        :param ceil_amount: Union[int, float]
            Drop records when their values are higher than the ceil amount if drop_outliers is True (the default),
            otherwise outliers will be replaced with the ceil amount (i.e., drop_outliers=False)

        :param drop_outliers: bool, default is True
            Drop outliers when it's True and recode outliers to bounding values when it's False

        :return: Summary of the recoding
        """
        recode_summary = dict()
        if floor_amount is not None:
            outlier_count_low = int(pd.Series(self.data['amount'] < floor_amount).sum())
            recode_summary[f'Number of outliers (< {floor_amount})'] = outlier_count_low
            self.data['amount'] = self.data['amount'].map(
                lambda x: np.nan if drop_outliers and x < floor_amount else max(x, floor_amount))
        else:
            recode_summary[f'Number of outliers below amount (not set)'] = "No outliers were requested"
        if ceil_amount is not None:
            outlier_count_high = int(pd.Series(self.data['amount'] > ceil_amount).sum())
            recode_summary[f'Number of outliers (> {ceil_amount})'] = outlier_count_high
            self.data['amount'] = self.data['amount'].map(
                lambda x: np.nan if drop_outliers and x > ceil_amount else min(x, ceil_amount))
        else:
            recode_summary[f'Number of outliers above amount (not set)'] = "No outliers were requested"
        if drop_outliers:
            self.drop_na_records()
        return recode_summary

    # noinspection PyTypeChecker
    def impute_data(self, impute="linear", last_record_action="ffill", maximum_allowed_gap_days=None,
                    biochemical_data=None, overridden_amount="infer"):
        """
        Impute the TLFB data

        :param impute: Union["uniform", "linear", None, int, float], how the missing TLFB data are imputed
            None: no imputation
            "uniform": impute the missing TLFB data using the mean value of the amounts before and after
            the missing interval
            "linear" (the default): impute the missing TLFB data by interpolating a linear trend based on the amounts
            before and after the missing interval
            Numeric value (int or float): impute the missing TLFB data using the specified value

        :param last_record_action: Union[int, float, "ffill", None]
            to interpolate one more record from the last record (presumably the day before the last visit), this action
            is useful when you compute abstinence data involving the last visit, which may have missing data on the TLFB
            data.
            "ffill" (the default): generate one more record with the same amount of substance use as the last record
            None: no actions with the last records
            int, float: a numeric value for interpolation all subjects' last records

        :type maximum_allowed_gap_days: None or int
        :param maximum_allowed_gap_days: When it's none, there is no limit on the length of the missing gap. If it's
            set (e.g., 90 days), when the missing gap exceeds the limit, even if the TLFB records at the start and
            end of the missing block indicate no substance use, the calculator will still impute the entire window as
            using substance.

        :type biochemical_data: TLFBData
        :param biochemical_data: The biochemical data that is used to impute the self-reported data

        :param overridden_amount: Union[float, int, 'infer']
            The default is 'infer', which means that when the particular date's record exceeds the biochemical
            abstinence cutoff, while its self-reported use is below the self-reported cutoff, the self-reported use
            is interpolated as 1 unit above the self-report cutoff. For instance, in most scenarios, the self-reported
            cutoff is 0, then the self-reported use will be interpolated as 1 when the biochemical value is above the
            cutoff while the patient's self-reported use is 0.
            You can also specify what other values to be used for the interpolation when the above described condition
            is met.

        :return: Summary of the imputation
        """
        if impute is None or str(impute).lower() == "none":
            return
        if not (impute in ("uniform", "linear") or str(impute).isnumeric()):
            raise InputArgumentError("The imputation mode can only be None, 'uniform', 'linear', "
                                     "or a numeric value.")
        if 'imputation_code' in self.data.columns:
            self.data = self.data[self.data['imputation_code'] == 0]

        self.data.sort_values(by=self._index_keys, inplace=True, ignore_index=True)
        self.data['imputation_code'] = DataImputationCode.RAW.value

        to_concat = list()

        if last_record_action is not None:
            last_records = self.data.loc[self.data.groupby('id')['date'].idxmax()].copy()
            last_records['date'] = last_records['date'].map(lambda x: x + timedelta(days=1))
            if last_record_action != 'ffill':
                last_records['amount'] = float(last_record_action)
            last_records['imputation_code'] = DataImputationCode.IMPUTED.value
            to_concat.append(last_records)

        if biochemical_data is not None:
            if "imputation_code" in biochemical_data.data.columns:
                _biochemical_data = \
                    biochemical_data.data.drop(columns="imputation_code").rename(columns={"amount": "bio_amount"})
            else:
                _biochemical_data = \
                    biochemical_data.data.rename(columns={"amount": "bio_amount"})
            _merged = self.data.merge(_biochemical_data, how="left", on=self._index_keys)
            bio_amount = (self.abst_cutoff + 1) if overridden_amount == 'infer' else float(overridden_amount)

            def interpolate_tlfb(row):
                amount = row['amount']
                imputation_code = DataImputationCode.RAW.value
                if pd.isnull(row['bio_amount']):
                    return pd.Series([amount, imputation_code])
                if row['amount'] <= self.abst_cutoff and row['bio_amount'] > biochemical_data.abst_cutoff:
                    amount = bio_amount
                    imputation_code = DataImputationCode.OVERRIDDEN.value
                return pd.Series([amount, imputation_code])

            _merged[['amount', 'imputation_code']] = _merged.apply(interpolate_tlfb, axis=1)
            self.data = _merged.drop(columns="bio_amount")

        self.data['diff_days'] = self.data.groupby(['id'])['date'].diff().map(
            lambda x: (x.days if self.use_raw_date else x) if pd.notnull(x) else 1)
        missing_data = self.data[self.data['diff_days'] > 1.0]
        imputed_records = []
        for data_row in missing_data.itertuples():
            start_data = self.data.iloc[data_row.Index - 1]
            start_record = TLFBRecord(start_data.id, start_data.date, start_data.amount, start_data.imputation_code)
            end_record = TLFBRecord(data_row.id, data_row.date, data_row.amount, start_data.imputation_code)
            imputed_records.extend(
                self._impute_missing_block(start_record, end_record, impute, maximum_allowed_gap_days))
        self.data.drop(['diff_days'], axis=1, inplace=True)
        imputed_tlfb_data = pd.DataFrame(imputed_records)

        to_concat.append(self.data)
        to_concat.append(imputed_tlfb_data)

        self.data = pd.concat(to_concat).sort_values(self._index_keys, ignore_index=True)
        impute_summary = self.data.groupby(['imputation_code']).size().reset_index(). \
            rename({0: "record_count"}, axis=1)
        impute_summary['imputation_code'] = impute_summary['imputation_code'].map(
            lambda x: DataImputationCode(x).name
        )
        return impute_summary

    def _get_missing_data(self):
        self.data['diff_days'] = self.data.groupby(['id'])['date'].diff().map(
            lambda x: (x.days if self.use_raw_date else x) if pd.notnull(x) else 1)
        return self.data[self.data['diff_days'] > 1.0]

    def _impute_missing_block(self, start_record: TLFBRecord, end_record: TLFBRecord, impute, maximum_allowed_gap_days):
        subject_id, start_date, start_amount, _ = start_record
        subject_id, end_date, end_amount, _ = end_record
        imputation_code = DataImputationCode.IMPUTED.value
        day_number = (end_date - start_date).days if self.use_raw_date else (end_date - start_date)
        imputed_records = []
        if not maximum_allowed_gap_days and day_number > maximum_allowed_gap_days:
            for i in range(1, day_number):
                imputed_date = start_date + (timedelta(days=i) if self.use_raw_date else i)
                imputed_records.append(TLFBRecord(subject_id, imputed_date, self.abst_cutoff + 1, imputation_code))
            return imputed_records

        if impute == "linear":
            m = (end_amount - start_amount) / day_number
            for i in range(1, day_number):
                imputed_date = start_date + (timedelta(days=i) if self.use_raw_date else i)
                imputed_amount = m * i + start_amount
                imputed_records.append(TLFBRecord(subject_id, imputed_date, imputed_amount, imputation_code))
        elif impute == "uniform":
            imputed_amount = np.mean([start_amount, end_amount])
            for i in range(1, day_number):
                imputed_date = start_date + (timedelta(days=i) if self.use_raw_date else i)
                imputed_records.append(TLFBRecord(subject_id, imputed_date, imputed_amount, imputation_code))
        else:
            imputed_amount = float(impute)
            for i in range(1, day_number):
                imputed_date = start_date + (timedelta(days=i) if self.use_raw_date else i)
                imputed_records.append(TLFBRecord(subject_id, imputed_date, imputed_amount, imputation_code))
        return imputed_records

    def get_subject_data(self, subject_id, start_date, end_date, mode):
        df = self.data
        if mode != "itt" and 'imputation_code' in self.data.columns:
            df = self.data.loc[self.data['imputation_code'] != DataImputationCode.IMPUTED.value, :]
        subject_data = df[(df['id'] == subject_id) &
                          (start_date <= df['date']) &
                          (df['date'] < end_date)]
        return subject_data
