import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from abstcal.calculator_error import InputArgumentError, _show_warning
from abstcal.calculator_data import CalculatorData, DataImputationCode


class VisitData(CalculatorData):
    def __init__(self, filepath, data_format="long", expected_ordered_visits="infer", included_subjects="all"):
        """
        Create the instance object for the visit data

        :param filepath: Union[str, path], the file path to the visit data
            for the web app, you can directly pass the buffer

        :param data_format: Union["long", "wide"], how the visit data are organized
            long: the data should have three columns, id, visit, and date
            wide: the data have multiple columns and one row per subject, the first column has to be named 'id'
                and the remaining columns are visits with values being the dates

        :type expected_ordered_visits: object
        :param expected_ordered_visits: Union["infer", list, None], default "infer"
            The expected order of the visits, such that when dates are out of order can be detected
            The list of visit names should match the actual visits in the dataset
            When None, no checking will be performed when you profile the visit data
            The default is infer, and it means that the order of the visits is sorted based on its numeric or
            alphabetic order

        :param included_subjects: Union[list, tuple], the list of subject ids that are included in the dataset,
            the default option "all" means that all subjects in the dataset will be used

        """
        if data_format == "wide":
            df_wide = super().read_data_from_path(filepath)
            df_long = df_wide.melt(id_vars="id", var_name="visit", value_name="date")
        else:
            df_long = super().read_data_from_path(filepath)
        self.data = self._validated_data(df_long)
        if included_subjects != "all":
            self.data = self.data.loc[self.data["id"].isin(included_subjects), :]
        self._index_keys = ['id', 'visit']
        self._value_key = 'date'
        if expected_ordered_visits is not None:
            if expected_ordered_visits != "infer":
                self.expected_ordered_visits = expected_ordered_visits
                self.data = self.data.loc[self.data["visit"].isin(expected_ordered_visits), :]
            else:
                self.expected_ordered_visits = sorted(self.data['visit'].unique())
        else:
            self.expected_ordered_visits = None

        self.visits = set(self.data['visit'].unique())
        self.subject_ids = set(self.data['id'].unique())
        self.data.reset_index(drop=True, inplace=True)

    @staticmethod
    def _validated_data(df):
        CalculatorData._validate_columns(df, ('id', 'visit', 'date'), "visit", "id, visit, and date")
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        return df

    def profile_data(self, min_date_cutoff=None, max_date_cutoff=None):
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

        :return: Tuple, summaries of the visit data at the sample and subject level
        """
        casted_min_date_cutoff = pd.to_datetime(min_date_cutoff, infer_datetime_format=True)
        casted_max_date_cutoff = pd.to_datetime(max_date_cutoff, infer_datetime_format=True)
        visit_summary_series = self._get_visit_sample_summary(casted_min_date_cutoff, casted_max_date_cutoff)
        visit_subject_summary = self._get_visit_subject_summary(casted_min_date_cutoff, casted_max_date_cutoff)
        return visit_summary_series, visit_subject_summary

    def get_out_of_order_visit_data(self):
        """
        Get the data with out of ordered visits

        :return: DataFrame or None
        """
        if self.expected_ordered_visits is None:
            return pd.DataFrame()

        subject_summary = self._get_visit_subject_summary().reset_index()
        out_of_order_ids = \
            subject_summary.loc[subject_summary['visit_dates_out_of_order'], "id"].to_list()
        return self.data.loc[self.data['id'].isin(out_of_order_ids), :].sort_values(by=['id', 'visit'])

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
            visit_dates = self._visit_data.dropna().loc[self._visit_data['visit'] == visit, :].\
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
            {f"visit_{visit}_attendance": f"{count} ({count / subject_count:.2%})"
             for visit, count in visit_record_counts.items()}
        )
        return pd.Series(visit_summary)

    def get_retention_rates(self, filepath=None):
        _data = self.data
        if "imputation_code" in _data.columns:
            _data = self.data[self.data['imputation_code'] == 0]

        last_attended_counts = _data.loc[_data.groupby('id')['date'].idxmax(), "visit"].value_counts().to_dict()

        sorted_visits = self.expected_ordered_visits or sorted(self.visits)

        tracking = dict()
        remaining = len(self.subject_ids)
        for i, visit in enumerate(sorted_visits):
            losing = last_attended_counts.get(visit, 0)
            tracking[visit] = remaining
            remaining -= losing

        retention_df = pd.Series(tracking).to_frame("subject_count")
        retention_df.index.name = "visit"
        retention_df.reset_index()
        subject_count = len(self.subject_ids)
        retention_df['retention_rate'] = retention_df['subject_count'].map(
            lambda x: f"{x / subject_count:.2%}"
        )
        retention_df['attrition_rate'] = retention_df['subject_count'].map(
            lambda x: f"{(subject_count - x) / subject_count:.2%}"
        )

        retention_df['attendance_rate'] = self.data['visit'].value_counts().map(
            lambda x: f"{x / subject_count:.2%}"
        )
        CalculatorData.write_data_to_path(retention_df, filepath, True)
        return retention_df

    def _get_visit_subject_summary(self, min_date_cutoff=None, max_date_cutoff=None):
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
        visit_dates_out_of_order = self._check_visit_order()
        if visit_dates_out_of_order is not None:
            summary_dfs.append(visit_dates_out_of_order)
            col_names.append('visit_dates_out_of_order')
        summary_df = pd.concat(summary_dfs, axis=1)
        summary_df.columns = col_names
        return summary_df.fillna(0)

    def _check_visit_order(self):
        if self.expected_ordered_visits is not None:
            _temp_data = self.data.copy()
            if isinstance(self.expected_ordered_visits, list):
                _temp_data['visit'] = self.data['visit'].map(
                    {visit: i for i, visit in enumerate(self.expected_ordered_visits)})
            else:
                _show_warning("Supported options for expected_ordered_visits are list of visits, None, and infer. "
                              "The expected visit order is inferred to check if the dates are in the correct order.")
            sorted_visit_data = _temp_data.sort_values(by=['id', 'visit'], ignore_index=True)
            sorted_visit_data['ascending'] = sorted_visit_data.groupby(['id'])['date'].diff().map(
                lambda x: True if x is pd.NaT or x.days >= 0 else False
            )
            visits_out_of_order = sorted_visit_data.groupby(['id'])['ascending'].all().map(lambda x: not x)
            if visits_out_of_order.sum() > 0:
                _show_warning(f"Please note that some subjects (n={visits_out_of_order.sum()}) appear to have their \
                visit dates out of the correct order. You can find out who they are in the visit data summary by \
                subject. Please fix them if applicable.")
            return visits_out_of_order

    def recode_outliers(self, floor_date, ceil_date, drop_outliers=True):
        """
        Recode the abnormal data of the visit dataset

        :param floor_date: date (or date-like strings: "07/15/2020")
            Drop records when their values are lower than the floor date if drop_outliers is True (the default),
            otherwise outliers will be replaced with the floor date (i.e., drop_outliers=False)

        :param ceil_date: date (or date-like strings: "07/15/2020")
            Drop records when their values are higher than the ceil date if drop_outliers is True (the default),
            otherwise outliers will be replaced with the ceil date (i.e., drop_outliers=False)

        :param drop_outliers: bool, default is True
            Drop outliers when it's True and recode outliers to bounding dates when it's False

        :return: summary of the recoding
        """
        recode_summary = dict()
        casted_floor_date = pd.to_datetime(floor_date, infer_datetime_format=True)
        outlier_count_low = pd.Series(self.data['date'] < casted_floor_date).sum()
        recode_summary[f"Number of outliers (< {casted_floor_date.strftime('%m/%d/%Y')})"] = outlier_count_low
        self.data['date'] = self.data['date'].map(
            lambda x: np.nan if drop_outliers and x < casted_floor_date else max(x, casted_floor_date))
        casted_ceil_date = pd.to_datetime(ceil_date, infer_datetime_format=True)
        outlier_count_high = pd.Series(self.data['date'] > casted_ceil_date).sum()
        recode_summary[f"Number of outliers (> {casted_ceil_date.strftime('%m/%d/%Y')})"] = outlier_count_high
        self.data['date'] = self.data['date'].map(
            lambda x: np.nan if drop_outliers and x > casted_ceil_date else min(x, casted_ceil_date))
        if drop_outliers:
            self.drop_na_records()
        return recode_summary

    # noinspection PyTypeChecker
    def impute_data(self, anchor_visit='infer', impute='freq'):
        """
        Impute any missing visit data.

        :type anchor_visit: object
        :param anchor_visit: The anchor visit, which needs to exist for all subjects and the date of which will be
            used to impute the missing visit dates

        :param impute: Union["freq", "mean", None, dict], how the missing visit data are imputed
        None: no imputation
        "freq": use the most frequent, see below for further clarification
        "mean": use the average, see below for further clarification
            Both imputation methods assume that the visit structure is the same among all subjects. It will first find
            the earliest visit as the anchor date, impute any missing visit dates either using the average or the most
            frequent interval. Please note the anchor dates can't be missing.
        dict: A dictionary object mapping the number of days since the anchor visit

        :return DataFrame
        """

        if impute is None or str(impute).lower() == "none":
            return
        if impute not in ('freq', 'mean') and not isinstance(impute, dict):
            raise InputArgumentError('You can only specify the imputation method to be "freq" or "mean". Alternatively,'
                                     'you can specify a dictionary object mapping the number of days since the anchor'
                                     'visit.')

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
            if impute == 'freq':
                used_days_diff = days_diff.value_counts().idxmax()
            elif impute == 'mean':
                used_days_diff = int(days_diff.mean())
            else:
                used_days_diff = impute[visit]

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
        return dates
