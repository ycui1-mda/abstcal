# %%
import os
import warnings
from enum import Enum
from datetime import timedelta
import pandas as pd
import numpy as np
from statistics import mean
from collections import namedtuple

# %%
TLFBRecord = namedtuple("TLFBRecord", "id date amount")
VisitRecord = namedtuple("VisitRecord", "id visit date")


def impute_tlfb_missing_block(start_record: TLFBRecord, end_record: TLFBRecord, impute="uniform"):
    subject_id, start_date, start_amount = start_record
    subject_id, end_date, end_amount = end_record
    day_number = (end_date - start_date).days
    imputed_records = []
    if impute == "linear":
        m = (end_amount - start_amount) / day_number
        for i in range(1, day_number):
            imputed_date = start_date + timedelta(days=i)
            imputed_amount = m * i + start_amount
            imputed_records.append(TLFBRecord(subject_id, imputed_date, imputed_amount))
    elif impute == "uniform":
        imputed_amount = mean([start_amount, end_amount])
        for i in range(1, day_number):
            imputed_date = start_date + timedelta(days=i)
            imputed_records.append(TLFBRecord(subject_id, imputed_date, imputed_amount))
    else:
        imputed_amount = float(impute)
        for i in range(1, day_number):
            imputed_date = start_date + timedelta(days=i)
            imputed_records.append(TLFBRecord(subject_id, imputed_date, imputed_amount))
    return imputed_records


# %%
class TLFBImputationCode(Enum):
    RAW = 0  # raw data, no imputation
    AB_AB = 1  # abstinent ... abstinent
    NONAB_NONAB = 2  # non-abstinent ... non-abstinent
    NONAB_AB = 3  # non-abstinent ... abstinent
    AB_NONAB = 4  # abstinent ... non-abstinent
    UNSPECIFIED = 9  # unspecified category

    @classmethod
    def code_for_missing_interval(cls, start_amount, end_amount, abst_cutoff):
        imputation_code = cls.UNSPECIFIED
        if start_amount <= abst_cutoff and end_amount <= abst_cutoff:
            imputation_code = cls.AB_AB
        elif start_amount > abst_cutoff and end_amount > abst_cutoff:
            imputation_code = cls.NONAB_NONAB
        elif start_amount > abst_cutoff >= end_amount:
            imputation_code = cls.NONAB_AB
        elif start_amount <= abst_cutoff < end_amount:
            imputation_code = cls.AB_NONAB
        return imputation_code


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
    def __init__(self, tlfb_filepath, visit_filepath, abst_cutoff=0):
        """
        Create an instance object to calculate abstinence
        :param tlfb_filepath: filepath for the TLFB data in the tabular format, str or pathlib.Path
        :param visit_filepath: filepath for the visit data in the tabular format, str or pathlib.Path
        :param abst_cutoff: The cutoff of abstinence, default=0, inclusive, for example if the cutoff is 0.1 then 0.1
        and below is considered abstinent
        """
        self._tlfb_data = read_data_from_path(tlfb_filepath)
        self._visit_data = read_data_from_path(visit_filepath)
        self.abst_cutoff = abst_cutoff
        self._tlfb_data_imputed = None
        self._visit_data_imputed = None
        self._tlfb_duplicates = None
        self._visit_duplicates = None
        self._subject_ids = None

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
        self.validate_tlfb_data()
        self.remove_tlfb_duplicates()
        self.sort_tlfb_data()
        self._subject_ids = self._tlfb_data['id'].unique()
        self._impute_tlfb_data(impute)

    def validate_tlfb_data(self):
        needed_cols_ordered = ('id', 'date', 'amount')
        needed_cols_unordered = set(needed_cols_ordered)
        current_cols_unordered = set(self._tlfb_data.columns)
        if len(current_cols_unordered) != len(needed_cols_unordered):
            raise FileFormatError('The TLFB data should have only id, date, and amount columns.')
        if needed_cols_unordered != current_cols_unordered:
            warning_message = """The TLFB data don't appear to have the needed columns: id, date, and amount. It has 
            been renamed in such order for calculation purposes. If your columns aren't in this order, and you'll 
            encounter errors in later calculation steps."""
            show_warning(warning_message)
            self._tlfb_data.columns = needed_cols_ordered
        self._tlfb_data['date'] = pd.to_datetime(self._tlfb_data['date'], errors='coerce', infer_datetime_format=True)
        self._tlfb_data['amount'] = self._tlfb_data['amount'].astype(float)
        if (self._tlfb_data['amount'] < 0).any():
            raise DataSourceQualityError("Fatal Error: the TLFB data have negative values.")

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

    def _impute_tlfb_data(self, how_impute_tlfb):
        if how_impute_tlfb is None or str(how_impute_tlfb).lower() == "none":
            return
        if not (how_impute_tlfb in ("uniform", "linear") or how_impute_tlfb.isnumeric()):
            raise InputArgumentError("The imputation mode can only be None, 'uniform', 'linear', "
                                     "or a numeric value.")
        tlfb_df = self._tlfb_data.copy()
        tlfb_df['diff_days'] = tlfb_df.groupby(['id'])['date'].diff().map(
            lambda x: x.days if pd.notnull(x) else 1)
        missing_data = tlfb_df[tlfb_df['diff_days'] > 1.0]
        imputed_records = []
        for row in missing_data.itertuples():
            start_data = tlfb_df.iloc[row.Index - 1]
            start_record = TLFBRecord(start_data.id, start_data.date, start_data.amount)
            end_record = TLFBRecord(row.id, row.date, row.amount)
            imputed_records.extend(impute_tlfb_missing_block(start_record, end_record, impute=how_impute_tlfb))
        tlfb_df.drop(['diff_days'], axis=1, inplace=True)
        imputed_tlfb_data = pd.DataFrame(imputed_records)
        self._tlfb_data_imputed = pd.concat([self._tlfb_data, imputed_tlfb_data]).\
            sort_values(['id', 'date']).reset_index(drop=True)
        message = f"The number of imputed records: {len(imputed_records)}"
        show_warning(message)

    def prepare_visit_data(self, impute="freq"):
        """
        Prepare the visit data ready for calculation.
        :param impute: how the missing visits are imputed, None=no imputation, "freq"=use the most frequent,
                        "mean"=use the average. Both imputation methods assume that the visit structure is the same
                        among all subjects. It will first find the earliest visit as the anchor date, impute any missing
                        visit dates either using the average or the most frequent interval. Please note the anchor dates
                        can't be missing.
        """
        self.validate_visit_data()
        self.remove_visit_duplicates_if_applicable()
        self._impute_visit_data(impute)

    def validate_visit_data(self):
        needed_cols_ordered = ('id', 'visit', 'date')
        needed_cols_unordered = set(needed_cols_ordered)
        current_cols_unordered = set(self._visit_data.columns)
        if len(current_cols_unordered) != len(needed_cols_unordered):
            raise FileFormatError('The visit data should have only id, visit, and date columns.')
        if needed_cols_unordered != current_cols_unordered:
            warning_message = """The visit data don't appear to have the needed columns: id, visit, and date. It has 
                    been renamed in such order for calculation purposes. If your columns aren't in this order, and 
                    you'll encounter errors in later calculation steps."""
            show_warning(warning_message)
            self._visit_data.columns = needed_cols_ordered
        self._visit_data['date'] = pd.to_datetime(self._visit_data['date'], errors='coerce', infer_datetime_format=True)
        self._visit_data['visit'] = self._visit_data['visit'].map(lambda x: 'v' + str(x))

        unique_TLFB_ids = set(self._tlfb_data['id'].unique())
        unique_visit_ids = set(self._visit_data['id'].unique())
        no_visit_ids = unique_TLFB_ids - unique_visit_ids
        if no_visit_ids:
            message = f"The visit data don't have the visit information for some subjects. Those subjects " \
                      f"({no_visit_ids}) who don't visit data can't be scored."
            show_warning(message)

    def remove_visit_duplicates_if_applicable(self):
        self._visit_duplicates = self._visit_data.duplicated(['id', 'visit'], keep=False)
        if self._visit_duplicates is not None:
            message = f"""The visits data have {len(self._visit_duplicates)} duplicates based on id. Extra 
            duplicates will be removed in the calculation. To get the duplicates, use get_visit_duplicates()."""
            show_warning(message)
            self._visit_data.drop_duplicates(['id', 'visit'], inplace=True)

    def get_visit_duplicates(self):
        return self._visit_duplicates

    def sort_visit_data(self):
        self._visit_data = self._visit_data.sort_values(by=["id", "visit"]).reset_index(drop=True)

    def _impute_visit_data(self, how_impute_visit):
        if how_impute_visit is None:
            return
        if how_impute_visit not in ('freq', 'mean'):
            raise InputArgumentError('You can only specify the imputation method to be "freq" or "mean".')
        min_date_indices = self._visit_data.groupby(['id'])['date'].idxmin()
        anchor_visit = self._visit_data.loc[min_date_indices, 'visit'].value_counts().idxmax()
        _visit_data_wide = self._visit_data.pivot(index='id', columns='visit', values='date')
        missing_anchor_ids = set(_visit_data_wide.index[_visit_data_wide[anchor_visit].isnull()])
        if missing_anchor_ids:
            message = f"Subjects {missing_anchor_ids} are missing anchor visit {anchor_visit}. " \
                      f"There might be problems calculating abstinence data for these subjects."
            show_warning(message)
        visits = set(_visit_data_wide.columns) - {'id', anchor_visit}
        anchor_visits = _visit_data_wide.loc[:, anchor_visit]
        for visit in visits:
            days_diff = (_visit_data_wide[visit] - anchor_visits).map(lambda day: day.days)
            used_days_diff = int(days_diff.median()) if how_impute_visit == 'freq' else int(days_diff.mean())
            _visit_data_wide[visit] = _visit_data_wide.apply(
                lambda x: x[visit] if pd.notnull(x[visit]) else x[anchor_visit] + timedelta(days=used_days_diff),
                axis=1)
        self._visit_data_wide = _visit_data_wide

    @staticmethod
    def format_visits_names(end_visits, abst_var_names, prefix=''):
        if not isinstance(end_visits, list):
            end_visits = [end_visits]
        if abst_var_names == 'inferred':
            abst_names = [f"{prefix}_{end_visit}" for end_visit in end_visits]
        elif isinstance(abst_var_names, list):
            abst_names = abst_var_names
        else:
            abst_names = [abst_var_names]
        if len(end_visits) != len(abst_names):
            raise InputArgumentError("The number of abstinence variable names should match the number of visits.")
        return end_visits, abst_names

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
        end_visits, abst_names = AbstinenceCalculator.format_visits_names(end_visits, abst_var_names,
                                                                          f'abst_cont_{start_visit}')
        results = []
        for i, subject_id in enumerate(self._subject_ids):
            start_date = self._get_visit_dates(subject_id, start_visit)
            end_dates = self._get_visit_dates(subject_id, end_visits, including_end)
            max_end_date = max(end_dates)
            subject_data = self._get_subject_data(subject_id, start_date, max_end_date)
            result = [subject_id]
            for end_date in end_dates:
                if start_date >= end_date:
                    show_warning(f"The end date of the time window for the subject {subject_id} isn't later "
                                 f"than the start date. Please verify that the visit dates are correct.")
                    abstinent = np.nan
                else:
                    abstinent = AbstinenceCalculator._continuous_abst(subject_data, start_date, end_date,
                                                                      self.abst_cutoff)
                result.append(abstinent)
            results.append(result)
        col_names = abst_names.insert(0, 'id')
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
            end_visits, abst_names = AbstinenceCalculator.format_visits_names(end_visits, abst_var_names,
                                                                              f'abst_pp{day}')
            all_abst_names += abst_names
        results = []
        for i, subject_id in enumerate(self._subject_ids):
            end_dates = self._get_visit_dates(subject_id, end_visits, including_anchor)
            result = [subject_id]
            for end_date in end_dates:
                for day in days:
                    start_date = end_date + timedelta(days=-day)
                    subject_data = self._get_subject_data(subject_id, start_date, end_date)
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
        for i, subject_id in enumerate(self._subject_ids):
            start_date = AbstinenceCalculator._get_visit_dates(subject_id, quit_visit, grace_days)
            end_dates = AbstinenceCalculator._get_visit_dates(subject_id, end_visits, including_anchor)
            max_end_date = max(end_dates)
            _subject_data = self._get_subject_data(subject_id, start_date, max_end_date)
            result = [subject_id]
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
        return pd.DataFrame.from_dict({'id': self._subject_ids, abst_name: results})

    def _get_subject_data(self, subject_id, start_date, end_date, mode):
        df = self._tlfb_data if mode == "itt" else self._tlfb_data_imputed
        subject_data = df[(df['id'] == subject_id) &
                          (start_date <= df['date']) &
                          (df['date'] < end_date)]
        return subject_data

    def _get_visit_dates(self, subject_id, visit_names, increment_days=0):
        if not isinstance(visit_names, list):
            visit_names = [visit_names]
        visit_dates_cond = (self._visit_data['id'] == subject_id) & (self._visit_data['visit'].isin(visit_names))
        dates = list(self._visit_data.loc[visit_dates_cond, 'date'])
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


def show_warning(warning_message):
    warnings.warn(warning_message)


def read_data_from_path(file_path):
    file_extension = os.path.splitext(file_path)[-1].lower()
    if file_extension == ".csv":
        df = pd.read_csv(file_path, infer_datetime_format=True)
    elif file_extension in (".xls", ".xlsx", ".xlsm", ".xlsb"):
        df = pd.read_excel(file_path, infer_datetime_format=True)
    elif file_extension == ".txt":
        df = pd.read_csv(file_path, sep='\t', infer_datetime_format=True)
    else:
        raise FileExtensionError(file_path)
    return df


# # %%
# tlfb_df = pd.read_sas("tbltimelinefb.sas7bdat", encoding="utf-8")
# tlfb_df.drop(["SubstanceId"], axis=1, inplace=True)
# tlfb_df.rename({"SubjectID": "id", "NumCig": "amount"}, axis=1, inplace=True)
# tlfb_df.to_csv("smartmod_tlfb_data.csv", index=False)
#
# visit_df = pd.read_sas("scored_visitrecord.sas7bdat")
# visit_df.drop(["attended"], axis=1, inplace=True)
# visit_df.rename({"SubjectID": "id"}, axis=1, inplace=True)
# visit_df.to_csv("smartmod_visit_data.csv", index=False)


# %%
abstinence_calculator = AbstinenceCalculator("smartmod_tlfb_data.csv", "smartmod_visit_data.csv")
abstinence_calculator.prepare_tlfb_data("uniform")
abstinence_calculator.prepare_visit_data("freq")
# abstinence_calculator.get_data_summary()
# help(AbstinenceCalculator)

# %%
# from math import log10
# def calculate_k(max_con, min_con, to_add=0.5):
#     k_value = log10(max_con) - log10(min_con) + to_add
#     print(k_value)
#
# def calculate_max_min_ratio(k_value, to_add=0.5):
#     ratio = (k_value - to_add)
#     print(10**ratio)
#
# calculate_k(8.72, 0.01)
#
# calculate_k(21.29, 0.1)