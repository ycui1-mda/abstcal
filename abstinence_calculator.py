import pandas as pd
import os
import warnings
from datetime import timedelta


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


class DataSourceQualityError(AbstinenceCalculatorError):
    pass


class FileFormatError(AbstinenceCalculatorError):
    pass


class InputArgumentError(AbstinenceCalculatorError):
    pass


# %%
class AbstinenceCalculator:
    def __init__(self, tlfb_filepath, visit_filepath):
        """
        Create an instance object to calculate abstinence
        :param tlfb_filepath: filepath for the TLFB data in the tabular format, str or pathlib.Path
        :param visit_filepath: filepath for the visit data in the tabular format, str or pathlib.Path
        """
        self._tlfb_data = read_data_from_path(tlfb_filepath)
        self._visit_data = read_data_from_path(visit_filepath)
        self._tlfb_duplicates = None
        self._visit_duplicates = None
        self._subject_ids = None

    @classmethod
    def read_data(cls, tlfb_filepath, visit_filepath):
        calculator = cls(tlfb_filepath, visit_filepath)
        return calculator

    def prepare_tlfb_data(self, impute=True):
        self.validate_tlfb_data()
        self.remove_tlfb_duplicates()
        self.sort_tlfb_data()
        self._subject_ids = self._tlfb_data['subject_id'].unique()
        if impute:
            self.impute_tlfb_data()

            'hello'

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

    def prepare_visit_data(self, impute=None):
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
        self.impute_visit_data(impute)

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

    def impute_visit_data(self, how_impute_visit):
        if how_impute_visit is None:
            return
        if how_impute_visit not in ('freq', 'mean'):
            raise InputArgumentError('You can only specify the imputation method to be "freq" or "mean".')
        min_date_indices = self._visit_data.groupby(['id'])['date'].idxmin()
        anchor_visit = self._visit_data.loc[min_date_indices, 'visit'].value_counts().idxmax()
        _visit_data_wide = self._visit_data.pivot(index='id', columns='visit', values='date')
        if missing_anchor_ids := set(_visit_data_wide.loc[_visit_data_wide[anchor_visit].isnull, 'id']):
            message = f"Subjects {missing_anchor_ids} are missing anchor visit {anchor_visit}. There might be problems " \
                      f"calculating abstinence data for these subjects."
            show_warning(message)
        visits = set(_visit_data_wide.columns) - {'id', anchor_visit}
        anchor_visits = _visit_data_wide.loc[:, anchor_visit]
        for visit in visits:
            days_diff = _visit_data_wide[visit] - anchor_visits
            used_days_diff = int(days_diff.median()) if how_impute_visit == 'freq' else int(days_diff.mean())
            _visit_data_wide[visit] = _visit_data_wide.apply(
                lambda x: x[visit] if pd.notnull(x[visit]) else anchor_visit + timedelta(days=used_days_diff),
                axis=1
            )

    def abstinence_cont(self, start_visit, end_visit, abst_name='inferred', abst_cutoff=0, including_end=False):
        """
        Calculates the continuous abstinence for the time window.
        :param start_visit: The name of the visit where the time window starts
        :param end_visit: The name of the visit where the time window ends
        :param abst_name: The name of the abstinence variable, by default, the name will be inferred
        :param abst_cutoff: The cutoff of abstinence, default=0, inclusive, for example if the cutoff is 0.1 then 0.1
        and below is considered abstinent
        :param including_end: Whether you want to include the end visit or not, default=False
        :return: Pandas DataFrame with two columns, subject id and abstinence result
        """
        results = []
        for i, subject_id in enumerate(self._subject_ids):
            start_date = self.get_visit_date(subject_id, start_visit)
            end_date = self.get_visit_date(subject_id, end_visit)
            if start_date >= end_date:
                show_warning(f"The end date of the time window for the subject {subject_id} isn't later "
                             f"than the start date. Please verify that the visit dates are correct.")
            if including_end:
                end_date = end_date + timedelta(days=1)
            subject_data = self.get_subject_data(subject_id, start_date, end_date)
            abstinent = AbstinenceCalculator.continuous_abst(subject_data, start_date, end_date, abst_cutoff)
            results.append(abstinent)
        if abst_name == 'inferred':
            abst_name = f"abst_cont_{start_visit}_{end_visit}"
        return pd.DataFrame.from_dict({'id': self._subject_ids, abst_name: results})

    def abstinence_pp(self, end_visit, days, abst_name='inferred', abst_cutoff=0, including_anchor=False):
        """
        Calculate the point-prevalence abstinence using the end visit's date.
        :param end_visit: The reference visit on which the abstinence is to be calculated
        :param days: The number of days preceding the end visit
        :param abst_name: The name of the abstinence variable, by default, the name will be inferred
        :param including_anchor: Whether you want to include the anchor visit or not, default=False
        :return: Pandas DataFrame with two columns, subject id and abstinence result
        """
        results = []
        for i, subject_id in enumerate(self._subject_ids):
            end_date = self.get_visit_date(subject_id, end_visit)
            if including_anchor:
                end_date = end_date + timedelta(days=1)
            start_date = end_date + timedelta(days=-days)
            subject_data = self.get_subject_data(subject_id, start_date, end_date)
            abstinent = AbstinenceCalculator.continuous_abst(subject_data, start_date, end_date, abst_cutoff)
            results.append(abstinent)
        return pd.DataFrame.from_dict({'id': self._subject_ids, abst_name: results})

    def abstinence_prolonged(self, quit_visit, end_visit, slip_allowed=True, slip_criterion='5 cigs', grace_days=14,
                             abst_name='inferred', abst_cutoff=0, including_anchor=False):
        """
        Calculate the prolonged abstinence using the time window
        :param quit_visit: The visit when the subjects are scheduled to quit smoking
        :param end_visit: The reference visit on which the abstinence is to be calculated
        :param slip_allowed: Whether slip is allowed
        :param slip_criterion: The criterion for a slip, slip is only examined for the window between the date when the
        grace period is over and the end date, supported criterion: '5 cigs'
        :param grace_days: The number of days for the grace period following the quit date
        :param abst_name: The name of the abstinence variable, by default, the name will be inferred
        :param including_anchor: Whether you want to include the anchor visit or not, default=False
        :return: Pandas DataFrame with two columns, subject id and abstinence result
        """
        results = []
        for i, subject_id in enumerate(self._subject_ids):
            quit_date = AbstinenceCalculator.get_visit_date(subject_id, quit_visit)
            start_date = quit_date + timedelta(days=grace_days)
            end_date = AbstinenceCalculator.get_visit_date(subject_id, end_visit)
            if including_anchor:
                end_date = end_date + timedelta(days=1)
            days_to_check = int((end_date - start_date).days)
            subject_data = AbstinenceCalculator.get_subject_data(subject_id, start_date, end_date)
            if subject_data['amount'].count() != days_to_check:
                abstinent = 0
                show_warning(f"There are some fewer data records for subject {subject_id}.")
            else:
                if not slip_allowed:
                    abstinent = int((subject_data['amount'] <= abst_cutoff).all())
                else:
                    if slip_criterion == '5 cigs':
                        abstinent = int(subject_data['amount'].sum() <= 5)
            results.append(abstinent)
        return pd.DataFrame.from_dict({'id': self._subject_ids, abst_name: results})

    def get_subject_data(self, subject_id, start_date, end_date):
        subject_data = self._tlfb_data[(self._tlfb_data['id'] == subject_id) &
                                       (start_date <= self._tlfb_data['date']) &
                                       (self._tlfb_data['date'] < end_date)]
        return subject_data

    def get_visit_date(self, subject_id, visit_name):
        visit_date_cond = (self._visit_data['id'] == subject_id) & (self._visit_data['visit'] == visit_name)
        return self._visit_data.loc[visit_date_cond, 'date']

    @staticmethod
    def continuous_abst(subject_data, start_date, end_date, abst_cutoff):
        days_to_check = int((end_date - start_date).days)
        return int((subject_data['amount'].count() == days_to_check) & (subject_data['amount'] <= abst_cutoff).all())


def show_warning(warning_message):
    warnings.warn(warning_message)


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


# %%
abstinence_calculator = AbstinenceCalculator()
tlfb_df = pd.read_csv("/Users/ycui/PycharmProjects/abstinence_calculator/venv/bin/tlfb_edited.csv")
abstinence_calculator.prepare_data()
abstinence_calculator.get_data_summary()
