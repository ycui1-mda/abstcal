from datetime import timedelta
import pandas as pd
import numpy as np
from abstcal.calculator_data import CalculatorData
from abstcal.tlfb_data import TLFBData
from abstcal.visit_data import VisitData
from abstcal.calculator_error import InputArgumentError, _show_warning


class AbstinenceCalculator:
    def __init__(self, tlfb_data: TLFBData, visit_data: VisitData):
        """
        Create an instance object to calculate abstinence

        :param tlfb_data: TLFBData, the TLFB data

        :param visit_data: VisitData, the visit data

        :return: None
        """
        self.tlfb_data = tlfb_data
        self.visit_data = visit_data
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

            start_dates = self.visit_data.get_visit_dates(subject_id, start_visit, mode=mode)
            start_date = start_dates[0] if start_dates else pd.NaT
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

    def abstinence_pp(self, end_visits, days, abst_var_names='infer', including_end=False, mode="itt"):
        """
        Calculate the point-prevalence abstinence using the end visit's date.

        :param end_visits: The reference visit(s) on which the abstinence is to be calculated

        :param days: The number of days preceding the end visit(s), it can be a single integer, or a list/tuple of days

        :param abst_var_names: The name(s) of the abstinence variable(s), by default, the name(s) will be inferred,
            if not inferred, the number of abstinence variable names should match the number of end visits

        :param including_end: Whether you want to include the anchor visit or not, default=False

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
            end_dates = self.visit_data.get_visit_dates(subject_id, end_visits, including_end, mode=mode)

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
                             including_end=False, mode="itt"):
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

        :param including_end: Whether you want to include the anchor visit or not, default=False

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
            start_dates = self.visit_data.get_visit_dates(subject_id, quit_visit, grace_days, mode=mode)
            start_date = start_dates[0] if start_dates else pd.NaT
            end_dates = self.visit_data.get_visit_dates(subject_id, end_visits, including_end, mode=mode)

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
                                assert len(parsed_criterion) == 4
                                cutoff_amount, cutoff_unit, window_amount, _ = parsed_criterion
                                cutoff_amount = float(cutoff_amount)
                                window_amount = int(float(window_amount))
                                index_list = \
                                    subject_data.index[subject_data['amount'] > self.tlfb_data.abst_cutoff].tolist()
                                for j in index_list:
                                    one_window = range(j, j + window_amount)
                                    lapse_tracking = 0
                                    for elem_i, elem in enumerate(one_window, j):
                                        if cutoff_unit == "days":
                                            lapse_tracking += elem in index_list
                                        else:
                                            if elem_i in subject_data.index:
                                                lapse_tracking += subject_data.loc[elem_i, "amount"]
                                        if lapse_tracking > cutoff_amount:
                                            if elem_i in subject_data.index:
                                                lapse = subject_data.loc[elem_i, :]
                                                break
                                    if lapse is not None:
                                        break
                            if subject_data['amount'].count() == int((end_date - start_date).days):
                                abstinent = int(lapse is None)

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
            _show_warning(f"The end date of the time window for the subject {subject_id} {end_date} isn't later \
            than the start date {start_date}. Please verify that the visit dates are correct.")
        elif pd.NaT in (start_date, end_date):
            _show_warning(f"Subject {subject_id} is missing some of the date information.")
        else:
            validated = True
        return validated

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
        if mode == "ro" and (not no_missing_data):
            return np.nan, None
        for record in subject_data.itertuples():
            if record.amount > self.tlfb_data.abst_cutoff:
                lapse_record = record
                break
        abstinent = int(no_missing_data and (lapse_record is None))
        return abstinent, lapse_record

    @staticmethod
    def _append_lapse_as_needed(lapses, lapse, abst_name):
        if lapse is not None and len(lapse):
            lapse_id, lapse_date, lapse_amount = lapse.id, lapse.date, lapse.amount
            lapses.append((lapse_id, lapse_date, lapse_amount, abst_name))

    @staticmethod
    def calculate_abstinence_rates(dfs, filepath=None):
        """
        Calculate Abstinence Rates

        :param dfs: One or more DataFrame objects, which are abstinence data generated from the calculations

        :param filepath: Union[str, Path, None], the path to the output file if set, the default is None, no output

        :return: DataFrame containing the abstinence rates
        """
        dfs = AbstinenceCalculator._listize_args(dfs)
        abst_df = pd.concat(dfs, axis=1)
        data_rows = list()
        for column in abst_df.columns:
            abst_count = int(abst_df[column].sum())
            subject_count = abst_df[column].count()
            data_row = [column, abst_count, subject_count, f"{abst_count / subject_count:.2%}"]
            data_rows.append(data_row)

        df = pd.DataFrame(data_rows, columns=['Abstinence Name', 'Abstinent Count', 'Subject Count', 'Abstinence Rate'])
        CalculatorData.write_data_to_path(df, filepath, True)
        return df

    @staticmethod
    def merge_abst_data(dfs, filepath=None):
        """
        Merge abstinence data and write the merged DataFrame to a file

        :param dfs: Union[list, tuple], the list of abstinence data results (DataFrame), it's also OK to have just one

        :param filepath: Union[str, Path], the output file name with a proper extension

        :return: DataFrame merged
        """
        dfs = AbstinenceCalculator._listize_args(dfs)
        merged_df = pd.concat(dfs, axis=1)
        CalculatorData.write_data_to_path(merged_df, filepath, True)
        return merged_df

    @staticmethod
    def merge_lapse_data(dfs, filepath=None):
        """
        Merge lapses data and write the merged DataFrame to a file

        :param dfs: Union[list, tuple], the list of lapse data results (DataFrame), it's also OK to have just one

        :param filepath: Union[str, path], the output file name with a proper extension

        :return: DataFrame merged
        """
        dfs = AbstinenceCalculator._listize_args(dfs)
        merged_df = pd.concat(dfs, axis=0)
        CalculatorData.write_data_to_path(merged_df, filepath)
        return merged_df
