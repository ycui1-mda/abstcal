import streamlit as st
import abstcal as ac
import sys
import datetime


test_tlfb_filepath = "/tests/test_tlfb.csv"
test_visit_filepath = "/tests/test_visit.csv"
test_bio_filepath = "/tests/test_co.csv"
bio_cutoff = 4

sys.tracebacklimit = 0

duplicate_options_mapped = {
    "Keep the minimal only": "min",
    "Keep the maximal only": "max",
    "Keep the mean only": "mean",
    "Remove all duplicates": False
}
duplicate_options = list(duplicate_options_mapped)

outlier_options_mapped = {
    "Don't examine outliers": None,
    "Remove the outliers": True,
    "Impute the outliers with the bounding values": False
}
outlier_options = list(outlier_options_mapped)


tlfb_data_params = dict.fromkeys([
    "filepath",
    "cutoff",
    "subjects",
    "duplicate_mode",
    "imputation_last_record",
    "imputation_mode",
    "imputation_gap_limit",
    "outliers_mode",
    "allowed_min",
    "allowed_max"
])

tlfb_imputation_options_mapped = {
    "Don't impute missing records": None,
    "Linear (a linear interpolation in the gap)": "linear",
    "Uniform (the same value in the gap)": "uniform",
    "Specified Value": 0
}
tlfb_imputation_options = list(tlfb_imputation_options_mapped)

visit_data_params = dict.fromkeys([
    "filepath",
    "file_format",
    "expected_visits",
    "duplicate_mode",
    "imputation_mode",
    "anchor_visit",
    "allowed_min",
    "allowed_max",
    "outliers_mode"
])
visit_file_formats = [
    "Long",
    "Wide"
]
visit_imputation_options_mapped = {
    "Don't impute dates": None,
    "The most frequent interval since the anchor visit": "freq",
    "The mean interval since the anchor visit": "mean"
}
visit_imputation_options = list(visit_imputation_options_mapped)

bio_data_params = dict.fromkeys([
    "filepath",
    "cutoff",
    "overridden_amount",
    "duplicate_mode",
    "imputation_mode",
    "allowed_min",
    "allowed_max",
    "outliers_mode",
    "enable_interpolation",
    "half_life",
    "days_interpolation"
])

abst_options = [
    "Point-Prevalence",
    "Prolonged",
    "Continuous"
]
abst_pp_params = dict()
abst_con_params = dict()
abst_prol_params = dict()
abst_params_shared = dict()

calculation_assumptions_mapped = {
    "Intent-to-Treat (ITT)": "itt",
    "Responders-Only (RO)": "ro"
}
calculation_assumptions = list(calculation_assumptions_mapped)


def _load_elements():
    st.title("Abstinence Calculator")
    st.markdown("This web app calculates abstinence using the Timeline-Followback data in addiction research. No data "
                "will be saved or shared.")
    st.markdown("For advanced use cases and detailed API references, please refer to the package's "
                "[GitHub](https://github.com/ycui1-mda/abstcal) page for more information.")
    _load_tlfb_elements()
    _load_visit_elements()
    _load_bio_elements()
    _load_cal_elements()


def _load_tlfb_elements():
    st.header("TLFB Data")
    st.markdown("""The dataset should have three columns: __*id*__, 
    __*date*__, and __*amount*__. The id column stores the subject ids, each of which should 
    uniquely identify a study subject. The date column stores the dates when daily substance 
    uses are collected. The amount column stores substance uses for each day. Supported file 
    formats include comma-separated (.csv), tab-delimited (.txt), and Excel spreadsheets (.xls, .xlsx).  \n\n
      \n\nid | date | amount 
    ------------ | ------------- | -------------
    1000 | 02/03/2019 | 10
    1000 | 02/04/2019 | 8
    1000 | 02/05/2019 | 12
      \n\n
    """)
    tlfb_data_params['filepath'] = st.text_input(
        "Specify the file path to the TLFB data on your computer.",
        value=test_tlfb_filepath
    )
    tlfb_subjects = list()
    if tlfb_data_params['filepath']:
        tlfb_data = ac.TLFBData(tlfb_data_params['filepath'])
        tlfb_subjects = sorted(tlfb_data.subject_ids)

    with st.beta_expander("TLFB Data Processing Advanced Configurations"):
        st.write("1. Specify the cutoff value for abstinence")
        tlfb_data_params["cutoff"] = st.number_input(
            "Equal or below the specified value is considered abstinent.",
            step=None
        )
        st.write("2. Subjects used in the abstinence calculation.")
        use_all_subjects = st.checkbox(
            "Use all subjects in the TLFB data",
            value=True
        )
        if use_all_subjects:
            tlfb_data_params["subjects"] = "all"
        else:
            tlfb_data_params["subjects"] = st.multiselect(
                "Choose the subjects of the TLFB data whose abstinence will be calculated.",
                tlfb_subjects,
                default=tlfb_subjects
            )
        st.write("3. TLFB Missing Data Imputation (missing data are those data gaps between study dates)")
        tlfb_imputation_mode = tlfb_imputation_options_mapped[st.selectbox(
            "Select your option",
            tlfb_imputation_options,
            index=1,
            key="tlfb_imputation_mode"
        )]
        if tlfb_imputation_mode is not None:
            imputation_gap_col, imputation_last_record_col, imputation_value_col = st.beta_columns(3)
            tlfb_data_params["imputation_gap_limit"] = imputation_gap_col.number_input(
                "Maximal Gap for Imputation (days)",
                value=30,
                step=1
            )
            tlfb_data_params["imputation_last_record"] = imputation_last_record_col.text_input(
                "Last Record Interpolation (fill foreword or a numeric value)",
                value="ffill"
            )
            if tlfb_imputation_mode == 0:
                tlfb_imputation_mode = imputation_value_col.number_input(
                    "Specify the value to fill the missing TLFB records.")
        tlfb_data_params["imputation_mode"] = tlfb_imputation_mode
        st.write("4. TLFB Duplicate Records Action (duplicates are those with the same id and date)")
        tlfb_data_params["duplicate_mode"] = duplicate_options_mapped[st.selectbox(
            "Select your option",
            duplicate_options,
            index=len(duplicate_options) - 2,
            key="tlfb_duplicate_mode"
        )]
        st.write("5. TLFB Outliers Actions (outliers are those lower than the min or higher than the max)")
        tlfb_data_params["outliers_mode"] = outlier_options_mapped[st.selectbox(
            "Select your option",
            outlier_options,
            key="tlfb_outliers_mode"
        )]
        if tlfb_data_params["outliers_mode"] is not None:
            left_col, right_col = st.beta_columns(2)
            tlfb_data_params["allowed_min"] = left_col.number_input(
                "Allowed Minimal Daily Value",
                step=None,
                value=0.0
            )
            tlfb_data_params["allowed_max"] = right_col.number_input(
                "Allowed Maximal Daily Value",
                step=None,
                value=100.0
            )


def _load_visit_elements():
    st.header("Visit Data")
    st.markdown("""It needs to be in one of the following two formats.  \n\n**The long format.** 
    The dataset should have three columns: __*id*__, __*visit*__, 
    and __*date*__. The id column stores the subject ids, each of which should uniquely 
    identify a study subject. The visit column stores the visits. The date column stores 
    the dates for the visits.  \n\nid | visit | date 
    ------------ | ------------- | -------------
    1000 | 0 | 02/03/2019
    1000 | 1 | 02/10/2019
    1000 | 2 | 02/17/2019  \n\n\n\n---
      \n**The wide format.** 
    The dataset should have the id column and additional columns 
    with each representing a visit.  \n\nid | v0 | v1 | v2 | v3 | v4 | v5
    ----- | ----- | ----- | ----- | ----- | ----- | ----- |
    1000 | 02/03/2019 | 02/10/2019 | 02/17/2019 | 03/09/2019 | 04/07/2019 | 05/06/2019
    1001 | 02/05/2019 | 02/13/2019 | 02/20/2019 | 03/11/2019 | 04/06/2019 | 05/09/2019""")
    visit_data_params['filepath'] = st.text_input(
        "Specify the file path to the Visit data on your computer.",
        value=test_visit_filepath
    )
    visit_data_params['file_format'] = st.selectbox("Specify the file format", visit_file_formats).lower()
    visits = list()
    if visit_data_params['filepath']:
        visit_data = ac.VisitData(visit_data_params['filepath'])
        visits = sorted(visit_data.visits)

    with st.beta_expander("Visit Data Processing Advanced Configurations"):
        st.write("1. Specify the expected order of the visits (for data normality check)")
        visit_data_params['expected_visits'] = st.multiselect(
            "Please adjust the order accordingly",
            visits,
            default=visits
        )
        st.write("2. Visit Missing Dates Imputation")
        visit_data_params["imputation_mode"] = visit_imputation_options_mapped[st.selectbox(
            "Select your option",
            visit_imputation_options,
            index=1,
            key="visit_imputation_mode"
        )]
        if visit_data_params["imputation_mode"] is not None:
            visit_data_params["anchor_visit"] = st.selectbox(
                "Anchor Visit for Imputation",
                visit_data_params['expected_visits'],
                index=0
            )
        st.write("3. Visit Duplicate Records Action")
        visit_data_params["duplicate_mode"] = duplicate_options_mapped[st.selectbox(
            "Select your option",
            duplicate_options,
            index=len(duplicate_options) - 2,
            key="visit_duplicate_mode"
        )]
        st.write("4. Visit Outliers Action (outliers are those lower than the min or higher than the max)")
        visit_data_params["outliers_mode"] = outlier_options_mapped[st.selectbox(
            "Select your option",
            outlier_options,
            key="visit_outliers_mode"
        )]
        if visit_data_params["outliers_mode"] is not None:
            left_col, right_col = st.beta_columns(2)
            visit_data_params["allowed_min"] = left_col.date_input(
                "Allowed Minimal Visit Date",
                value=datetime.datetime.today() - datetime.timedelta(days=365*10)
            )
            visit_data_params["allowed_max"] = right_col.date_input(
                "Allowed Maximal Visit Date",
                value=None
            )


def _load_bio_elements():
    st.header("Biochemical Data (Optional)")
    st.markdown("""
    If your study has collected biochemical verification data, such as carbon monoxide for smoking or breath alcohol 
    concentration for alcohol intervention, these biochemical data can be integrated into the TLFB data. 
    In this way, non-honest reporting can be identified (e.g., self-reported of no use, but biochemically un-verified), 
    the self-reported value will be overridden, and the updated record will be used in later abstinence calculation.
    
    Please note that the biochemical measures dataset should have the same data structure as you TLFB dataset. 
    In other words, it should have three columns: id, date, and amount.
    
    id | date | amount 
    ------------ | ------------- | -------------
    1000 | 02/03/2019 | 4
    1000 | 02/11/2019 | 6
    1000 | 03/04/2019 | 10
    ***
    """)
    bio_data_params['filepath'] = st.text_input(
        "Specify the file path to the Biochemical data on your computer.",
        value=test_bio_filepath
    )
    with st.beta_expander("Biochemical Data Processing Advanced Configurations"):
        st.write("1. Specify the cutoff value for biochemically-verified abstinence")
        bio_data_params["cutoff"] = st.number_input(
            "Equal or below the specified value is considered abstinent.",
            value=0.0,
            step=None,
            key="bio_cutoff"
        )
        st.write("2. Override False Negative TLFB Records")
        bio_data_params["overridden_amount"] = st.number_input(
            "Specify the TLFB amount to override a false negative TLFB record. "
            "(self-report TLFB records say abstinent, but biochemical data invalidate them).",
            value=tlfb_data_params["cutoff"] + 1
        )
        st.write("3. Biochemical Data Interpolation.")
        st.markdown("The calculator will estimate the biochemical levels based on the "
                    "current measures in the preceding days using the half-life.")
        bio_data_params["enable_interpolation"] = st.checkbox(
            "Enable Data Interpolation"
        )
        if bio_data_params["enable_interpolation"]:
            left_col, right_col = st.beta_columns(2)
            bio_data_params["half_life"] = left_col.number_input(
                "Half Life of the Biochemical Measure in Days",
                value=1
            )
            bio_data_params["days_interpolation"] = right_col.number_input(
                "The Number of Days of Imputation",
                value=1,
                step=1
            )
            if bio_data_params["half_life"] == 0:
                raise ValueError("The half life of the biochemical measure should be greater than zero.")


def _load_cal_elements():
    st.header("Calculate Abstinence")
    abst_params_shared["target_dir"] = st.text_input("Specify the directory where your data will be saved.")
    abst_params_shared["assumption"] = st.selectbox("Abstinence Assumption", calculation_assumptions)

    pp_col, prol_col, con_col = columns = st.beta_columns(3)
    abst_params_list = (abst_pp_params, abst_prol_params, abst_con_params)
    abst_var_name_options = ("Infer from configurations", "Specify custom variable names")
    for abst_option, col, abst_params in zip(abst_options, columns, abst_params_list):

        col.write(abst_option)
        abst_params["visits"] = col.multiselect(
            "1. Visits for Abstinence Calculation",
            visit_data_params['expected_visits'],
            key=abst_option
        )
        abst_var_name_option = col.selectbox(
            "2. Abstinent Variable Names",
            options=abst_var_name_options,
            key=abst_option + "_name_option"
        )
        if abst_var_name_option != abst_var_name_options[0]:
            abst_names = col.text_input(
                "Custom Abstinence Variable Names (They should match the number of abstinence variables).",
                key=abst_option + "_name")
        else:
            abst_names = "infer"
        abst_params["abst_var_names"] = abst_names

        if col is pp_col:
            abst_params["days"] = col.text_input(
                "3. Specify a list of the number of days preceding the visit dates. \n"
                "Enter your options and separate them by commas. Example: 7, 14, 21"
            )
        elif col is prol_col:
            abst_params["quit_visit"] = col.selectbox(
                "3. Specify the quit visit",
                visit_data_params['expected_visits']
            )
            abst_params["lapse_definitions"] = col.text_input(
                "4. Specify lapse definitions. Enter your options and separate them"
                "by commas. Example: False, 5 cigs. See GitHub page for more details."
            )
        else:
            abst_params["start_visit"] = col.selectbox(
                "3. Specify the start visit",
                visit_data_params['expected_visits']
            )

    if st.button("Get Abstinence Results"):
        _run_analysis()


def _run_analysis():
    message = f"TLFB: {tlfb_data_params}\n\nVisit: {visit_data_params}\n\nCO: {bio_data_params}"
    st.write(message)
    tlfb_data = ac.TLFBData(
        tlfb_data_params["filepath"],
        tlfb_data_params["cutoff"],
        tlfb_data_params["subjects"]
    )
    # tlfb_data.profile_data(tlfb_data_params["allowed_min"], tlfb_data_params["allowed_max"])
    tlfb_na_number = tlfb_data.drop_na_records()
    st.write(f"Removed Records With N/A Values: {tlfb_na_number}")
    tlfb_duplicates = tlfb_data.check_duplicates(tlfb_data_params["duplicate_mode"])
    st.write(f"Duplicate Records: {tlfb_duplicates}")
    if tlfb_data_params["outliers_mode"] is not None:
        st.write(tlfb_data.recode_outliers(
            tlfb_data_params["allowed_min"],
            tlfb_data_params["allowed_max"],
            tlfb_data_params["outliers_mode"])
        )
    if tlfb_data_params["imputation_mode"] is not None:
        imputation_params = [
            tlfb_data_params["imputation_mode"],
            tlfb_data_params["imputation_last_record"],
            tlfb_data_params["imputation_gap_limit"]
        ]
    if bio_data_params["filepath"]:
        biochemical_data = ac.TLFBData(
            bio_data_params["filepath"],
            bio_data_params["cutoff"]
        )
        if bio_data_params["enable_interpolation"]:
            biochemical_data.interpolate_biochemical_data(
                bio_data_params["half_life"],
                bio_data_params["days_interpolation"]
            )
        biochemical_data.drop_na_records()
        biochemical_data.check_duplicates()
        imputation_params.extend((biochemical_data, str(bio_data_params["overridden_amount"])))

    st.write(tlfb_data.impute_data(*imputation_params))


def _max_width_():
    max_width_str = f"max-width: 1200px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    _max_width_()
    _load_elements()
