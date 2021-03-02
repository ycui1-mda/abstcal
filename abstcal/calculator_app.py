import base64
import datetime
import io
import sys
import pandas as pd
import streamlit as st
from web_utils import get_saved_session
from tlfb_data import TLFBData
from visit_data import VisitData
from abstinence_calculator import AbstinenceCalculator

abstcal_version = '0.7.3'
update_date = datetime.date.today().strftime("%m/%d/%Y")

# Hide tracebacks
sys.tracebacklimit = 0
session_state = get_saved_session(tlfb_data=None, visit_data=None)

# Shared options
duplicate_options_mapped = {
    "Keep the minimal only": "min",
    "Keep the maximal only": "max",
    "Keep the mean only": "mean",
    "Remove all duplicates": False
}
duplicate_options = list(duplicate_options_mapped)
duplicate_options_mapped_reversed = {value: key for key, value in duplicate_options_mapped.items()}

outlier_options_mapped = {
    "Don't examine outliers": None,
    "Remove the outliers": True,
    "Impute the outliers with the bounding values": False
}
outlier_options = list(outlier_options_mapped)
outlier_options_mapped_reversed = {value: key for key, value in outlier_options_mapped.items()}

# TLFB data-related params
tlfb_data_params = dict.fromkeys([
    "data",
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
    "Don't impute missing records":               None,
    "Linear (a linear interpolation in the gap)": "linear",
    "Uniform (the same value in the gap)":        "uniform",
    "Specified Value":                            0
}
tlfb_imputation_options = list(tlfb_imputation_options_mapped)

# Visit data-related params
visit_data_params = dict.fromkeys([
    "data",
    "data_format",
    "expected_visits",
    "subjects",
    "duplicate_mode",
    "imputation_mode",
    "anchor_visit",
    "allowed_min",
    "allowed_max",
    "outliers_mode"
])
visit_data_formats = [
    "Long",
    "Wide"
]
visit_imputation_options_mapped = {
    "Don't impute dates":                                None,
    "The most frequent interval since the anchor visit": "freq",
    "The mean interval since the anchor visit":          "mean"
}
visit_imputation_options = list(visit_imputation_options_mapped)
visit_imputation_options_mapped_reversed = {value: key for key, value in visit_imputation_options_mapped.items()}

# Biochemical data-related params
bio_data_params = dict.fromkeys([
    "data",
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

# Calculator-related params
abst_pp_params = dict()
abst_cont_params = dict()
abst_prol_params = dict()
abst_params_shared = dict()

calculation_assumptions_mapped = {
    "Intent-to-Treat (ITT)": "itt",
    "Responders-Only (RO)":  "ro"
}
calculation_assumptions = list(calculation_assumptions_mapped)
abst_options = [
    "Point-Prevalence",
    "Prolonged",
    "Continuous"
]


def _load_elements():
    st.title("Abstinence Calculator")
    _load_overview_elements()
    st.markdown("***")
    _load_tlfb_elements()
    st.markdown("***")
    _load_visit_elements()
    st.markdown("***")
    _load_cal_elements()


def _load_overview_elements():
    st.markdown("This web app calculates abstinence using the Timeline-Followback data in addiction research. No data "
                "will be saved or shared.")
    st.markdown("For advanced use cases and detailed API references, please refer to the package's "
                "[GitHub](https://github.com/ycui1-mda/abstcal) page for more information.")
    st.markdown(f"Current Version of abstcal: __{abstcal_version}__")
    st.markdown(f"Last Update Date: __{update_date}__")
    st.markdown("**Disclaimer**: Not following the steps or variation in your source data may result in incorrect "
                "abstinence results. Please verify your results for accuracy.")
    st.subheader("Basic Steps:")
    st.markdown("""
    1. Process the TLFB data in Section 1
        * The TLFB data file needs to be prepared accordingly as instructed below.
        * You can optionally integrate any biochemical measures for abstinence verification purposes.
    2. Process the Visit data in Section 2
        * The Visit data file needs to be prepared accordingly as instructed below.
    3. Calculate abstinence results in Section 3
        * It supports continuous, point-prevalence, and prolonged abstinence.
    """)

    st.subheader("Advanced Settings")
    st.markdown("If you want to re-do your calculation, please press the following button. If you need to update the "
                "uploaded files, please remove them manually and re-upload the new ones.")

    if st.button("Reset Data"):
        session_state.tlfb_data = None
        session_state.visit_data = None

    st.markdown("Generate a Python script that you can integrate into your current data processing workflow.")
    if st.button("Generate Python Script"):
        calculation_script = "This is the generated script"
        _pop_download_link(calculation_script, 'calculate_abst', 'Python Script', file_type='py')


def _load_tlfb_elements():
    st.header("Section 1. TLFB Data")
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
    container = st.beta_container()
    uploaded_file = container.file_uploader(
        "Specify the TLFB data file on your computer.",
        ["csv", "txt", "xlsx"]
    )
    tlfb_subjects = list()
    if uploaded_file:
        data = io.BytesIO(uploaded_file.getbuffer())
        tlfb_data_params["data"] = df = pd.read_csv(data)
        container.write(df)
        tlfb_data = TLFBData(df)
        tlfb_subjects = sorted(tlfb_data.subject_ids)
    else:
        container.write("The TLFB data are shown here after loading.")

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
        imputation_mode_col, imputation_value_col = st.beta_columns(2)
        tlfb_imputation_mode = tlfb_imputation_options_mapped[imputation_mode_col.selectbox(
            "Select your option",
            tlfb_imputation_options,
            index=1,
            key="tlfb_imputation_mode"
        )]
        if tlfb_imputation_mode == 0:
            tlfb_imputation_mode = imputation_value_col.number_input(
                "Specify the value to fill the missing TLFB records.")

        if tlfb_imputation_mode is not None:
            enable_gap = st.checkbox("Set limit for the maximal gap for imputation")
            if enable_gap:
                tlfb_data_params["imputation_gap_limit"] = st.number_input(
                    "Maximal Gap for Imputation (days)",
                    value=30,
                    step=1
                )
            enable_last_record = st.checkbox(
                "Interpolate Last Record For Each Subject",
                value=True
            )
            if enable_last_record:
                tlfb_data_params["imputation_last_record"] = st.text_input(
                    "Last Record Interpolation (fill foreword or a numeric value)",
                    value="ffill"
            )
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
        st.write("6. Biochemical Data for Abstinence Verification (Optional)")
        has_bio_data = st.checkbox("Integrate Biochemical Data For Abstinence Calculation")
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
        if has_bio_data:
            bio_container = st.beta_container()
            uploaded_file = bio_container.file_uploader(
                "Specify the Biochemical data file on your computer.",
                ["csv", "txt", "xlsx"]
            )
            if uploaded_file:
                data = io.BytesIO(uploaded_file.getbuffer())
                bio_data_params["data"] = pd.read_csv(data)
                bio_container.write(bio_data_params["data"])
            else:
                bio_container.write("The Biochemical data are shown here after loading.")

            st.write("6.1. Specify the cutoff value for biochemically-verified abstinence")
            bio_data_params["cutoff"] = st.number_input(
                "Equal or below the specified value is considered abstinent.",
                value=0.0,
                step=None,
                key="bio_cutoff"
            )
            st.write("6.2. Override False Negative TLFB Records")
            bio_data_params["overridden_amount"] = st.number_input(
                "Specify the TLFB amount to override a false negative TLFB record. "
                "(self-report TLFB records say abstinent, but biochemical data invalidate them).",
                value=tlfb_data_params["cutoff"] + 1
            )
            st.write("6.3. Biochemical Data Interpolation")
            st.markdown("The calculator will estimate the biochemical levels based on the "
                        "current measures in the preceding days using the half-life.")
            bio_data_params["enable_interpolation"] = st.checkbox(
                "Enable Data Interpolation"
            )
            if bio_data_params["enable_interpolation"]:
                left_col, right_col = st.beta_columns(2)
                bio_data_params["half_life"] = left_col.number_input(
                    "Half Life of the Biochemical Measure in Days",
                    min_value=0.0,
                    value=1.0,
                    step=0.01
                )
                bio_data_params["days_interpolation"] = right_col.number_input(
                    "The Number of Days of Interpolation",
                    value=1,
                    step=1
                )
                if bio_data_params["half_life"] == 0:
                    raise ValueError("The half life of the biochemical measure should be greater than zero.")

    processed_data = st.button("Get/Refresh TLFB Data Summary")

    if processed_data or session_state.tlfb_data is not None:
        _process_tlfb_data()


def _process_tlfb_data():
    tlfb_df = tlfb_data_params["data"]
    if tlfb_df is None:
        raise ValueError("Please specify the TLFB data in the file uploader above.")

    tlfb_data = TLFBData(
        tlfb_df,
        tlfb_data_params["cutoff"],
        tlfb_data_params["subjects"]
    )
    session_state.tlfb_data = tlfb_data
    abst_params_shared["tlfb_data"] = tlfb_data
    _load_data_summary(tlfb_data, tlfb_data_params)

    tlfb_imputation_mode = tlfb_data_params["imputation_mode"]
    if tlfb_imputation_mode is not None:
        st.write("Imputation Summary")
        imputation_params = [
            tlfb_data_params["imputation_mode"],
            tlfb_data_params["imputation_last_record"],
            tlfb_data_params["imputation_gap_limit"]
        ]
        bio_messages = list()
        if bio_data_params["data"] is not None:
            bio_messages.append("Note: Biochemical Data are used for TLFB imputation")
            biochemical_data = TLFBData(
                bio_data_params["data"],
                bio_data_params["cutoff"]
            )
            bio_messages.append(f"Cutoff: {bio_data_params['cutoff']}")
            bio_messages.append(f"Interpolation: {bio_data_params['enable_interpolation']}")
            if bio_data_params["enable_interpolation"]:
                bio_messages.append(f'Half Life in Days: {bio_data_params["half_life"]}')
                bio_messages.append(f'Interpolated Days: {bio_data_params["days_interpolation"]}')
                bio_messages.append(f'Overridden Amount for False Negative TLFB Records: '
                                    f'{bio_data_params["overridden_amount"]}')
                biochemical_data.interpolate_biochemical_data(
                    bio_data_params["half_life"],
                    bio_data_params["days_interpolation"]
                )
            biochemical_data.drop_na_records()
            biochemical_data.check_duplicates()
            imputation_params.extend((biochemical_data, str(bio_data_params["overridden_amount"])))

        st.write(tlfb_data.impute_data(*imputation_params))
        if bio_messages:
            st.write("; ".join(bio_messages))
    else:
        st.write("Imputation Action: None")


def _load_visit_elements():
    st.header("Section 2. Visit Data")
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
    container = st.beta_container()
    uploaded_file = container.file_uploader(
        "Specify the Visit data file on your computer.",
        ["csv", "txt", "xlsx"]
    )
    visit_data_params['data_format'] = st.selectbox("Specify the file format", visit_data_formats).lower()
    visits = list()
    if uploaded_file:
        data = io.BytesIO(uploaded_file.getbuffer())
        visit_data_params["data"] = df = pd.read_csv(data)
        container.write(df)
        visit_data = VisitData(df, visit_data_params['data_format'])
        visits = sorted(visit_data.visits)
        visit_data_params['expected_visits'] = visits
        visit_subjects = sorted(visit_data.subject_ids)
    else:
        container.write("The Visit data are shown here after loading.")

    with st.beta_expander("Visit Data Processing Advanced Configurations"):
        st.write("1. Specify the expected order of the visits (for data normality check)")
        visit_data_params['expected_visits'] = st.multiselect(
            "Please adjust the order accordingly",
            visits,
            default=visits
        )
        st.write("2. Subjects used in the abstinence calculation.")
        use_all_subjects = st.checkbox(
            "Use all subjects in the Visit data",
            value=True
        )
        if use_all_subjects:
            visit_data_params["subjects"] = "all"
        else:
            visit_data_params["subjects"] = st.multiselect(
                "Choose the subjects of the TLFB data whose abstinence will be calculated.",
                visit_subjects,
                default=visit_subjects
            )
        st.write("3. Visit Missing Dates Imputation")
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
        st.write("4. Visit Duplicate Records Action")
        visit_data_params["duplicate_mode"] = duplicate_options_mapped[st.selectbox(
            "Select your option",
            duplicate_options,
            index=len(duplicate_options) - 2,
            key="visit_duplicate_mode"
        )]
        st.write("5. Visit Outliers Action (outliers are those lower than the min or higher than the max)")
        visit_data_params["outliers_mode"] = outlier_options_mapped[st.selectbox(
            "Select your option",
            outlier_options,
            key="visit_outliers_mode"
        )]
        if visit_data_params["outliers_mode"] is not None:
            left_col, right_col = st.beta_columns(2)
            visit_data_params["allowed_min"] = left_col.date_input(
                "Allowed Minimal Visit Date",
                value=datetime.datetime.today() - datetime.timedelta(days=365 * 10)
            )
            visit_data_params["allowed_max"] = right_col.date_input(
                "Allowed Maximal Visit Date",
                value=None
            )

    processed_data = st.button("Get/Refresh Visit Data Summary")

    if processed_data or session_state.visit_data is not None:
        _process_visit_data()


def _process_visit_data():
    visit_df = visit_data_params["data"]
    if visit_df is None:
        raise ValueError("Please upload your Visit data and make sure it's loaded successfully.")

    st.write("Data Overview")
    visit_data = VisitData(
        visit_df,
        visit_data_params["data_format"],
        visit_data_params["expected_visits"],
        visit_data_params["subjects"]
    )
    session_state.visit_data = visit_data
    abst_params_shared["visit_data"] = visit_data
    _load_data_summary(visit_data, visit_data_params)
    imputation_mode = visit_data_params["imputation_mode"]
    if imputation_mode is not None:
        st.write(f'Imputation Summary (Parameters: anchor visit={visit_data_params["anchor_visit"]}, '
                 f'mode={visit_imputation_options_mapped_reversed[imputation_mode]})')
        st.write(visit_data.impute_data(visit_data_params["anchor_visit"], visit_data_params["imputation_mode"]))
    else:
        st.write("Imputation Action: None")
    st.write("Visit Attendance Summary")
    st.write(visit_data.get_retention_rates())


def _load_data_summary(data, data_params):
    st.subheader("Data Overview")
    data_all_summary, data_subject_summary = \
        data.profile_data(data_params["allowed_min"], data_params["allowed_max"])
    st.write(data_all_summary)
    st.write(data_subject_summary)

    na_number = data.drop_na_records()
    st.write(f"Removed Records With N/A Values Count: {na_number}")

    duplicate_mode = data_params["duplicate_mode"]
    duplicates = data.check_duplicates(duplicate_mode)
    st.write(f"Duplicate Records Count: {duplicates}; "
             f"Duplicate Records Action: {duplicate_options_mapped_reversed[duplicate_mode]}")

    outliers_mode = data_params["outliers_mode"]
    if outliers_mode is not None:
        st.write(f"Outliers Summary for Action: {outlier_options_mapped_reversed[outliers_mode]}")
        st.write(data.recode_outliers(
            data_params["allowed_min"],
            data_params["allowed_max"],
            data_params["outliers_mode"])
        )
    else:
        st.write("Outliers Action: None")



def _load_cal_elements():
    st.header("Section 3. Calculate Abstinence")
    abst_params_shared["mode"] = calculation_assumptions_mapped[
        st.selectbox("Abstinence Assumption Mode", calculation_assumptions)
    ]
    abst_params_shared["including_end"] = st.checkbox(
        "Including each of the visit dates as the end of the time window examined, otherwise the time window of concern"
        " will end the day before the visit date."
    )
    pp_col, prol_col, cont_col = columns = st.beta_columns(3)
    abst_params_list = (abst_pp_params, abst_prol_params, abst_cont_params)
    abst_var_name_options = ("Infer automatically", "Specify custom variable names")
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
            days_text = col.text_input(
                "3. Specify a list of the number of days preceding the visit dates. \n"
                "Enter your options and separate them by commas. Example: 7, 14, 21"
            )
            abst_params["days"] = eval(f"[{days_text}]")
        elif col is prol_col:
            abst_params["quit_visit"] = col.selectbox(
                "3. Specify the quit visit",
                visit_data_params['expected_visits']
            )
            lapse_text = col.text_input(
                "4. Specify lapse definitions. Enter your options and separate them "
                "by commas. When lapses are not allowed, its definition is False. For all definitions, "
                "please enclose each of them within single quotes. "
                "Example: 'False', '5 cigs', '5 cigs/14 days'. "
                "See GitHub page for more details."
            )
            definitions = eval(f"[{lapse_text}]")
            for i, definition in enumerate(definitions):
                if definition.lower() == "false":
                    definitions[i] = False
            abst_params["lapse_definitions"] = definitions
            abst_params["grace_period"] = col.slider(
                "5. Specify the grace period in days (default: 14 days)",
                value=14,
                min_value=1,
                max_value=100,
                step=1
            )
        else:
            abst_params["start_visit"] = col.selectbox(
                "3. Specify the start visit",
                visit_data_params['expected_visits']
            )

    if st.button("Get Abstinence Results"):
        _calculate_abstinence()


def _calculate_abstinence():
    st.header("Calculation Results Summary")
    if session_state.tlfb_data is None or session_state.visit_data is None:
        raise ValueError("Please process the TLFB and Visit data first.")

    calculator = AbstinenceCalculator(session_state.tlfb_data, session_state.visit_data)
    calculation_results = list()
    if abst_pp_params["visits"]:
        calculation_results.append(calculator.abstinence_pp(
            abst_pp_params["visits"],
            abst_pp_params["days"],
            abst_pp_params["abst_var_names"],
            abst_params_shared["including_end"],
            abst_params_shared["mode"]
        ))
    if abst_prol_params["visits"]:
        calculation_results.append(calculator.abstinence_prolonged(
            abst_prol_params["quit_visit"],
            abst_prol_params["visits"],
            abst_prol_params["lapse_definitions"],
            abst_prol_params["grace_period"],
            abst_prol_params["abst_var_names"],
            abst_params_shared["including_end"],
            abst_params_shared["mode"]
        ))
    if abst_cont_params["visits"]:
        calculation_results.append(calculator.abstinence_cont(
            abst_cont_params["start_visit"],
            abst_cont_params["visits"],
            abst_cont_params["abst_var_names"],
            abst_params_shared["including_end"],
            abst_params_shared["mode"]
        ))
    st.subheader("Abstinence Calculated for Subjects")
    st.markdown(f"{sorted(calculator.subject_ids)}")
    abst_df = calculator.merge_abst_data([x[0] for x in calculation_results])
    st.subheader("Abstinence Data")
    st.write(abst_df)
    _pop_download_link(abst_df, "abstinence_data", "Abstinence Data", True)

    lapse_df = calculator.merge_lapse_data([x[1] for x in calculation_results])
    st.subheader("Lapse Data")
    st.write(lapse_df)
    _pop_download_link(lapse_df, "lapse_data", "Lapse Data", False)


def _pop_download_link(df, filename, link_name, kept_index=False, file_type='csv'):
    if isinstance(df, pd.DataFrame):
        data_to_download = df.to_csv(index=kept_index)
    else:
        data_to_download = df
    b64 = base64.b64encode(data_to_download.encode()).decode()
    href = f'<a href="data:file/{file_type};base64,{b64}" download="{filename}.{file_type}">Download {link_name}</a>'
    st.markdown(href, unsafe_allow_html=True)


def _max_width_(width):
    st.markdown(
        f"""<style>.reportview-container .main .block-container{{max-width: {width}px;}}</style>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    _max_width_(1200)
    _load_elements()
