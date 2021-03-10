import base64
import datetime
import io
import os
import sys
import pandas as pd
import streamlit as st
sys.path.append(os.getcwd())
from abstcal.calculator_web_utils import get_saved_session
from abstcal.tlfb_data import TLFBData
from abstcal.visit_data import VisitData
from abstcal.abstinence_calculator import AbstinenceCalculator


abstcal_version = '0.7.3'
update_date = datetime.date.today().strftime("%m/%d/%Y")
sidebar = st.sidebar
supported_file_types = ["csv", "xls", "xlsx"]

# Hide tracebacks
# sys.tracebacklimit = 0
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
    "use_raw_date",
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
    "use_raw_date",
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


def _create_df_from_upload(file_upload):
    uploaded_df = None
    if file_upload is None:
        return uploaded_df
    uploaded_data = io.BytesIO(file_upload.getbuffer())
    try:
        uploaded_df = pd.read_csv(uploaded_data)
    except:
        uploaded_df = pd.read_excel(uploaded_data)
    finally:
        return uploaded_df


def _load_elements():
    _show_preprocessing_elements()
    st.title("Abstinence Calculator")
    _load_overview_elements()
    st.markdown("***")
    _load_tlfb_elements()
    st.markdown("***")
    _load_visit_elements()
    st.markdown("***")
    _load_cal_elements()
    st.markdown("***")
    _load_script_element()


def _show_preprocessing_elements():
    sidebar.header("Pre-Processing Tools")
    _create_data_conversion_tool()
    sidebar.markdown("***")
    _create_date_masking_tool()
    sidebar.markdown("***")
    _create_additional_visit_dates()

def _create_data_conversion_tool():
    sidebar.subheader("Data Conversion (Wide to Long)")
    sidebar.write(
        "The calculator uses datasets in the long format. Convert your wide-formatted data to the long format."
    )
    conversion_tool = sidebar.beta_expander("Conversion Tool Settings", False)
    wide_file = conversion_tool.file_uploader("Wide Formatted Data", type=supported_file_types)
    data_source_type = conversion_tool.radio("Data Source", ["Visit", "TLFB"])
    subject_col_name = conversion_tool.text_input("Subject ID Variable Name", "id")
    if data_source_type == "tlfb":
        conversion_tool.write("The resulted TLFB data will have three columns: id, date, and amount.")
    else:
        conversion_tool.write("The resulted Visit data will have three columns: id, visit, and date.")
    convert_button = conversion_tool.button("Convert")
    if convert_button:
        if wide_file:
            wide_df = _create_df_from_upload(wide_file)
            long_df = AbstinenceCalculator.from_wide_to_long(wide_df, data_source_type.lower(), subject_col_name)
            conversion_tool.write(long_df)
            _pop_download_link(long_df, f"wide_{data_source_type}", "Long-Formatted Data",
                               container=conversion_tool)
        else:
            conversion_tool.error("Please upload your dataset first.")


def _create_date_masking_tool():
    sidebar.subheader("Date Masking Tool")
    sidebar.write(
        "Mask the dates in the Visit and TLFB data using an anchor visit or arbitrary date for all subjects"
    )
    masking_tool = sidebar.beta_expander("Masking Tool Settings", False)
    masking_tool.write("Visit Data (Long Format, 3 columns: id, visit, date)")
    visit_file = masking_tool.file_uploader("Upload Visit Data", type=supported_file_types)
    visit_df = _create_df_from_upload(visit_file)
    masking_tool.write("TLFB Data (Long Format, 3 columns: id, date, amount)")
    tlfb_file = masking_tool.file_uploader("Upload TLFB Data", type=supported_file_types)
    tlfb_df = _create_df_from_upload(tlfb_file)

    masking_tool.write("Optional Biochemical Data (Long Format, 3 columns: id, date, amount)")
    bio_file = masking_tool.file_uploader("Upload Bio Data", type=supported_file_types)
    bio_df = _create_df_from_upload(bio_file)

    masking_reference_options = ["Anchor Visit", "Arbitrary Date"]
    masking_reference_index = masking_reference_options.index(
        masking_tool.radio("Reference Using:", masking_reference_options)
    )
    reference = None
    if masking_reference_index == 0:
        if visit_df is not None:
            reference = masking_tool.selectbox("Select Visit", visit_df['visit'].unique())
            masking_tool.write(f"Each subject will use his/her {reference}'s date to calculate day counters.")
    else:
        reference = masking_tool.date_input("Arbitrary Date")
        masking_tool.write(f"All subjects will share {reference} as the reference to calculate day counters.")

    if masking_tool.button("Mask Date"):
        if visit_df is None or tlfb_df is None:
            masking_tool.error("Please update both files to mask dates consistently.")
        else:
            masked_tlfb_df, *masked_bio_df, masked_visit_df = \
                AbstinenceCalculator.mask_dates(tlfb_df, bio_df, visit_df, reference)
            _pop_download_link(masked_tlfb_df, "masked_tlfb", "Masked TLFB Data",
                               container=masking_tool)
            if masked_bio_df:
                _pop_download_link(masked_tlfb_df[0], "masked_bio", "Masked Bio Data",
                                   container=masking_tool)
            _pop_download_link(masked_visit_df, "masked_visit", "Masked Visit Data",
                               container=masking_tool)


@st.cache(allow_output_mutation=True)
def _get_visit_dates():
    return pd.DataFrame(columns=["New", "Ref", "+ Days"])


def _create_additional_visit_dates():
    sidebar.subheader("Visit Dates Creation Tool")
    sidebar.write(
        "Create extra visit dates based on existing visit days"
    )
    visit_tool = sidebar.beta_expander("Creation Date Settings", True)
    visit_file = visit_tool.file_uploader("Upload Visit Data", type=supported_file_types, key="visit_tool_file")
    visit_df = _create_df_from_upload(visit_file)
    if (visit_df is not None) and (not visit_df.empty):
        dates_options = ["Using Actual Dates", "Using Day Counters"]
        using_date = dates_options.index(visit_tool.radio("Specify the date column's data", dates_options)) == 0
        new_visit_name = visit_tool.text_input("New Visit Name (Should be distinct)")
        reference_visit = visit_tool.selectbox("Reference Visit", visit_df['visit'].unique())
        days_to_add = visit_tool.number_input("Days to Add", value=0)
        dates_df = _get_visit_dates()
        if visit_tool.button("Add Row"):
            if new_visit_name:
                dates_df.loc[dates_df.shape[0]] = [new_visit_name, reference_visit, days_to_add]
            else:
                visit_tool.error("New visit name can't be empty.")
        dates_df.drop_duplicates('New', inplace=True, ignore_index=True)
        if not dates_df.empty:
            to_drop = visit_tool.multiselect("Remove Unwanted Rows", dates_df.index)
            if visit_tool.button("Drop Rows"):
                dates_df.drop(to_drop, inplace=True)
                dates_df.reset_index(drop=True, inplace=True)
            visit_tool.write("Overview of The Parameters of Extra Visit Dates")
            visit_tool.write(dates_df)
            if visit_tool.button("Create Output Data"):
                dates_list = list(dates_df.to_records(index=False))
                updated_visit_df = VisitData.add_additional_visit_dates(visit_df, dates_list, using_date)
                _pop_download_link(updated_visit_df, "updated_visit", "Updated Visit Data",
                                   container=visit_tool)


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

    st.subheader("Automatic Code Generation (Optional Feature)")
    st.markdown("At the end of this page, after specifying the parameters (you don't have to upload any data for "
                "privacy and security concerns or other reasons), you can generate a Python script for off-line use.")
    spss_link = "https://www.ibm.com/support/knowledgecenter/en/SSLVMB_24.0.0/spss/" \
                "programmability_option/python_scripting_intro.html"
    st.markdown(f"""
    * __Python Users__: you can use the script natively in your code.
    * __R Users__: the reticulate package allows you to use Python 
    ([Use Python in R](https://cran.r-project.org/web/packages/reticulate/))
    * __Stata Users__: Stata 16.0+ has integrated Python compatibility 
    ([Call Python from Stata](https://www.stata.com/python/))
    * __SAS Users__: SAS 9.4M6 supports Python-based functionalities
    ([Using Python functions inside SAS programs]
    (https://blogs.sas.com/content/sgf/2019/06/04/using-python-functions-inside-sas-programs/))
    * __SPSS Users__: you can run Python script using the GUI 
    ([Run Python script in SPSS]({spss_link}))
    """)
    st.markdown("Non-Python users need to set up the Python environment with the necessary dependencies installed. If "
                "you're not sure about this, it's recommended that you use this web app directly. If you're able to "
                "integrate the Python script in your workflow using a different language, you're welcome to share "
                "your experience here to benefit the research community.")
    st.markdown("If you're interested in creating a package in other statistical languages (e.g., R), please feel "
                "free to do so, and we're happy to collaborate.")


def _generate_calculation_script():
    script_lines = list()
    script_lines.append("# This script is automatically generated by the abstcal web app based on user input.\n"
                        "# Please carefully review the script and adjust applicable parameters before you run it.\n"
                        "# Author: Yong Cui, Ph.D.; Email: ycui1@mdanderson.org\n"
                        "# Please feel free to contact me if you need any help.")
    script_lines.append("from abstcal import TLFBData, VisitData, AbstinenceCalculator")

    needed_tlfb_data_params = tlfb_data_params.copy()
    del needed_tlfb_data_params['data']
    script_lines.append("################################################\n"
                        "# Processing TLFB Data\n"
                        "################################################")
    script_lines.append(f'# The parameters for processing the TLFB data\n'
                        f'# Please update the parameters as applicable.\n'
                        f'tlfb_data_params = {needed_tlfb_data_params}')
    script_lines.append('# The path to the TLFB data on your computer\n'
                        'tlfb_source = "Please specify the file path to the TLFB data"')
    script_lines.append("""# Create the TLFBData instance\ntlfb_data = TLFBData(
    tlfb_source,
    tlfb_data_params["cutoff"],
    tlfb_data_params["subjects"],
    tlfb_data_params["use_raw_date"]
)""")
    script_lines.append("# Profile the data\n"
                        "tlfb_data.profile_data(tlfb_data_params['allowed_min'], tlfb_data_params['allowed_max'])")
    script_lines.append("# Drop any records with missing data\n"
                        "tlfb_data.drop_na_records()\n\n"
                        "# Remove any duplicates\n"
                        "tlfb_data.check_duplicates(tlfb_data_params['duplicate_mode'])\n\n"
                        "# Recode any outliers\n"
                        "tlfb_data.recode_outliers(tlfb_data_params['allowed_min'], tlfb_data_params['allowed_min'],"
                        "tlfb_data_params['outliers_mode'])")
    if tlfb_data_params["imputation_mode"] is not None:
        script_lines.append("""# Impute the TLFB data\nimputation_params = [
    tlfb_data_params["imputation_mode"],
    tlfb_data_params["imputation_last_record"],
    tlfb_data_params["imputation_gap_limit"]
]""")
        if bio_data_params["data"] is not None:
            script_lines.append("""# Use biochemical data\nbiochemical_data = TLFBData(
    bio_data_params["data"],
    bio_data_params["cutoff"]
)""")
            if bio_data_params["enable_interpolation"]:
                script_lines.append("""# Interpolate biochemical data\nbiochemical_data.interpolate_biochemical_data(
    bio_data_params["half_life"],
    bio_data_params["days_interpolation"]
)""")
            script_lines.append("# Clean up biochemical\nbiochemical_data.drop_na_records()\n"
                                "biochemical_data.check_duplicates()")
            script_lines.append("# Apply the biochemical data to TLFB data preps\n"
                                "imputation_params.extend((biochemical_data, "
                                "str(bio_data_params['overridden_amount'])))")
        script_lines.append("tlfb_data.impute_data(*imputation_params)")

    script_lines.append("################################################\n"
                        "# Processing Visit Data\n"
                        "################################################")
    needed_visit_data_params = visit_data_params.copy()
    del needed_visit_data_params['data']
    if not needed_visit_data_params['expected_visits']:
        needed_visit_data_params['expected_visits'] = 'infer'
    script_lines.append(f'# The parameters for processing the visit data, you need to update some of them\n'
                        f'visit_data_params = {needed_visit_data_params}')
    script_lines.append('# The path to the visit data on your computer\n'
                        'visit_source = "Please specify the file path to the visit data"')
    script_lines.append("""# Create the VisitData instance\nvisit_data = VisitData(
    visit_source,
    visit_data_params["data_format"],
    visit_data_params["expected_visits"],
    visit_data_params["subjects"],
    visit_data_params["use_raw_date"]
)""")
    script_lines.append("# Profile the data\n"
                        "visit_data.profile_data(visit_data_params['allowed_min'], visit_data_params['allowed_max'])")
    script_lines.append("# Remove any duplicates\n"
                        "visit_data.check_duplicates(visit_data_params['duplicate_mode'])")
    script_lines.append("# Recode any outliers\n"
                        "visit_data.recode_outliers(visit_data_params['allowed_min'], visit_data_params['allowed_min'],"
                        "visit_data_params['outliers_mode'])")
    script_lines.append("# Impute missing dates\nvisit_data.impute_data(visit_data_params['imputation_mode'], "
                        "anchor_visit=visit_data_params['anchor_visit'])")

    script_lines.append("################################################\n"
                        "# Abstinence Calculation\n"
                        "################################################\n"
                        "# Create the calculator using TLFB and visit data\n"
                        "calculator = AbstinenceCalculator(tlfb_data, visit_data)")

    script_lines.append(f"# The shared parameters for abstinence calculation\n"
                        f"abst_params_shared = {abst_params_shared}")
    script_lines.append("# Create a list to hold calculation results\n"
                        "calculation_results = list()")
    script_lines.append(f"# Point-prevalence abstinence\n"
                        f"# Please update the parameters for point-prevalence abstinence calculations\n"
                        f"abst_pp_params = {abst_pp_params}")
    script_lines.append("""abstinence_pp = calculator.abstinence_pp(
    abst_pp_params["visits"],
    abst_pp_params["days"],
    abst_pp_params["abst_var_names"],
    abst_params_shared["including_end"],
    abst_params_shared["mode"]
)
calculation_results.append(abstinence_pp)
""")

    script_lines.append(f"# Point-Prolonged abstinence\n"
                        f"# Please update the parameters for prolonged abstinence calculations\n"
                        f"abst_prol_params = {abst_prol_params}")
    script_lines.append("""abstinence_prol = calculator.abstinence_prolonged(
    abst_prol_params["quit_visit"],
    abst_prol_params["visits"],
    abst_prol_params["lapse_definitions"],
    abst_prol_params["grace_period"],
    abst_prol_params["abst_var_names"],
    abst_params_shared["including_end"],
    abst_params_shared["mode"]
)
calculation_results.append(abstinence_prol)
""")

    script_lines.append(f"# Continuous abstinence\n"
                        f"# Please update the parameters for continuous abstinence calculations\n"
                        f"abst_cont_params = {abst_cont_params}")
    script_lines.append("""abstinence_cont = calculator.abstinence_cont(
    abst_cont_params["start_visit"],
    abst_cont_params["visits"],
    abst_cont_params["abst_var_names"],
    abst_params_shared["including_end"],
    abst_params_shared["mode"]
)
calculation_results.append(abstinence_cont)
""")

    script_lines.append("# Write the results to files on your computer\n"
                        "# Set the file path to save the abstinence results\n"
                        "abst_filepath = 'set the file path'\n"
                        "# Set the file path to save the lapse results\n"
                        "lapse_filepath = 'set the file path'")
    script_lines.append("# Save the abstinence data\n"
                        "calculator.merge_abst_data([x[0] for x in calculation_results], abst_filepath)")
    script_lines.append("# Save the lapse data\n"
                        "calculator.merge_lapse_data([x[1] for x in calculation_results], lapse_filepath)")
    generated_script = '\n\n'.join(script_lines)
    return generated_script


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
        st.write("1. The TLFB data's date column can use either actual dates or arbitrary day counters. Please specify "
                 "the date data type.")
        tlfb_data_params["use_raw_date"] = st.checkbox(
            "Raw dates are used.",
            value=True
        )
        if tlfb_data_params["use_raw_date"]:
            st.write("The TLFB dataset uses the actual dates.")
        else:
            st.write("The TLFB dataset uses arbitrary day counters.")
        st.markdown("***")

        st.write("2. Specify the cutoff value for abstinence")
        tlfb_data_params["cutoff"] = st.number_input(
            "Cutoff Level",
            step=None
        )
        st.write(f"Data records with a value higher than {tlfb_data_params['cutoff']} are considered non-abstinent.")
        st.markdown("***")

        st.write("3. Subjects used in the abstinence calculation.")
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
        st.write(f"Subjects used in the calculation: {tlfb_data_params['subjects']}")
        st.markdown("***")

        st.write("4. TLFB Missing Data Imputation (missing data are gaps having no records between study dates)")
        imputation_summary = dict()
        imputation_mode_col, imputation_value_col = st.beta_columns(2)
        selected_imputation_mode = imputation_mode_col.selectbox(
            "Select your option",
            tlfb_imputation_options,
            index=1,
            key="tlfb_imputation_mode"
        )
        imputation_summary['Imputation Mode'] = selected_imputation_mode
        tlfb_imputation_mode = tlfb_imputation_options_mapped[selected_imputation_mode]
        if tlfb_imputation_mode == 0:
            tlfb_imputation_mode = imputation_value_col.number_input(
                "Specify the value to fill the missing TLFB records.",
                value=0
            )
            imputation_summary['Missing data will be imputed using the value'] = tlfb_imputation_mode

        if tlfb_imputation_mode is not None:
            enable_gap = st.checkbox("Set limit for the maximal gap for imputation")
            if enable_gap:
                tlfb_data_params["imputation_gap_limit"] = st.number_input(
                    "Maximal Gap for Imputation (days)",
                    value=30,
                    step=1
                )
                imputation_summary['Maximally Allowed Gap for Imputation'] = \
                    f'{tlfb_data_params["imputation_gap_limit"]} days'
            else:
                imputation_summary['Enable Gap Limit'] = \
                    "There is no set limit for the maximally allowed gap for imputation."
            enable_last_record = st.checkbox(
                "Interpolate Last Record For Each Subject",
                value=True
            )
            if enable_last_record:
                tlfb_data_params["imputation_last_record"] = st.text_input(
                    "Last Record Interpolation (fill forward or a numeric value)",
                    value="ffill"
                )
                if tlfb_data_params["imputation_last_record"] == "ffill":
                    imputation_summary["Last Record Action"] = "Be filled with the last record for each subject"
                else:
                    imputation_summary["Last Record Filled With Value"] = tlfb_data_params["imputation_last_record"]
            else:
                imputation_summary["Last Record Action"] = "Will not interpolate any records beyond the last records"
        tlfb_data_params["imputation_mode"] = tlfb_imputation_mode
        imputation_summary_text = '\n\n* '.join([f'{key}: {value}' for key, value in imputation_summary.items()])
        st.markdown(f"_Summary of the Imputation Parameters_\n\n* {imputation_summary_text}")
        st.markdown("***")

        st.write("5. TLFB Duplicate Records Action (duplicates are those with the same id and date)")
        selected_duplicate_mode = st.selectbox(
            "Select your option",
            duplicate_options,
            index=len(duplicate_options) - 2,
            key="tlfb_duplicate_mode"
        )
        tlfb_data_params["duplicate_mode"] = duplicate_options_mapped[selected_duplicate_mode]
        st.write(f"Duplicate Records: {selected_duplicate_mode}")
        st.markdown("***")

        st.write("6. TLFB Outliers Actions (outliers are those lower than the min or higher than the max)")
        selected_outlier_mode = st.selectbox(
            "Select your option",
            outlier_options,
            key="tlfb_outliers_mode"
        )
        tlfb_data_params["outliers_mode"] = outlier_options_mapped[selected_outlier_mode]
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
        _show_outlier_action_summary(tlfb_data_params, selected_outlier_mode)

        st.write("7. Biochemical Data for Abstinence Verification (Optional)")
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

            st.write("7.1. Specify the cutoff value for biochemically-verified abstinence")
            bio_data_params["cutoff"] = st.number_input(
                "Equal or below the specified value is considered abstinent.",
                value=0.0,
                step=None,
                key="bio_cutoff"
            )
            st.write(f'Data records with a value higher than {bio_data_params["cutoff"]} are considered non-abstinent.')

            st.write("7.2. Override False Negative TLFB Records")
            bio_data_params["overridden_amount"] = st.number_input(
                "Specify the TLFB amount to override a false negative TLFB record. "
                "(self-report TLFB records say abstinent, but biochemical data invalidate them).",
                value=tlfb_data_params["cutoff"] + 1
            )
            st.write(f'False negative TLFB records will be overridden with {bio_data_params["overridden_amount"]}.')

            st.write("7.3. Biochemical Data Interpolation")
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
                    value=0.25,
                    step=0.01
                )
                bio_data_params["days_interpolation"] = right_col.number_input(
                    "The Number of Days of Interpolation",
                    value=1,
                    step=1
                )
                if bio_data_params["half_life"] == 0:
                    st.error("The half life of the biochemical measure should be greater than zero.")

                st.write(f'Additional biochemical records (n={bio_data_params["days_interpolation"]}) will be '
                         f'interpolated based on a half life of {bio_data_params["half_life"]} day(s).')
            else:
                st.write("There will be no interpolations for biochemical data.")

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
        st.write("1. The TLFB data's date column can use either actual dates or arbitrary day counters. "
                 "Please specify the date data type.")
        visit_data_params['use_raw_date'] = st.checkbox(
            "Raw dates are used",
            True,
            key="visit_data_date_col"
        )
        if visit_data_params["use_raw_date"]:
            st.write("The visit dataset uses the actual dates.")
        else:
            st.write("The visit dataset uses arbitrary day counters.")
        st.markdown("***")

        st.write("2. Specify the expected order of the visits (for data normality check)")
        visit_data_params['expected_visits'] = st.multiselect(
            "Please adjust the order accordingly",
            visits,
            default=visits
        )
        if visit_data_params["expected_visits"]:
            st.write(f'The expected order for visits: {visit_data_params["expected_visits"]}')
        else:
            st.write("The expected order for visits will be inferred from the numeric/alphabetic values.")
        st.markdown("***")

        st.write("3. Subjects used in the abstinence calculation.")
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
        st.write(f"Subjects used in the calculation: {visit_data_params['subjects']}")
        st.markdown("***")

        st.write("4. Visit Missing Dates Imputation")
        imputation_summary = dict()
        selected_imputation_mode = st.selectbox(
            "Select your option",
            visit_imputation_options,
            index=1,
            key="visit_imputation_mode"
        )
        imputation_summary['Imputation Mode'] = selected_imputation_mode
        visit_data_params["imputation_mode"] = visit_imputation_options_mapped[selected_imputation_mode]
        if visit_data_params["imputation_mode"] is not None:
            visit_data_params["anchor_visit"] = st.selectbox(
                "Anchor Visit for Imputation",
                visit_data_params['expected_visits'],
                index=0
            )
            imputation_summary['Anchor Visit'] = visit_data_params["anchor_visit"]
        imputation_summary_text = '\n\n* '.join([f'{key}: {value}' for key, value in imputation_summary.items()])
        st.markdown(f"_Summary of the Imputation Parameters_\n\n* {imputation_summary_text}")
        st.markdown("***")

        st.write("5. Visit Duplicate Records Action")
        selected_duplicate_mode = st.selectbox(
            "Select your option",
            duplicate_options,
            index=len(duplicate_options) - 2,
            key="visit_duplicate_mode"
        )
        visit_data_params["duplicate_mode"] = duplicate_options_mapped[selected_duplicate_mode]
        st.write(f"Duplicate Records: {selected_duplicate_mode}")
        st.markdown("***")

        st.write("6. Visit Outliers Action (outliers are those lower than the min or higher than the max)")
        selected_outlier_mode = st.selectbox(
            "Select your option",
            outlier_options,
            key="visit_outliers_mode"
        )
        visit_data_params["outliers_mode"] = outlier_options_mapped[selected_outlier_mode]
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
        _show_outlier_action_summary(visit_data_params, selected_outlier_mode)

    processed_data = st.button("Get/Refresh Visit Data Summary")

    if processed_data or session_state.visit_data is not None:
        _process_visit_data()


def _show_outlier_action_summary(data_params, selected_outlier_mode):
    if selected_outlier_mode == outlier_options[0]:
        st.write("Any potential outliers won't be examined.")
    elif selected_outlier_mode == outlier_options[1]:
        st.write(f'Outliers will be removed if they are smaller than {visit_data_params["allowed_min"]} or '
                 f'greater than {visit_data_params["allowed_max"]}')
    else:
        st.write(f'Outliers will be set to the boundary values if they are smaller than '
                 f'{visit_data_params["allowed_min"]} or greater than {visit_data_params["allowed_max"]}')
    st.markdown("***")


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
        st.write(visit_data.impute_data(visit_data_params["imputation_mode"], visit_data_params["anchor_visit"]))
    else:
        st.write("Imputation Action: None")
    st.write("Visit Attendance Summary")
    st.write(visit_data.get_retention_rates())


def _load_data_summary(data, data_params):
    st.subheader("Data Overview")
    data_all_summary, data_subject_summary, grid = \
        data.profile_data(data_params["allowed_min"], data_params["allowed_max"])
    st.write(data_all_summary)
    st.write(data_subject_summary)
    st.pyplot(grid)

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
    selected_mode = st.selectbox("Abstinence Assumption Mode", calculation_assumptions)
    abst_params_shared["mode"] = calculation_assumptions_mapped[selected_mode]
    st.write(f"Abstinence Assumption: {selected_mode}")
    abst_params_shared["including_end"] = st.checkbox(
        "Including each of the visit dates as the end of the time window examined."
    )
    if abst_params_shared["including_end"]:
        st.write("The end visit's date will be included in the calculation. For example, say EOT is 07/14/2020, if the "
                 "option is checked, the TLFB records up to 07/14/2020 will be included in the calculation.")
    else:
        st.write("The end visit's date won't be included in the calculation. For example, EOT is 07/14/2020, if the "
                 "option is checked, the TLFB up to 07/13/2020 will be included in the calculation.")
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
        if abst_params["visits"]:
            col.write(f"The following visit's pp abstinence will be calculated: {abst_params['visits']}")

        abst_var_name_option = col.selectbox(
            "2. Abstinent Variable Names",
            options=abst_var_name_options,
            key=abst_option + "_name_option"
        )
        if abst_var_name_option != abst_var_name_options[0]:
            abst_names_text = col.text_input(
                "Custom Abstinence Variable Names (enclose each name within single quotes and names should be separated"
                "by commas). Example: 'abst_var1', 'abst_var2', 'abst_var3'",
                key=abst_option + "_name")
            try:
                abst_names_list = eval(f"[{abst_names_text}]")
            except SyntaxError:
                col.error("Please follow the instruction to give names.")
            else:
                abst_params["abst_var_names"] = abst_names_list
                col.write(f"Abstinence variable names: {abst_names_list}")
        else:
            abst_params["abst_var_names"] = "infer"
            col.write("Abstinence variable names will be inferred.")

        if col is pp_col:
            days_text = col.text_input(
                "3. Specify a list of the number of days preceding the visit dates. \n"
                "Enter your options and separate them by commas. Example: 7, 14, 21"
            )
            try:
                days_list = eval(f"[{days_text}]")
            except SyntaxError:
                col.error("Please follow the instruction to give options.")
            else:
                abst_params["days"] = days_list
                col.write(f"Calculate the following days' pp: {days_list}")
        elif col is prol_col:
            abst_params["quit_visit"] = col.selectbox(
                "3. Specify the quit visit",
                visit_data_params['expected_visits']
            )
            col.write(f'Selected Quit Visit: {abst_params["quit_visit"]}')
            lapse_text = col.text_input(
                "4. Specify lapse definitions. Enter your options and separate them "
                "by commas. When lapses are not allowed, its definition is False. For all definitions, "
                "please enclose each of them within single quotes. "
                "Example: 'False', '5 cigs', '5 cigs/14 days'. "
                "See GitHub page for more details."
            )
            try:
                definitions = eval(f"[{lapse_text}]")
            except SyntaxError:
                col.error("Please follow the instruction to give options.")
            else:
                for i, definition in enumerate(definitions):
                    if definition.lower().startswith("fal"):
                        definitions[i] = False
                abst_params["lapse_definitions"] = definitions

                col.write(f"Lapse Definitions (Adjusted for syntax): {definitions}")
            abst_params["grace_period"] = col.slider(
                "5. Specify the grace period in days (default: 14 days)",
                value=14,
                min_value=1,
                max_value=100,
                step=1
            )
            col.write(f"Grace Period (days): {abst_params['grace_period']}")
        else:
            abst_params["start_visit"] = col.selectbox(
                "3. Specify the start visit",
                visit_data_params['expected_visits']
            )
            col.write(f'Start Visit of the Continuous Window: {abst_params["start_visit"]}')

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


def _load_script_element():
    st.subheader("Python Script Generation")
    st.markdown("To automatize data processing, you can generate a Python script based on the input.")
    if st.button("Generate Python Script"):
        calculation_script = _generate_calculation_script()
        st.write("You can download the script here.")
        _pop_download_link(calculation_script, 'calculate_abst', 'Python Script', file_type='py')
        st.write("Alternatively, you can copy the script directly by hitting the copy icon in the right upper corner "
                 "in the box")
        st.code(calculation_script)

def _pop_download_link(df, filename, link_name, kept_index=False, file_type='csv', container=st):
    if isinstance(df, pd.DataFrame):
        data_to_download = df.to_csv(index=kept_index)
    else:
        data_to_download = df
    b64 = base64.b64encode(data_to_download.encode()).decode()
    href = f'<a href="data:file/{file_type};base64,{b64}" download="{filename}.{file_type}">Download {link_name}</a>'
    container.markdown(href, unsafe_allow_html=True)


def _max_width_(width):
    st.markdown(
        f"""<style>.reportview-container .main .block-container{{max-width: {width}px;}}</style>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":

    _max_width_(1200)
    _load_elements()
