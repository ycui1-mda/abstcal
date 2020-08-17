# Abstinence Calculator

A Python package to calculate abstinence results using the timeline followback interview data

## Installation
Install the package using the pip tool. If you need instruction on how to install Python, you 
can find information at the [Python](https://www.python.org/) website. To install the pip tool,
which is the most common Python package management tool, you can find information at the 
[pypa](https://pip.pypa.io/en/stable/installing/) website.

```
pip install abstcal
```

## Overview of the Package
This package is developed to score abstinence using the Timeline Followback (TLFB) and 
visit data in clinical substance use research. It provides functionalities to preprocess 
the datasets to remove duplicates and outliers. In addition, it can impute missing data 
using various criteria. It supports the calculation of abstinence of varied definitions, 
including continuous, point-prevalence, and prolonged. It can optionally integrate biochemical
verification data.

## Required Datasets
**The Timeline Followback Data (Required)** The dataset should have three columns: __*id*__, 
__*date*__, and __*amount*__. The id column stores the subject ids, each of which should 
uniquely identify a study subject. The date column stores the dates when daily substance 
uses are collected. The amount column stores substance uses for each day.

id | date | amount 
------------ | ------------- | -------------
1000 | 02/03/2019 | 10
1000 | 02/04/2019 | 8
1000 | 02/05/2019 | 12
1000 | 02/06/2019 | 9
1000 | 02/07/2019 | 10
1000 | 02/08/2019 | 8

**The Biochemical Measures Dataset (Optional)** The dataset should have three columns: __*id*__, 
__*date*__, and __*amount*__. The id column stores the subject ids, each of which should 
uniquely identify a study subject. The date column stores the dates when daily substance 
uses are collected. The amount column stores the biochemical measures that verify substance use status.

id | date | amount 
------------ | ------------- | -------------
1000 | 02/03/2019 | 4
1000 | 02/11/2019 | 6
1000 | 03/04/2019 | 10
1000 | 03/22/2019 | 8
1000 | 03/28/2019 | 6
1000 | 04/15/2019 | 5

**The Visit Data (Required)** It needs to be in one of the following two formats.
1. **The long format.** The dataset should have three columns: __*id*__, __*visit*__, 
and __*date*__. The id column stores the subject ids, each of which should uniquely 
identify a study subject. The visit column stores the visits. The date column stores 
the dates for the visits.

id | visit | date 
------------ | ------------- | -------------
1000 | 0 | 02/03/2019
1000 | 1 | 02/10/2019
1000 | 2 | 02/17/2019
1000 | 3 | 03/09/2019 
1000 | 4 | 04/07/2019 
1000 | 5 | 05/06/2019

2. **The wide format.** The dataset should have the id column and additional columns 
with each representing a visit.

id | v0 | v1 | v2 | v3 | v4 | v5
----- | ----- | ----- | ----- | ----- | ----- | ----- |
1000 | 02/03/2019 | 02/10/2019 | 02/17/2019 | 03/09/2019 | 04/07/2019 | 05/06/2019
1001 | 02/05/2019 | 02/13/2019 | 02/20/2019 | 03/11/2019 | 04/06/2019 | 05/09/2019

## Supported Abstinence Definitions
The following abstinence definitions have both calculated as the intent-to-treat (ITT) 
or responders-only options. By default, the ITT option is used.
1. **Continuous Abstinence**: No substance use in the defined time window. Usually, it 
starts with the target quit date.
2. **Point-Prevalence Abstinence**: No substance use in the defined time window preceding 
the assessment time point.
3. **Prolonged Abstinence Without Allowing Lapses**: No substance use after the grace 
period (usually 2 weeks) til the assessment time point.
4. **Prolonged Abstinence With Lapses Allowed**: Lapses are allowed after the grace period 
til the assessment time point. If lapses exceed the defined threshold, relapses will be 
identified.

## Use Example

### 1. Import the Package
```
from abstcal import TLFBData, VisitData, AbstinenceCalculator
```

### 2. Process the TLFB Data
#### 2a. Read the TLFB data
You can either specify the full path of the TLFB data or just the filename if the dataset 
is in your current work directory. Supported file formats include comma-separated (.csv), 
tab-delimited (.txt), and Excel spreadsheets (.xls, .xlsx).
```
tlfb_data = TLFBData('path_to_tlfb.csv')
```

#### 2b. Profile the TLFB data
In this step, you will see a report of the data summary, such as the number of records, 
the number of subjects, and any applicable abnormal data records, including duplicates 
and outliers. In terms of outliers, you can specify the minimal and maximal values for 
the substance use amounts. Those values outside of the range are considered outliers 
and are shown in the summary report.
```
# No outlier identification
tlfb_data.profile_data()

# Identify outliers that are outside of the range
tlfb_data.profile_data(0, 100)
```
#### 2c. Drop data records with any missing values
Those records with missing *id*, *date*, or *amount* will be removed. The number of removed
records will be reported.
```
tlfb_data.drop_na_records()
```

#### 2d. Check and remove any duplicate records
Duplicate records are identified based on __*id*__ and __*date*__. There are different 
ways to remove duplicates: *min*, *max*, or *mean*, which keep the minimal, maximal, 
or mean of the duplicate records. You can also have the options to remove all duplicates. 
You can also simply view the duplicates and handle these duplicates manually.
```
# Check only, no actions for removing duplicates
tlfb_data.check_duplicates(None)

# Check and remove duplicates by keeping the minimal
tlfb_data.check_duplicates("min")

# Check and remove duplicates by keeping the maximal
tlfb_data.check_duplicates("max")

# Check and remove duplicates by keeping the computed mean (all originals will be removed)
tlfb_data.check_duplicates("mean")

# Check and remove all duplicates
tlfb_data.check_duplicates(False)
```

#### 2e. Recode outliers (optional)
Those values outside the specified range are considered outliers. The values lower than 
the minimal will be recoded as the minimal, while the values higher than the maximal will
be recoded as the maximal. The number of recoded outliers will be reported.
```
# Set the minimal and maximal
tlfb_data.recode_data(0, 100)
```

#### 2f. Impute the missing TLFB data
To calculate the ITT abstinence, the TLFB data will be imputed for the missing records.
All contiguous missing intervals will be identified. Each of the intervals will be imputed
based on the two values, one before the interval and the other after the interval. You can
choose to impute the missing values for the interval using the mean of these two values or
interpolate the missing values for the interval using the linear values generated from the
two values. Alternatively, you can specify a fixed value, which will be used to impute all
missing values.
```
# Use the mean
tlfb_data.impute_data("uniform")

# Use the linear interpolation
tlfb_data.impute_data("linear")

# Use a fixed value, whichever is appropriate to your research question
tlfb_data.impute_data(1)
tlfb_data.impute_data(5)
```

### 3. Process the Visit Data
#### 3a. Read the visit data
Similar to reading the TLFB data, you can read files in .csv, .txt, .xls, or .xlsx format. 
```
# Read the visit data in the long format (the default option)
visit_data = VisitData("file_path.csv")

# Read the visit data in the wide format
visit_data = VisitData("file_path.csv", "wide")
```

#### 3b. Profile the visit data
You will see a report of the data summary, such as the number of records, the number of 
subjects, and any applicable abnormal data records, including duplicates and outliers. 
In terms of outliers, you can specify the minimal and maximal values for the dates. The
dates will be inferred from strings. Please use the format *mm/dd/yyyy*.
Importantly, it will also detect if any subjects have their visits with the dates that
are out of the order. By default, the order is inferred using the numeric or alphabetic 
order of the visits. These records with possibly incorrect data may result in wrong
abstinence calculations.
```
# No outlier identification
visit_data.profile_data()

# Outlier identification
visit_data.profile_data("07/01/2000", "12/08/2020")

# Specify the expected order of the visits
visit_data.profile_data(expected_visit_order=[1, 2, 3, 5, 4])
```

#### 3c. Drop data records with any missing values 
Those records with missing *id*, *visit*, or *date* will be removed. The number of removed
records will be reported.
```
visit_data.drop_na_records()
```

#### 3d. Check and remove any duplicate records
Duplicate records are identified based on __*id*__ and __*visit*__. There are different 
ways to remove duplicates: *min*, *max*, or *mean*, which keep the minimal, maximal, 
or mean of the duplicate records. The options are the same as how you deal with duplicates
in the TLFB data.
```
# Check only, no actions for removing duplicates
visit_data.check_duplicates(None)

# Check and remove duplicates by keeping the minimal
visit_data.check_duplicates("min")

# Check and remove duplicates by keeping the maximal
visit_data.check_duplicates("max")

# Check and remove duplicates by keeping the computed mean (all originals will be removed)
visit_data.check_duplicates("mean")

# Check and remove all duplicates
visit_data.check_duplicates(False)
```

#### 3e. Recode outliers (optional)
Those values outside the specified range are considered outliers. The values lower than 
the minimal will be recoded as the minimal, while the values higher than the maximal will
be recoded as the maximal. The number of recoded outliers will be reported.
```
# Set the minimal and maximal
visit_data.recode_data("07/01/2000", "12/08/2020")
```

#### 3f. Impute the missing visit data
To calculate the ITT abstinence, the visit data will be imputed for the missing records.
The program will first find the earliest visit date as the anchor visit, which is 
presumed to be non-missing for all subjects. Then it will calculate the difference in 
days between the later visits and the anchor visit. Based on these difference values, the
following two imputation options are available. The *"freq"* option will use the most
frequent difference value, which is the default option. The *"mean"* option will use the
mean difference value.

```
# Use the most frequent difference value between the missing visit and the anchor visit
visit_data.impute_data("freq")

# Use the mean difference value between the missing visit and the anchor visit
visit_data.impute_data("mean")
```

### 4. Calculate Abstinence
#### 4a. Create the abstinence calculator using the TLFB and visit data
You can specify the cutoff for the abstinence. By default, only subjects who have both the
TLFB and visit data will be used for calculating abstinence.
```
abst_cal = AbstinenceCalculator(tlfb_data, visit_data)
```

#### 4b. Check data availability (optional)
You can find out how many subjects have the TLFB data and how many have the visit data.
```
abst_cal.check_data_availability()
```

#### 4c. Calculate abstinence
For all the function calls to calculate abstinence, you can request the calculation to be
ITT (intent-to-treat) or RO (responders-only). You can optionally specify the calculated
abstinence variable names. By default, the abstinence names will be inferred. Another shared
argument is whether you want to include the ending date. Notably, each method will generate
the abstinence dataset and a dataset recording first lapses that make a subject non-abstinent
for a particular abstinence calculation.

##### Continuous abstinence
To calculate the continuous abstinence, you need to specify the visit when the window starts
and the visit when the window ends. To provide greater flexibility, you can specify a series
of visits to generate multiple time windows.
```
# Calculate only one window
abst_df, lapse_df = abst_cal.abstinence_cont(2, 5)

# Calculate two windows
abst_df, lapse_df = abst_cal.abstinence_cont(2, [5, 6])

# Calculate three windows with abstinence names specified
abst_df, lapse_df = abst_cal.abstinence_cont(2, [5, 6, 7], ["abst_var1", "abst_var2", "abst_var3"])
```

##### Point-prevalence abstinence
To calculate the point-prevalence abstinence, you need to specify the visits. You'll need to
specify the number of days preceding the time points. To provide greater flexibility, you
can specify multiple visits and multiple numbers of days.
```
# Calculate only one time point, 7-d point-prevalence
abst_df, lapse_df = abst_cal.abstinence_pp(5, 7)

# Calculate multiple time points, multiple day conditions
abst_df, lapse_df = abst_cal.abstinence_pp([5, 6], [7, 14, 21, 28])
```

##### Prolonged abstinence
To calculate the prolonged abstinence, you need to specify the quit visit and the number of
days for the grace period (the default length is 14 days). You can calculate abstinence for
multiple time points. There are several options regarding how a lapse is defined. See below
for some examples.
```
# Lapse isn't allowed
abst_df, lapse_df = abst_cal.abstinence_prolonged(3, [5, 6], False)

# Lapse is defined as exceeding a defined amount of substance use
abst_df, lapse_df = abst_cal.abstinence_prolonged(3, [5, 6], '5 cigs')

# Lapse is defined as exceeding a defined number of substance use days
abst_df, lapse_df = abst_cal.abstinence_prolonged(3, [5, 6], '3 days')

# Lapse is defined as exceeding a defined amount of substance use over a time window
abst_df, lapse_df = abst_cal.abstinence_prolonged(3, [5, 6], '5 cigs/7 days')

# Lapse is defined as exceeding a defined number of substance use days over a time window
abst_df, lapse_df = abst_cal.abstinence_prolonged(3, [5, 6], '3 days/7 days')

# Combination of these criteria
abst_df, lapse_df = abst_cal.abstinence_prolonged(3, [5, 6], ('5 cigs', '3 days/7 days'))
```

### 5. Output Datasets
#### 5a. The abstinence datasets
To output the abstinence datasets that you have created from calling the abstinence calculation
methods, you can use the following method to create a combined dataset, something like below.

id | itt_abst_cont_v5_v2 | itt_abst_cont_v6_v2 | itt_abst_pp7_v5 | itt_abst_pp7_v6
------------ | ------------- | ------------- | ------------ | -------------
1000 | 1 | 1 | 1 | 1
1001 | 1 | 0 | 1 | 0
1002 | 1 | 1 | 1 | 1
1003 | 0 | 0 | 1 | 1
1004 | 0 | 0 | 1 | 0 
1005 | 0 | 0 | 0 | 1
```
abst_cal.merge_abst_data_to_file([abst_df0, abst_df1, abst_df2], "merged_abstinence_data.csv")
```
#### 5b. The lapse datasets
To output the lapse datasets that you have created from calling the abstinence calculation
methods, you can use the following method to create a combined dataset, something like below.

id | date | amount | abst_name
------------ | ------------- | ------------- | -------------
1000 | 02/03/2019 | 10 | itt_abst_cont_v5
1001 | 03/05/2019 | 8 | itt_abst_cont_v5
1002 | 04/06/2019 | 12 | itt_abst_cont_v5
1000 | 02/06/2019 | 9 | itt_abst_cont_v6
1001 | 04/07/2019 | 10 | itt_abst_cont_v6
1002 | 05/08/2019 | 8 | itt_abst_cont_v6
```
abst_cal.merge_lapse_data_to_file([lapse_df0, lapse_df1, lapse_df2], "merged_lapse_data.csv")
```

## Additional Features
### I. Integration of Biochemical Verification Data
If your study has collected biochemical verification data, such as carbon monoxide for smoking or breath alcohol 
concentration for alcohol intervention, these biochemical data can be integrated into the TLFB data. In this way,
non-honest reporting can be identified (e.g., self-reported of no use, but biochemically un-verified), the 
self-reported value will be overridden, and the updated record will be used in later abstinence calculation.

The following code shows you a possible work flow. Please note that the biochemical measures dataset should have the 
same data structure as you TLFB dataset. In other words, it should have three columns: __*id*__, __*date*__, and 
__*amount*__.

#### Ia. Prepare the Biochemical Dataset
A key operation to prepare the biochemical dataset is to interpolate extra meaningful records based on the exiting 
records using the `interpolate_biochemical_data` function, as shown below.
```
# First read the biochemical verification data
biochemical_data = TLFBData("beam_co.csv", included_subjects=included_subjects, abst_cutoff=4)
biochemical_data.profile_data()

# Interpolate biochemical records based on the half-life
biochemical_data.interpolate_biochemical_data(0.5, 1)

# Other data cleaning steps
biochemical_data.drop_na_records()
biochemical_data.check_duplicates()
```

#### Ib. Integrate the Biochemical Dataset with the TLFB data
The following code shows you how the integration can be performed. Everything else stays the same, except that in the
`impute_data` method, you need to **specify the `biochemical_data` argument**.
```
tlfb_data = TLFBData("beam_tlfb.csv", included_subjects=included_subjects)
tlfb_sample_summary, tlfb_subject_summary = tlfb_data.profile_data()
tlfb_data.drop_na_records()
tlfb_data.check_duplicates()
tlfb_data.recode_data()
tlfb_data.impute_data(biochemical_data=biochemical_data)
```

### II. Calculate Retention Rates
You can also calculate the retention rate with the visit data with a simple function call, as shown below. 
If a filepath is specified, it will write to a file.
```
# Just show the retention rates results
visit_data.get_retention_rates()

# Write the retention rates to an external file
visit_data.get_retention_rates('retention_rates.csv')
```

### III. Calculate Abstinence Rates
You can calculate the computed abstinence by providing the list of pandas DataFrame objects.
```
# Calculate abstinence by various definitions
abst_pp, lapses_pp = abst_cal.abstinence_pp([9, 10], 7, including_end=True)
abst_pros, lapses_pros = abst_cal.abstinence_prolonged(4, [9, 10], '5 cigs')
abst_prol, lapses_prol = abst_cal.abstinence_prolonged(4, [9, 10], False)

# Calculate abstinence rates for each
abst_cal.calculate_abstinence_rates([abst_pp, abst_pros, abst_prol])
abst_cal.calculate_abstinence_rates([abst_pp, abst_pros, abst_prol], 'abstinence_results.csv')
```
It will create the following DataFrame as the output. If a filepath is specified, it will write to a file.

Abstinence Name | Abstinence Rate
------------ | -------------
itt_pp7_v9                  | 0.159091
itt_pp7_v10                 | 0.170455
itt_prolonged_5_cigs_v9     | 0.159091
itt_prolonged_5_cigs_v10    | 0.113636
itt_prolonged_False_v9      | 0.102273
itt_prolonged_False_v10     | 0.068182

## Questions or Comments
If you have any questions about this package, please feel free to leave comments here or
send me an email to ycui1@mdanderson.org.

## License
MIT License