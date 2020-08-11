# Abstinence Calculator

A Python package to calculate abstinence results using the timeline follow back interview data

## Installation
Install the package using the pip tool 

```
pip install abstcal
```

## Overview of the Package
This package is developed to score abstinence using the Timeline Follow Back (TLFB) and 
visit data in clinical substance use research. It provides functionalities to preprocess 
the datasets to remove duplicates and outliers. In addition, it can impute missing data 
using various criteria. It supports the calculation of abstinence of varied definitions, 
including continuous, point-prevalence, and prolonged.

## Required Datasets
**The Timeline Follow Back Data.** The dataset should have three columns: __*id*__, 
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

**The Visit Data.** It needs to be in one of the following two formats.
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
##### 2a. Read the TLFB data
You can either specify the full path of the TLFB data or just the filename if the dataset 
is in your current work directory. Supported file formats include comma-separated (.csv), 
tab-delimited (.txt), and Excel spreadsheets (.xls, .xlsx).
```
tlfb_data = TLFBData('path_to_tlfb.csv')
```

##### 2b. Profile the TLFB data
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
##### 2c. Drop data records with any missing values
Those records with missing *id*, *date*, or *amount* will be removed. The number of removed
records will be reported.
```
tlfb_data.drop_na_records()
```

##### 2d. Check and remove any duplicate records
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

##### 2e. Recode outliers (optional)
Those values outside the specified range are considered outliers. The values lower than 
the minimal will be recoded as the minimal, while the values higher than the maximal will
be recoded as the maximal. The number of recoded outliers will be reported.
```
# Set the minimal and maximal
tlfb_data.recode_data(0, 100)
```

##### 2f. Impute the missing TLFB data
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
##### 3a. Read the visit data
Similar to reading the TLFB data, you can read files in .csv, .txt, .xls, or .xlsx format. 
```
# Read the visit data in the long format (the default option)
visit_data = VisitData("file_path.csv")

# Read the visit data in the wide format
visit_data = VisitData("file_path.csv", "wide")
```

##### 3b. Profile the visit data
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

##### 3c. Drop data records with any missing values 
Those records with missing *id*, *visit*, or *date* will be removed. The number of removed
records will be reported.
```
visit_data.drop_na_records()
```

##### 3d. Check and remove any duplicate records
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

##### 3e. Recode outliers (optional)
Those values outside the specified range are considered outliers. The values lower than 
the minimal will be recoded as the minimal, while the values higher than the maximal will
be recoded as the maximal. The number of recoded outliers will be reported.
```
# Set the minimal and maximal
visit_data.recode_data("07/01/2000", "12/08/2020")
```

##### 2f. Impute the missing visit data
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
```
abst_cal = AbstinenceCalculator(tlfb_data, visit_data)
abst_cal.check_data_availability()
abst_cal.abstinence_cont(2, [5, 6])
abst_cal.abstinence_pp([5, 6], [7, 14, 21, 28])
abst_cal.abstinence_prolonged(3, [5, 6], '5 cigs')
```