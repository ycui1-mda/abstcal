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
##### a. Read the TLFB data
You can either specify the full path of the TLFB data or just the filename if the dataset 
is in your current work directory. Supported file formats include comma-separated (.csv), 
tab-delimited (.txt), and Excel spreadsheets (.xls, .xlsx).
```
tlfb_data = TLFBData('path_to_tlfb.csv')
```

##### b. Profile the TLFB data
In this step, you will see a report of the data summary, such as the number of records, 
the number of subjects, and any applicable abnormal data records, including duplicates 
and outliers. In terms of outliers, you can specify the minimal and maximal values for 
the substance use amounts. Those values outside of the range are considered outliers 
and are shown in the summary report.
```
tlfb_data.profile_data()
```
##### c. Drop data records with any missing values
```
tlfb_data.drop_na_records()
```

##### d. Check and remove any duplicate records
Duplicate records are identified based on __*id*__ and __*date*__. There are different 
ways to remove duplicates: *min*, *max*, or *mean*, which keep the minimal, maximal, 
or mean of the duplicate records. You can also have the options to remove all duplicates. 
You can also simply view the duplicates and handle these duplicates manually.
```
tlfb_data.check_duplicates()
```

##### e. Recode outliers
```
tlfb_data.recode_data()
```

##### f. Impute the missing TLFB data
```
tlfb_data.impute_data()
```

```
visit_data = VisitData("../smartmod_visit_data.csv")
visit_data.profile_data()
visit_data.drop_na_records()
visit_data.check_duplicates()
visit_data.recode_data()
visit_data.impute_data()

abst_cal = AbstinenceCalculator(tlfb_data, visit_data)
abst_cal.check_data_availability()
abst_cal.abstinence_cont(2, [5, 6])
abst_cal.abstinence_pp([5, 6], [7, 14, 21, 28])
abst_cal.abstinence_prolonged(3, [5, 6], '5 cigs')
```