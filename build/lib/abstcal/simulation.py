from datetime import timedelta, datetime
import random
import pandas as pd
import numpy as np

# %%
subject_count = 500
visit_count = 10
subject_id_start = 1001
subject_ids = list(range(subject_id_start, subject_id_start + subject_count))
visit_attendance_allowance = 2
visits = list(range(visit_count))
random.seed(1000)

# %%
# Create the visit table without missing data
v0_dates = [datetime(2019, 1, 1) + timedelta(days=random.randint(0, 364)) for _ in range(subject_count)]
visit_data = pd.DataFrame({visits[0]: v0_dates}, index=subject_ids)
visit_data.index.name = 'id'
visit_offsets = [7, 21, 28, 35, 49, 63, 77, 105, 189]

for i, visit_offset in enumerate(visit_offsets, 1):
    visit_offset += random.randint(-visit_attendance_allowance, visit_attendance_allowance)
    visit_data[visits[i]] = visit_data[visits[0]] + timedelta(days=visit_offset)

# Randomly dropping out subjects
retention_rates = [round(1 - 0.05*i, 2) for i in range(visit_count)]
dropout_rates = [0, *(round(retention_rates[i] - retention_rates[i+1], 2) for i in range(visit_count-1))]
subjects_to_drop = dict()
remained_ids = subject_ids
for i in range(visit_count):
    dropped_subject_count = int(subject_count * dropout_rates[i])
    dropped_ids = random.sample(remained_ids, dropped_subject_count)
    remained_ids = set(remained_ids) - set(dropped_ids)
    subjects_to_drop[visits[i]] = dropped_ids

for visit, subjects in subjects_to_drop.items():
    if not subjects:
        continue
    visit_data.iloc[visit_data.index.isin(subjects), visit:visit_count] = np.nan

# Randomly pick 10 subjects without any missing data, with each of them having one visit
# date adding x days to mimic data entry mistakes, x is a random number between 10-50

no_missing_ids = list(visit_data.loc[visit_data.isnull().sum(axis=1) < 1, :].index)
used_ids = random.sample(no_missing_ids, 10)
for subject_id in used_ids:
    visit_data.loc[subject_id, random.randint(0, 9)] += timedelta(days=random.randint(10, 50))

simulated_visit_data = visit_data.reset_index().melt(id_vars='id', var_name='visit', value_name='date').dropna()
simulated_visit_data.to_csv('simulated_visit_data.csv', index=False)