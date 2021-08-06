### Assignement Requirements:
Extract Time In Range metrics from the CGM data given.


Metrics to be extracted:
a) Percentage time in hyperglycemia (CGM > 180 mg/dL), 
b) percentage of time in hyperglycemia critical (CGM > 250 mg/dL), 
c) percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL), 
d) percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL), 
e) percentage time in hypoglycemia level 1 (CGM < 70 mg/dL),  
f) percentage time in hypoglycemia level 2 (CGM < 54 mg/dL).

The metrics need to be calculated for 2 cases:
1. Manual Mode
2. Auto mode

Metrics need to be caculated for 3 different intervals during the day. 
1. Daytime(6AM - 12AM)
2. Overnight(12AM - 6AM)
3. WholeDay(12AM - 12AM)

In total 2 Modes * 6 Metrics * 3 Intervals   ->  36 Metrics per day to be extracted.

Working with timeseries data.

