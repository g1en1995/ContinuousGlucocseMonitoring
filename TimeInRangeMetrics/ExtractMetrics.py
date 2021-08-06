#importing libraries
import pandas as pd

#Reading insulin data and reversing the order of data
pd.read_csv(r'InsulinData.csv').iloc[::-1].to_csv(r'ReversedInsulinData.csv', index = False)


df1 = pd.read_csv(r'ReversedInsulinData.csv')
del df1['Index']
df1['DateTime'] = df1['Date']+ ' ' + df1['Time']                #introducing datetime column for comparision
df1['DateTime'] = pd.to_datetime(df1['DateTime'])               

#Reading CGM data and reversing the order of data
pd.read_csv(r'CGMData.csv').iloc[::-1].to_csv(r'ReversedCGMData.csv', index = False)

df2 = pd.read_csv(r'ReversedCGMData.csv')
del df2['Index']
df2['DateTime'] = df2['Date']+ ' ' + df2['Time']
df2['DateTime'] = pd.to_datetime(df2['DateTime'])


date_values=[]
keys = df2['Date'].value_counts().keys().to_list()              # Removing the inadequate dates from data frame
values = df2['Date'].value_counts().to_list()
for dates, count in zip(keys,values):
    if (count < 200 or count > 288):
        date_values.append(dates)

for i in date_values:
    indexNames = df2[df2['Date'] == i].index
    df2.drop(indexNames, inplace = True)


x = df1.loc[df1['Alarm'] =='AUTO MODE ACTIVE PLGM OFF']         # Detection of switch from Manual Mode to Auto Mode
Auto_mode_switch_Time = x.iloc[0].DateTime

# collecting values for 6 metrics
# % Time in Hyperglycemia (CGM > 180 mg/dL)
# % Time in Hyperglycemia critical(CGM >250 mg/dL )
# % Time in Range(CGM >= 70 mg/dL and CGM <= 180 mg/dL)
# % Time in Range Secondary(CGM >= 70 mg/dL and CGM <= 150 mg/dL)
# % Time in Hypoglycemia level 1(CGM < 70 )
# % Time in Hypoglycemia level 2(CGM < 54 )


filt1 = (df2['DateTime'] <= Auto_mode_switch_Time)              # Splitting the dataframe into auto and manual mode based on timestamp
filt2 = (df2['DateTime'] > Auto_mode_switch_Time)

Automode = df2[filt2]
Manualmode = df2[filt1]

#Splitting the dataframe based on day time and overnight time
Auto_overnight = Automode.set_index('DateTime').between_time('00:00', '06:00', include_end=False).reset_index()
Auto_daytime = Automode.set_index('DateTime').between_time('06:00', '00:00',include_end= False).reset_index()
Manual_overnight =Manualmode.set_index('DateTime').between_time('00:00', '06:00', include_end=False).reset_index()
Manual_daytime = Manualmode.set_index('DateTime').between_time('06:00', '00:00', include_end=False).reset_index()


#General function for calculating the metric value
def calculate_metric_value(x):
    
    Above180 = []
    Above250 = []
    Between70and180 = []
    Between70and150 = []          
    Below70 = []
    Below54 = []
    
    for date, group in x:
        r = group['Sensor Glucose (mg/dL)'].isna().sum()

        group.interpolate(method = 'linear', inplace = True)
        
        group1 = 0
        group2 = 0
        group3 = 0
        group4 = 0
        group5 = 0
        group6 = 0
        for value in group['Sensor Glucose (mg/dL)']:
            if (value > 180.0):
                group1 +=1
            if(value > 250.0):
                group2 += 1
            if( 70.0 <= value <= 180.0):
                group3 += 1
            if(70.0 <= value <= 150.0):
                group4 += 1
            if(value < 70.0):
                group5 += 1
            if(value < 54.0):
                group6 += 1
        Above180.append(group1/288)
        Above250.append((group2/288))
        Between70and180.append(group3/288)
        Between70and150.append(group4/288)
        Below70.append(group5/288)
        Below54.append(group6/288)
    
    return (sum(Above180)/len(Above180), sum(Above250)/len(Above250), sum(Between70and180)/len(Between70and180),
            sum(Between70and150)/len(Between70and150),sum(Below70)/len(Below70),sum(Below54)/len(Below54))
        

#calculating metrics
a1,b1,c1,d1,e1,f1 = calculate_metric_value(Auto_overnight.groupby('Date', sort = False))
a2,b2,c2,d2,e2,f2 = calculate_metric_value(Auto_daytime.groupby('Date', sort = False))
a3,b3,c3,d3,e3,f3 = calculate_metric_value(Manual_overnight.groupby('Date', sort = False))
a4,b4,c4,d4,e4,f4 = calculate_metric_value(Manual_daytime.groupby('Date', sort = False))
a5,b5,c5,d5,e5,f5 = calculate_metric_value(Automode.groupby('Date', sort = False))
a6,b6,c6,d6,e6,f6 = calculate_metric_value(Manualmode.groupby('Date', sort = False))


#creating a dataframe and saving to results.csv
data = {'overnight_1': [a3,a1], 'overnight_2' : [b3,b1], 'overnight_3' : [c3,c1], 'overnight_4': [d3,d1], 'overnight_5': [e3,e1], 'overnight_6':[f3,f1],
        'daytime_1': [a4,a2], 'daytime_2': [b4,b2],'daytime_3': [c4,c2],'daytime_4': [d4,d2],'daytime_5': [e4,e2],'daytime_6': [f4,f2],
        'day_1': [a6,a5], 'day_2': [b6,b5],'day_3': [c6,c5],'day_4': [d6,d5],'day_5':[e6,e5],'day_6': [f6,f5]}
results = pd.DataFrame(data, index=['Manual', 'Auto'])
results.to_csv('Fernandes_Results.csv')
