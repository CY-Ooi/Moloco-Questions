import pandas as pd

df = pd.read_excel('Adops & Data Scientist Sample Data.xlsx', sheetname='Q1 Analytics')

# Analytics Question 1
df_BDV = df[df['country_id']=='BDV']
df_BDV = df_BDV[['site_id', 'user_id']].drop_duplicates()
df_BDV['Count'] = 1
count_number = df_BDV.set_index(['site_id', 'user_id']).count(level='site_id')
print(count_number.idxmax().values[0], count_number.max().values[0])

# Analytics Question 2
df2 = df[(df['ts']>='2019-02-03 00:00:00') & (df['ts']<='2019-02-04 23:59:59')]
df2['user_site'] = df2[['user_id', 'site_id']].apply(tuple, axis=1)
df_user_site = df2['user_site']
count_number2 = df_user_site.value_counts()
count_ten_more = count_number2[count_number2 > 10]
print(count_ten_more)

# Analytics Question 3
df3 = df[['user_id', 'site_id', 'ts']]
df3['ts_site'] = df3[['ts', 'site_id']].apply(tuple, axis=1)
df_last = df3.groupby('user_id').max()
df_last = df_last['ts_site'].apply(pd.Series).rename(columns={0: 'last_ts', 1: 'last_site'})
site_last = df_last['last_site']
count_number = site_last.value_counts()
top_three = count_number[0:3]
print(top_three)

# Analytics Question 4
df4 = df[['user_id', 'site_id', 'ts']]
df4['ts_site'] = df4[['ts', 'site_id']].apply(tuple, axis=1)
df_first = df4.groupby('user_id').min()
df_first = df_first['ts_site'].apply(pd.Series).rename(columns={0: 'first_ts', 1: 'first_site'})
df_last = df4.groupby('user_id').max()
df_last = df_last['ts_site'].apply(pd.Series).rename(columns={0: 'last_ts', 1: 'last_site'})
same_site = (df_first['first_site']==df_last['last_site']).sum()
one_time_user = (df_first['first_ts']==df_last['last_ts']).sum()
print(same_site, '(include one time users)')
print(same_site - one_time_user, '(exclude one time users)')
