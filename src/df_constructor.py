# Title: df Contructor
# Author: Doug Hart
# Date Created: 2/25/2020
# Last Updated: 3/1/2020

import pandas as pd 
import numpy as np
import multiprocessing as mp
from functions import convert_panel, int_convert, assign_names

#reading in and restructuring my data to fit panel format
df = pd.read_csv('medsales.csv',header=None)
panel_ready = convert_panel(df,4)
dfpr = pd.DataFrame(panel_ready)

df2 = pd.read_csv('scount2.csv', header=None)
newsheet = convert_panel(df2,4)
dfpr2 = pd.DataFrame(newsheet)

df3 = pd.read_csv('med_cut.csv',header=None)
newsheet3 = convert_panel(df3,4)
dfpr4 = pd.DataFrame(newsheet3)

df5 = pd.read_csv('n_days.csv', header=None)
newsheet5 = convert_panel(df5,4)
dfpr5 = pd.DataFrame(newsheet5)

df6 = pd.read_csv('foreclosed.csv',header=None)
newsheet6 = convert_panel(df6,4)
dfpr6= pd.DataFrame(newsheet6)

df7 = pd.read_csv('raw_inv.csv',header=None)
newsheet7 = convert_panel(df7,4)
dfpr7 = pd.DataFrame(newsheet7)

df8 = pd.read_csv('new_listings.csv',header=None)
newsheet8 = convert_panel(df8,4)
dfpr8 = pd.DataFrame(newsheet8)

df9 = pd.read_csv('med_daily.csv',header=None)
newsheet9 = convert_panel(df9,4)
dfpr9 = pd.DataFrame(newsheet9)

#naming columns so that I can conveniently call them when merging
dfpr.rename(columns={0: 'date', 
                   1: 'id', 2: 'city', 3: 'state', 4: 'med_price' }, inplace=True)
dfpr2.rename(columns={0: 'date', 
                   1: 'id', 2: 'city', 3: 'state', 4: 'sale_count' }, inplace=True)
dfpr4.rename(columns={0: 'date', 
                   1: 'id', 2: 'city', 3: 'state', 4: 'med_price_cut' }, inplace=True)
dfpr5.rename(columns={0: 'date', 
                   1: 'id', 2: 'city', 3: 'state', 4: 'med_days_listed' }, inplace=True)
dfpr6.rename(columns={0: 'date', 
                   1: 'id', 2: 'city', 3: 'state', 4: 'percent_foreclosed' }, inplace=True)
dfpr7.rename(columns={0: 'date', 
                   1: 'id', 2: 'city', 3: 'state', 4: 'num_listed' }, inplace=True)
dfpr8.rename(columns={0: 'date', 
                   1: 'id', 2: 'city', 3: 'state', 4: 'new_listing_m' }, inplace=True)
dfpr9.rename(columns={0: 'date', 
                   1: 'id', 2: 'city', 3: 'state', 4: 'med_daily_inv' }, inplace=True)


#creating final use dataframe and merging columns into it
final = pd.merge(dfpr,dfpr2, on= ['date','id','city','state'], how = 'left')

final = pd.merge(final,dfpr4, on= ['date','id','city'], how = 'left')
final.shape

final.rename(columns = {'state_x':'perm_state'}, inplace = True)

final = pd.merge(final,dfpr5, on= ['date','id','city'], how = 'left')
final.shape
final = pd.merge(final,dfpr6, on= ['date','id','city'], how = 'left')
final.shape
final = pd.merge(final,dfpr7, on= ['date','id','city'], how = 'left')
final.shape
final = pd.merge(final,dfpr8, on= ['date','id','city'], how = 'left')
final.shape
final = pd.merge(final,dfpr9, on= ['date','id','city'], how = 'left')
final.shape

final.drop('state_y', inplace=True, axis = 1)
final.drop('state_x', inplace=True, axis = 1)
final.drop('state', inplace =True, axis = 1)

final.rename(columns = {'perm_state':'state'}, inplace = True)

#Turns out this is ALL strings. All strings all day. Next I convert
pool = mp.Pool(mp.cpu_count())
final.med_price_cut = pool.map(int_convert,[row for row in final.med_price_cut])
final.sale_count = pool.map(int_convert,[row for row in final.sale_count])
final.med_price = pool.map(int_convert,[row for row in final.med_price])
final.med_days_listed = pool.map(int_convert,[row for row in final.med_days_listed])
final.percent_foreclosed = pool.map(int_convert,[row for row in final.percent_foreclosed])
final.num_listed = pool.map(int_convert,[row for row in final.num_listed])
final.new_listing_m = pool.map(int_convert,[row for row in final.new_listing_m])
final.med_daily_inv = pool.map(int_convert,[row for row in final.med_daily_inv])
pool.close()

#next to implement multi index, consulting with Nik tells me
#this may be easier by just declaring new df with multi index
#then adding in data columns
final.date = pd.to_datetime(final.date, format= "%Y/%m")

new = pd.DataFrame(index=[final.id, final.date])

#without .values, pandas frustratingly wants to convert everything to nan
new['city']= final.city.values
new['state'] = final.state.values
new['med_price'] = final.med_price.values
new['sale_count']= final.sale_count.values
new['med_price_cut'] = final.med_price_cut.values
new['med_days_listed'] = final.med_days_listed.values
new['percent_foreclosed'] = final.percent_foreclosed.values
new['num_listed'] = final.num_listed.values
new['new_listing_m'] = final.new_listing_m.values
new['med_daily_inv'] = final.med_daily_inv.values

new.index.names = ['city_id', 'date']

#Next I repeat this process with data for various home sizes for deeper analysis
#I don't group this all into a function because of run time, I probably should

condo = pd.read_csv('condos.csv', header=None)
condo.drop(4, axis=1, inplace=True)
panel_ready = convert_panel(condo,4)
condop = pd.DataFrame(panel_ready)
assign_names(condop)
condop.to_pickle('condo.pkl', compression='zip')

obd = pd.read_csv('onebedroom.csv', header=None)
obd.drop(4, axis=1, inplace=True)
next_sheet = convert_panel(obd,4)
obdp = pd.DataFrame(next_sheet)
assign_names(obdp)
obdp.to_pickle('obr.pkl', compression='zip')

tbr = pd.read_csv('twobedroom.csv', header=None)
tbr.drop(4, axis=1, inplace=True)
next_sheet2 = convert_panel(tbr,4)
tbrp = pd.DataFrame(next_sheet2)
assign_names(tbrp)
tbrp.to_pickle('tbr.pkl', compression='zip')

threebr = pd.read_csv('threebedroom.csv', header=None)
threebr.drop(5, axis=1, inplace=True)
threebr.drop(4, axis=1, inplace=True)
next_sheet4 = convert_panel(threebr,4)
threebrp = pd.DataFrame(next_sheet4)
assign_names(threebrp)
threebrp.to_pickle('threebr.pkl', compression='zip')

fourbr = pd.read_csv('fourbedroom.csv', header=None)
fourbr.drop(5, axis=1, inplace=True)
fourbr.drop(4, axis=1, inplace=True)
next_sheet5 = convert_panel(fourbr,4)
fourbrp = pd.DataFrame(next_sheet5)
assign_names(fourbrp)
fourbrp.to_pickle('fourbr.pkl', compression='zip')

fplus = pd.read_csv('five_plus.csv', header=None)
fplus.drop(5, axis=1, inplace=True)
fplus.drop(4, axis=1, inplace=True)
next_sheet3 = convert_panel(fplus,4)
fplusp = pd.DataFrame(next_sheet3)
assign_names(fplusp)
fplusp.to_pickle('fplus.pkl', compression='zip')

#setting time variable and converting from strings to floats
pool = mp.Pool(mp.cpu_count())
dfpr.est_val = pool.map(int_convert,[row for row in dfpr.est_val])
obdp.est_val = pool.map(int_convert,[row for row in obdp.est_val])
tbrp.est_val = pool.map(int_convert,[row for row in tbrp.est_val])
threebrp.est_val = pool.map(int_convert,[row for row in threebrp.est_val])
fplusp.est_val = pool.map(int_convert,[row for row in fplusp.est_val])
fourbrp.est_val = pool.map(int_convert,[row for row in fourbrp.est_val])

fourbrp['date'] = pool.map(make_date,[row for row in fourbrp.date])
dfpr['date'] = pool.map(make_date,[row for row in dfpr.date])
obdp['date'] = pool.map(make_date,[row for row in obdp.date])
tbrp['date'] = pool.map(make_date,[row for row in tbrp.date])
threebrp['date'] = pool.map(make_date,[row for row in threebrp.date])
fplusp['date'] = pool.map(make_date,[row for row in fplusp.date])

pool.close()

dfpr.dropna(axis=0, inplace=True, how='any')
condwa = dfpr[dfpr["state"] == 'WA']
condtrain = dfpr[dfpr["state"] != 'WA']




