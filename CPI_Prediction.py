#!/usr/bin/env python
# coding: utf-8

# In[165]:


import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


np.random.seed(1412)


# In[151]:


price_quantity = np.random.randn(4100,2)


# In[152]:


base_price_quantity = price_quantity[0]


# In[153]:


initialDF = pd.DataFrame(price_quantity,columns=["price","quantity"])
initialDF.head()


# In[154]:


initialDF['cpi'] = pd.Series(initialDF["price"] * initialDF["quantity"]/base_price_quantity[0] * base_price_quantity[1])
initialDF.head()


# In[30]:


cpi_index = pd.read_csv("data/CPIndex_Jan13-To-Jul19.csv")
cpi_index.head()


# In[29]:


list(cpi_index)


# In[37]:


fuel_cpi_index = cpi_index[cpi_index.Group==5.0]
fuel_cpi_index.head()


# In[57]:


combined_fuel_cpi = fuel_cpi_index[['Year', 'Month','Combined']]
months = {"January" : 1, "February" : 2, "March" : 3, "April" : 4, "May": 5, "June" : 6, "July" : 7, "August" : 8, "September" : 9, "October" : 10, "November" : 11, "December" : 12}
combined_fuel_cpi['Month'] = combined_fuel_cpi['Month'].map(lambda x: months[x])
combined_fuel_cpi['Timestamp'] = pd.to_datetime({'year' : combined_fuel_cpi['Year'], 'month' : combined_fuel_cpi['Month'], 'day':[1] * combined_fuel_cpi.shape[0]})
combined_fuel_cpi.head()


# In[58]:


combined_fuel_cpi = combined_fuel_cpi.set_index("Timestamp")
combined_fuel_cpi.head()


# In[59]:


combined_fuel_cpi = combined_fuel_cpi.drop(["Year", "Month"], axis = 1)
combined_fuel_cpi.head()


# In[77]:


plot_acf(combined_fuel_cpi.values)


# In[78]:


plot_pacf(combined_fuel_cpi.values)


# In[95]:


total_dataset_size = len(combined_fuel_cpi.values)


# In[170]:


arima_model = ARIMA(combined_fuel_cpi.values[:int(0.7 * total_dataset_size)], order = (1, 1, 1))


# In[171]:


arima_results = arima_model.fit(disp=0)
arima_results.save("models/arima_model.pkl")


# In[172]:


arima_results.summary()


# In[99]:


def normalize(x):
    return (x - x.min())/(x.max() - x.min())


# In[100]:


normalized_combined_fuel_cpi = normalize(combined_fuel_cpi)
normalized_combined_fuel_cpi.head()


# In[173]:


normalized_arima_model = ARIMA(normalized_combined_fuel_cpi.values[:int(0.7 * total_dataset_size)], order = (1, 1, 1))
normalized_arima_model_results = normalized_arima_model.fit()
normalized_arima_model_results.save("models/normalized_arima_model.pkl")


# In[174]:


normalized_arima_model_results.summary()


# In[107]:


print(combined_fuel_cpi.values[int(0.7 * total_dataset_size):])
arima_results.forecast(total_dataset_size - int(0.7 * total_dataset_size))[0]


# In[112]:


print(normalized_combined_fuel_cpi.values[int(0.7 * total_dataset_size):])
normalized_arima_model_results.forecast(total_dataset_size - int(0.7 * total_dataset_size))[0]


# In[113]:


initialDF.head()


# In[131]:


pd.date_range(end='2019-09-01',periods = 4100, freq = 'MS')


# In[155]:


initalDF_total_size = initialDF.shape[0]


# In[156]:


initialDF = initialDF.set_index(pd.date_range(end='2019-09-01', periods = initalDF_total_size, freq = 'MS'))


# In[157]:


initialDF = initialDF.drop(['price','quantity'], axis = 1)
initialDF.head()


# In[175]:


initialDF_arima_model = ARIMA(initialDF.values[:int(0.7 * initalDF_total_size)], order = (1, 1, 1))
initialDF_arima_model_results = initialDF_arima_model.fit(disp = 0)
initialDF_arima_model_results.save("data/initialDF_arima_model.pkl")


# In[176]:


initialDF_arima_model_results.summary()


# In[164]:


print(initialDF.values[int(0.7 * initalDF_total_size): int(0.7 * initalDF_total_size) + 10])
print(initialDF_arima_model_results.forecast(10)[0])


# In[ ]:




