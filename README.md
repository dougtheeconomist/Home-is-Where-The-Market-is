# Homes_For_Sale
Building a predictive model of the housing market

## Introduction

In the U.S., over 14,000 homes are bought and sold every day! With the baby boomer generation reaching retirement age, and coming to a point in their lives when moving out of their family rearing homes makes sense this number may be increasing. 

With an asset as expensive as a house, making either a purchase or a sale is no small financial feat. The more informed a party is before entering the market, the better off they position themselves to make the best deal that they can. While there is some information on the market via sites such as Zillow about future predictions, these can be limited to simply what the average price will do in a year. There is more information available for the rental market, but I find a distict lack of informational resources for the buyers market. In order to provide interested parties with useful information I build a forecast of key indicators for housing market in a given city utilizing panel data. My forecast will only consist of a short term outlook, nothing past a year, however, for a given city my model will provide monthly information within that year. This can be a useful for anyone looking to buy or sell sometime within the coming year with no exact deadline; having an idea of the best time to sell or buy can make a significant difference for a budget. The size of the price of a home is so high that even a one percent change may be well worth waiting a couple of months for. 

## Exploratory Data Analysis

In order to address this issue I obtain real estate market data from Zillow, as well as economic data published by various government agencies. Once I collect and compile the data into a format that is compatible with analysis of panel data the first step is to take a look at the data and see what I have. 

I first look at the variable that I am most interested in predicting; median sale price. I get an idea of what is going on as a whole, and then take a look at the spread of  the mean of all markets medians through time, and finally look at the spread of the means grouped by year across the different markets. 

![total_spread_hist](images/tot_medprice.png)

![mean_across_cities_hist](images/mean_medprice_gbtime.png)

![mean_over_time_hist](images/mean_medprice_gbcity.png)

Lastly I look at a plot of this variable over time for a single market, I choose to look at Seattle as this is where I am currently located, as well as several other random locations. These can be seen below to get an idea of how the price within a given city has been fluctuating since 2008 or so. 

![snapshot_of_different_markets](images/sample_cities_medprice.png)

