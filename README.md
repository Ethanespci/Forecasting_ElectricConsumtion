# Technical report

The project aims to predict the future electricity consumption of various clients based on their previous consumption. The dataset can be found at this link ‘ElectricityLoadDiagrams20112014 - UCI Machine Learning Repository’ and is composed of the energy consumption of 370 clients in Portugal from January 2011 to January 2015. We will use machine learning models, specifically time series ones, to forecast the evolution over time.
Each of the 3 files one can see in this repository implement a different model. The goal is then to compare the results of a Long Short-Term Memory (LSTM) model with Temporal Fusion Transformer (TFT) model and the Facebook Prophet model. 
The 3 models does not require the same format for the inputs however some common preprocessing transformation have been applied to all.

## Preprocessing and data cleaning

The file provided by the UCI Machine Learning repository is a .txt file that is first converted into a .csv file and then loaded as a Panda data frame on Python. 
The first column is the time of the energy consumption measure with the following format 'yyyy-mm-dd hh:mm:ss'. Each data is spaced 15min apart, so one day comprises 96 records and a week comprises 672 records. 
There are 370 other columns, each one being the consumption of a different client and each of the values is a string. The values represent the client's energy consumption in kW during a 15min interval. 
The data is already clean (without any missing values). However most of the  time series begin with long period of zeros that could badly influence th emodels during training. 
The first preprocessing is removing all these zeros. Further to improve the prediction efficiency the data is cumulated so that the average between 4 (or 96) data points is taken to create one new datapoint which correspond to the electric consumption for 1 hour (for 1 day).

Moreover, some functions to clean and analyze the data are gathered in the section Base Functions. 
Each of them has a little description explaining the inputs and outputs of the functions. Here is a short description of 4 of them since they are the functions used for every model as they help clean the data:

- open_data : It opens the .csv file and transforms the data as Panda data frame with the time as an index.
It is possible to choose which clients one want to extract or decide to extract everything.
It is inside this function that the path to the .csv file is defined, so modify the path to your liking.

- remove_zeros : It will remove all the zeros that are present at the beginning of the time series.
If the data frame is composed of multiple columns, it will change these zeros to NaN, so you might have to use .dropna() to remove the unneeded rows.

- sum_to_increase_timespace : It sums the data together to reduce the number of points.
This transformation is useful because instead of using a moving average to filter the high-frequency variation, the time series has now 1 point for every day for example.
It is possible to change the time interval for which we want to add all the data. Possible intervals are the summation over 4 points to get points every hour or the summation over 4*24 points to get one point daily. 

- keep_important : It removes from the data set all the clients with less than 3 years of data, this rough cut will prevent the models from learning on short datasets.
We will see later that other clients must be removed as their data is too unpredictable. This function also returns the number of clients and the list of the clients' names that have been removed.

## Long Short-Term Memory Model

The data is trained on hourly data and not daily data like the following model. This type of model is efficient for sequential temporal data. 
To predict the following data points the model needs to remember the previous data sometimes looking far away from the past, it is a deep learning model comparable to RNN but solving the problem of vanishing gradient.
Before applying the LSTM module, the data is normalized between [-1;1] to improve performance when calculating the gradient. 
Regarding the meaning of the data, the hyperparameter of the LSTM are chosen equal to 100.
This means that after learning the weight of the parameters with the whole training dataset, the module tries to predict the next data point with the last 100 data. 
100 data correspond roughly to the electric consumption over a day. So basically the LSTM module predicts the next data point by focusing on the data points from the previous day.

## Temporal Fusion Transformer model

The Temporal Fusion Transformer is an improved version of the previous LSTM model. 
Indeed, the architecture of TFT is composed of LSTM modules and decoder/encoder modules that allow predicting the daily electric consumption with other features like the month or the day. 
The well-thought architecture of the model  was designed to combine high-performance multi-horizon forecasting and interpretable insights. [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting Bryan Lima, Sercan O. Ar, Nicolas Loeff, Tomas Pfister - Sept2020]
The TFT model for predicting one client uses the DARTS library and manipulates the data as a Time Series. 
So first, before transforming the data to the DARTS Time series type, a column time is added with the date and hour in the Panda data frame.
Then, after transforming the dataset to the good type of data,the Scaler() function is used from the library to normalize the data between 0 and 1, the function inverse_transform() will allow us to go back to the original range of values once the model has predicted the data.
Next, after separating the data in a training and a validation set, a covariate data frame is created that will store the additional features which are discrete values for the year, the month and the day. 
It is possible to add other features using .stack() to the data frame covariates and add in the parenthesis the feature that needs to be a DARTS Time Series. 
The following code block defines the model hyper-parameter and the quantile for the prediction’s confidence intervals.


## Facebook Prophet model

Facebook Prophet is a time series forecasting tool that employs an additive model, combining various components, such as trends, seasonality, holidays, and special events, to make accurate predictions. 
It decomposes time series data into these components and uses them to model and forecast future values. 
It can handle missing data and outliers and is particularly effective for data with strong seasonality and multiple seasonal components.
This corresponds to electrical data, as one can easily observe yearly and seasonal patterns, and therefore, it is expected to perform well for this prediction problem.
A Facebook prophet model fits a multi-line approximation to the real curve and has several parameters: 
- “seasonality mode” to describe how to take into account the effect of seasonality
- “growth”, to model the trend of the model
- “changepoint prior scale” sets up how easy it is to add a breaking point in the approximation
- “holidays” contains data indicating for each day whether a special event occurred or not. It can help to take into account unexpected increases/decreases

The major drawback of a Prophet model is that it learns directly a relationship between the date and the value of the time series: it doesn’t predict new values based on the previous ones, it just associates a consistent value with each date.
It means that the model can only be trained on one time serie (= 1 customer), so one model per customer is needed to be accurate (otherwise the model would predict the same consumption for every customer).

However, training a Facebook Prophet with the appropriate parameters is quite fast, and therefore, fitting all the models doesn’t take too much time.
To choose the right parameters, a grid-search approach is used, meaning that each combination of parameters is tried and only the best one is kept. 
To do so, a random subset of all time series is selected (to avoid a long processing time), and grid search is made on this subset. 
The set of parameters that gives the best overall MAPE over 1 month of prediction is:
- multiplicative seasonality
- linear growth
- 0.05 changepoint before the scale
- the holiday data is used as it does improve the accuracy of the model




