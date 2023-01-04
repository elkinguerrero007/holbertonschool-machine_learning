# 0x0E. Time Series Forecasting


### Connect with Google drive
![1](https://user-images.githubusercontent.com/85587286/210636424-a7792e25-40cc-4190-8550-c0facc022379.png)

#### Dependencies
![2](https://user-images.githubusercontent.com/85587286/210636602-6da10401-d377-43c7-b35e-d693ceb3e3b8.png)

### Universal constant
![3](https://user-images.githubusercontent.com/85587286/210636619-2dd53929-2c9b-4ed6-97ca-9e43c097a177.png)

>> * PS: Try to provide and fork dataset from this links
>> * https://drive.google.com/file/d/1-2qi4aEYeMuvESgPemzWUgRhwZwmMOE8/view
>> * https://drive.google.com/file/d/1-1k5W0hhJPCt7Kaxrw6ECsYcWxCiMdt7/view

BITSTAMP_CSV = '/content/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
COINBASE_CSV = '/content/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'


### Extract dataframe from csv
![4](https://user-images.githubusercontent.com/85587286/210636631-affd3628-2458-4d0e-bed8-ffb5f3641b59.png)


### Data preprocessing
We need to check the percentage of missing value per column, try to visualize dataframe. In addition the dataframe is windowed per 60 second so we need to recast for need by windowing hourly
![5](https://user-images.githubusercontent.com/85587286/210636639-f2a71b28-909c-4ddf-a9b8-93ac527d89ed.png)

### There is a big difference between the count of timestamp (raws) and the available value per column
![6](https://user-images.githubusercontent.com/85587286/210636649-48c5dbc1-14c0-443f-8fcc-17cd541f2b5f.png)

### In order to handle this missing value we use teh Forward Filling method in order to cast this missing value
### I find this resource on Kaggle https://www.kaggle.com/juejuewang/handle-missing-values-in-time-series-for-beginners
![7](https://user-images.githubusercontent.com/85587286/210636652-86badbb4-d89d-4d0b-91e4-7ff8969f02c3.png)

### let's drop the Timestamp column
![8](https://user-images.githubusercontent.com/85587286/210636658-b083bbf5-bace-4e7c-bcad-3ad02a89d0d3.png)

### Convert out dataframe to be hourly windowed instead of minutes with a shift by 8 row
![9](https://user-images.githubusercontent.com/85587286/210636672-b2c2ead8-459b-41cc-81a9-5abd6b5f482f.png)

### Data correlation
![10](https://user-images.githubusercontent.com/85587286/210636680-f63e0ff7-6a71-423c-80a0-d35a9a019a49.png)

### Observation
>> * There are a fully correlated Features which make working of the full features a redundant process

![11](https://user-images.githubusercontent.com/85587286/210636689-edf3b54a-1378-4418-b79c-3b0788a33fae.png)


### Visualizing data
![12](https://user-images.githubusercontent.com/85587286/210636695-2570d719-0bb4-4645-b3cf-133779f8b0ac.png)


### Split data
![13](https://user-images.githubusercontent.com/85587286/210636707-dc685926-2691-4ee9-a389-b6795f27e8ea.png)


#### Normalization
![14](https://user-images.githubusercontent.com/85587286/210636714-28cf77e6-3399-4d4c-809c-eebc40f12351.png)


### Split to window
![15](https://user-images.githubusercontent.com/85587286/210636723-0060e065-3c84-476e-92ea-940d3d279e1f.png)


### Build datasets
![16](https://user-images.githubusercontent.com/85587286/210636731-79058c8d-cd3d-4fed-8866-b3387f4dd3a7.png)


### Create train and validation datasets
![17](https://user-images.githubusercontent.com/85587286/210636742-853c1e3e-eb0a-43f5-9840-5099561fa886.png)

### Compile and fit model
![18](https://user-images.githubusercontent.com/85587286/210636745-62f22fde-a625-4005-8e22-b43fb400e26b.png)


### Build RNN module using LSTM Gate
![19](https://user-images.githubusercontent.com/85587286/210636748-29936248-c598-400a-9414-6708a72c914d.png)


### Train model
![20](https://user-images.githubusercontent.com/85587286/210636760-04c56e55-2acb-4fd5-bf70-ea4f37263af8.png)



![21](https://user-images.githubusercontent.com/85587286/210636769-88af9961-ae8c-4bf6-bdc5-2082b7484880.png)


### Plotting prediction
![22](https://user-images.githubusercontent.com/85587286/210636776-b2b1d482-7dd8-4d24-b477-97134cc26a03.png)
