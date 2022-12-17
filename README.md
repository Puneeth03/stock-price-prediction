# stock-price-prediction
This is a model to predict stock price using its history dataset using some deep learning(machine learning) through recurrent neural networks.

1. The dataset is first loaded and then unwanted columns are dropped from the dataset (data preprocessing)
2. Some useful features can be created which helps in better performance of the model.
   So, moving average of 5-day is added to the added to the data along with 'Open', 'Low', 'High' and 'Volume' (feature creation)
3. Features are scaled(normalized) using minmaxscalar within the range (0,1).
4. We split the dataset into train data and test data with a window size of 22(experimental value).
5. After splitting the data, the train data is used to train the model built using a sequence of lstm, convolutional, dropout and dense layers.
   Note: All the numerical values found by trail and error model because values can be different for each stock but valid for most of the stocks.
6. The model is compiled using 'adam' optimizer and 'root mean error' as loss function.
7. The model is then fitted with 15 batch size and 20 epochs.
8. Then the values are predicted for test data and displayed.

This model show high performance and accuracy but cannot be used directly because there are many other factors that the price of a stock depend on. The model is to only give some insights how stock MIGHT move.
