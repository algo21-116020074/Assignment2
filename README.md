# Stock Price Prediction Based on ```LSTM Neural Network```

## 1. Abstract

In this project, the application of deep learning model -- ``long and short term memory network LSTM`` in the field of stock price prediction will be introduced, and the prediction code under the framework of Python language ```TensorFlow (Keras)``` will be learnt.





## 2. LSTM Structure

### 2.1 Definition & Advantages:

Long Short-term Memory Networks (LSTM) is a kind of *special Recurring neural network (RNN)*. Recurring neural networks have certain advantages when learning the **nonlinear** characteristics of a sequence because it is **memorability, parameter sharing and Turing completeness**. Recurring neural networks have been used in Natural Language Processing (NLP), such as speech recognition, Language modeling, machine translation and other fields, and also used in various time series prediction.

### 2.2 Structure

Through the module chain network constituted by the **LSTM cell units**, the future can be predicted through continuous learning of the input data
The following figure is the neural unit structure of the LSTM neural network. Specific and detailed introduction of LSTM structure is in the attached paper.

![image](https://github.com/algo21-116020074/Assignment2/blob/main/LSTM_Image.png)

## 3. LSTM Python Coding Part

### 3.1 Data 

#### Data Acquisition

Data gained from JoinQuant: select _"600519.XSHG" MaoTai_ as stock_code
```
df = get_bars(security="600519.XSHG", count=5000, unit='1d',
         fields=['date','open','high','low','close','volume','money'],
         include_now=False, end_dt=None, fq_ref_date=None, df=True)
```

#### Data Process: Standardization

The data dimension is different, the value size is very different. Therefore, we need to introduce data standardization. Data standardization processing mainly includes data synchemotaxis and dimensionless processing. There are many methods of data standardization: min-max standardization, z-score standardization.

_**Min-max standardization formula is: new data = (original data - minimum value)/(maximum - minimum value)**_

_**Z-score standardization formula is: new data = (original data - mean)/standard deviation**_

The standardization method adopted in this case is Z-score standardization first, then min-max standardization.

### 3.2 Parameters Setting

(1) ```look_back = 50; forward_days = 5```

Use past 50-day closing price data to predict future 5-day closing price. 

(2) ```train_test_split(X, y, test_size=0.20, random_state=42)```

Train set and test set are splitted by 4:1

(3) 
```
model = keras.Sequential()
model.add(layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1:])))
model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.LSTM(32))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(5)) #用于预测未来5天

model.compile(optimizer=keras.optimizers.Adam(), loss='mae',metrics=['accuracy'])
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.7, min_lr=0.000000005)

history = model.fit(X_train, y_train,
                    batch_size = 128,
                    epochs=70,
                    validation_data=(X_validate, y_validate),
                    callbacks=[learning_rate_reduction])
```

Tensorflow is used as the deep learning structure. Set 4 layers: first LSTM layer's dimension is 64; second LSTM layer's dimension is 64; third LSTM layer's dimension is 32; Forth Dropout layer's dimension is 5 and dropout rate is 0.1, which is to predict future 5-day. Adam optimizer was used for estimation parameters, LR attenuation mode was adopted for learning rate, and the maximum number of iterations was set to 70 times. 

According to the curve of loss; If the accuracy of the curve after the train is stable is too much higher than that of the test, it is generally overfitting.
