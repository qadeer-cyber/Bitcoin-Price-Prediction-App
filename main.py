import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QMessageBox
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Bitcoin Price Prediction'
        self.left = 50
        self.top = 50
        self.width = 500
        self.height = 300
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.label1 = QLabel('Enter the number of days for prediction:', self)
        self.label1.move(50, 50)
        self.textbox1 = QLineEdit(self)
        self.textbox1.move(300, 50)
        self.textbox1.resize(100, 25)

        self.label2 = QLabel('Predicted Bitcoin price:', self)
        self.label2.move(50, 100)
        self.label3 = QLabel('', self)
        self.label3.move(300, 100)

        self.button = QPushButton('Predict', self)
        self.button.move(200, 200)
        self.button.clicked.connect(self.predict_price)

        self.show()

    def get_data(self):
        try:
            url = 'https://api.binance.com/api/v3/klines'
            params = {'symbol': 'BTCUSDT', 'interval': '1d'}
            res = requests.get(url, params=params)
            data = pd.DataFrame(res.json(),
                                columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                         'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                         'taker_buy_quote_asset_volume', 'ignore'])
            data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
            data = data[['open_time', 'close']]
            data.set_index('open_time', inplace=True)
            return data
        except Exception as e:
            QMessageBox.critical(self, 'Error',
                                 f'Unable to retrieve Bitcoin price data. Error message: {str(e)}\nPlease check your internet connection and try again.')

    def predict_price(self):
        try:
            days = int(self.textbox1.text())
            if days <= 0:
                QMessageBox.critical(self, 'Error', 'Please enter a positive integer for the number of days.')
                return

            # Get data from Binance
            data = self.get_data()
            if data is None:
                return
            last_date = data.index[-1]

            # Add time step column
            data['timestep'] = np.arange(len(data))

            # Scale data
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data[['timestep', 'close']])

            # Prepare data for model
            data_scaled = pd.DataFrame(data, columns=['timestep', 'close'])
            scaler = MinMaxScaler()
            data_scaled[['timestep', 'close']] = scaler.fit_transform(data_scaled[['timestep', 'close']])

            # Split data into training and testing sets
            train_data = data_scaled[:-days]
            test_data = data_scaled[-days:]

            # Split data into input and output variables
            X_train, y_train = train_data[['timestep']], train_data['close']
            X_test, y_test = test_data[['timestep']], test_data['close']

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            inputs = data_scaled[-days:][['timestep']]
            predictions = model.predict(inputs)
            predictions = scaler.inverse_transform(np.hstack((inputs, predictions.reshape(-1, 1))))
            predictions = predictions[:, -1]

            # Display predictions
            self.label3.setText('${:.2f}'.format(predictions[-1]))
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error occurred while predicting Bitcoin price: {str(e)}')
            return


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
