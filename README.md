# Bitcoin-Price-Prediction-App
This is a simple desktop application built with Python and PyQt5 that predicts the price of Bitcoin for a given number of days.

This is a Python script that uses PyQt5 to create a GUI application for predicting Bitcoin prices. The application retrieves historical price data for Bitcoin from Binance using their API and predicts the future price based on a linear regression model.

The GUI has a text box where the user can enter the number of days they want to predict the price for, and a button to initiate the prediction. The predicted Bitcoin price is displayed in a label below the button.

The script defines a class App that inherits from QWidget, which is the base class for all user interface objects in PyQt5. The initUI method creates the layout of the application, adding labels, text boxes, and buttons to the window.

The get_data method retrieves the historical price data from Binance using their API and returns it as a Pandas DataFrame.
