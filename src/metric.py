# from sklearn.metrics import roc_auc_score
# import numpy as np
# import pandas as pd

# def rmse(y_true, y_predict):
#     """
#     Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

#     Parameters:
#     y_true (numpy array or list): Array of true values.
#     y_predict (numpy array or list): Array of predicted values.

#     Returns:
#     float: RMSE value.
#     """
#     y_true = np.array(y_true)
#     y_predict = np.array(y_predict)
    
#     # Calculate the difference between true and predicted values
#     error = y_true - y_predict
    
#     # Square the errors
#     squared_error = np.square(error)
    
#     # Calculate the mean of the squared errors
#     mean_squared_error = np.mean(squared_error)
    
#     # Take the square root of the mean squared error to get RMSE
#     rmse_value = np.sqrt(mean_squared_error)
    
#     return rmse_value