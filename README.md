# House Price Prediction 🏠

This is my learning project where I tried to predict house prices with linear regression.
I wrote the model myself with numpy (without sklearn models) because I wanted to understand how gradient descent really works.

## Dataset

The data is in `house_prices.csv`. It has 21613 houses sold in King County (USA) and 21 columns: price, bedrooms, bathrooms, square feet, condition, year built, location and more.

## What I did

1. Loaded the csv file and looked at all the columns
2. Found that `waterfront` and `condition` are text, so I changed them to numbers (`N`/`Y` -> 0/1, `Poor` ... `Very Good` -> 1 ... 5)
3. Made scatter plots of every feature vs price to see which ones are useful
4. Chose 9 features: bedrooms, bathrooms, sqft_living, grade, sqft_above, sqft_basement, yr_renovated, lat, sqft_living15
5. Normalized the features with Z-score (StandardScaler)
6. Wrote my own `MyModel` class with:
   - prediction `X @ w + b`
   - MSE loss
   - gradients for w and b (with small L2 regularization)
   - gradient descent
7. Trained it for 10000 epochs with lr = 0.001 and plotted the loss curve

## Result

```
Epoch: 10000. Loss: 26010024912.92108
```

The loss looks really big, but it's because prices are in dollars and the loss is MSE / 2.
RMSE is around $228,000. The loss curve goes flat at the end, so the model converged.

## What I want to try next

- split data into train and test
- add more features (waterfront, view, zipcode)
- try log of price
- compare my model with sklearn

## How to run

Put `house_prices.csv` in the same folder as the notebook, then:

```
pip install -r requirements.txt
jupyter notebook house_prices_predict.ipynb
```
