# Linear Regression Practice
This will be an easy linear regression model just for practice (because I like repetition and it’s fun). We will be trying to figure out the best features to feed our model to predict the miles per gallon.

There might be a better model to use but the whole point of this is to use LinearRegression from sklearn.
The data file I will be looking at is from UCI. Here is the link for the specific data. You can read that into pandas.
I would recommend researching anything you do not understand because it will help you learn it better and from different perspectives, but I always welcome questions.

```
Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
%matplotlib inline
```

## Read Data
```
# We have to set the column names since it does not come with one
col_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
# Setting the file to the url location of dataset.
f = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original'
# Loading the dataset into our dataframe
df = pd.read_csv(f, names=col_names, delim_whitespace=True)
```
I am grabbing the dataset directly from the website and it does not come with column names, so I’ll just set those myself from here. I need to set delim_whitespace=True since it is not separated by commas.
```
# For reference
df.head()
```
`mpg` — miles per gallon — continuous
`cylinders` — “A cylinder is the central working part of a reciprocating engine or pump, the space in which a piston travels.” (Wiki)
`displacement` — “Engine displacement is the swept volume of all the pistons inside the cylinders of a reciprocating engine in a single movement from top dead center (TDC) to bottom dead center (BDC).” (Wiki)
`horsepower` — “Horsepower (hp) is a unit of measurement of power or the rate at which work is done.” (Wiki)
`weight` — The weight of the vehicle.
`acceleration` — “A car’s acceleration is calculated when the car is not in motion (0 mph), until the amount of time it takes to reach a velocity of 60 miles per hour.” (Glenn Elert)
`model_year` — The year of the model.
`origin` — To be honest I couldn’t find what it was talking about.
`car_name` — The car’s name.
```
# Lets check it out
df.info()
```
We can see here that both mpg and horsepower both have null values because there are 406 entries and in those columns, it only shows 398 and 400 non-null, respectively. We will have to get rid of the rows with nulls in mpg since that is what we are trying to predict. As for horsepower, there are several things we could do but since the focus of this blog is linear regression and not handling nulls; we’ll simply use the mean or mode.

## Clean Data
#### Horsepower
```
# Take a look at distribution of horsepower to see if we want to
# use mean or mode
plt.hist(df['horsepower'], color='grey');
plt.axvline(df['horsepower'].mean(), color='orange');
plt.axvline(df['horsepower'].mode()[0], color='blue');
```
Taking a look at the horsepower distribution with mean (orange) and mode (blue) lines, we can see that it would be better to fill the nulls with the mean of the column. On a side note, pandas.Series.mode has a default parameter of ignoring nulls.
```
# Fill na of horsepower
df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)
Filling nulls is always a two-part process for me. I first check it without inplace=True and then I set the parameter to True.
```
#### MPG
```
# Get rid of na in mpg rows since that is what we are trying
# to predict
df.dropna(inplace=True)
```
The second column to clean is mpg which is simply removing the null rows. Easy enough.

#### Zeroes and Negatives
```
# Check if any are 0 or negative
(df.iloc[:,:-1] <= 0).any()
```
I always like to look through the numerical columns to see if there are any weird consistencies. In this case, not that we can see.

#### Data Exploration
```
# Looking at correlation if we made car name into a dummy variable
# does not look promising; we will not do that.
# Maybe later.
pd.get_dummies(df, columns=['car_name'], drop_first=True).corr()['mpg']
```
We are probably not going to make dummy features out of this because it just does not seem worth it for a Linear Regression model because it will quickly become a too complex model.
```
# Take a look at the relationship
sns.pairplot(df);
```
A couple of things that I did notice is some of the independent variables seem to not be so independent. So we will have to look into this.
```
# Create features
features = df._get_numeric_data().columns[1:]
# Create features as string
feature_names = ' + '.join(features)
# Get y and X dataframes
y, X = dmatrices('mpg ~ ' + feature_names, df, return_type='dataframe')
# Create VIF
vif = pd.DataFrame()
# Load vif factor
vif['vif factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# Load features
vif['features'] = X.columns
# Inspect
vif.round(1)
```
It looks like there’s a lot going on here but really all we are trying to do is inspect the VIF of our data. We do not want our VIF factor to be above 10. This means that they are depending too much on each other. However, we might only have to remove one, so we will remove displacement and see how that changes our VIF factor.
```
# Create features
features = list(df._get_numeric_data().columns[1:])
features.remove('displacement')
# Create features as string
feature_names = ' + '.join(features)
# Get y and X dataframes
y, X = dmatrices('mpg ~ ' + feature_names, df, return_type='dataframe')
# Create VIF
vif = pd.DataFrame()
# Load vif factor
vif['vif factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# Load features
vif['features'] = X.columns
# Inspect
vif.round(1)
```
There we go; everything is under 10.

Next thing we will want to look at is the P-values of the coefficients to see if we reject or not the null hypothesis that there is a correlation to the dependent variable, mpg.
```
# Creating X and y
X = df[features]
y = df['mpg']
# Adding a constant to X for OLS
X = sm.add_constant(X)
# Initiating OLS model
model = sm.OLS(y, X)
# Fitting model to results
results = model.fit()
# Looking at only the P values that are less than .05 to see which
# ones reject the null hypothesis
results.pvalues < .05
```
First, we ignore const and focus on others. We can see that we really only need weight, model_year, and acceleration as they are the only ones that we can successfully say to reject the null hypothesis (which is they do not influence mpg , in case you forgot).
```
features = ['weight', 'model_year', 'origin']
```
We will set the features according to the P-values.

#### Model
The moment we have been waiting for. Nothing too complicated here; just creating our model and looking at the r2 score of our training and testing data to see how it fairs.
```
# Create X and y
X = df[features]
y = df['mpg']
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# Initiate mode
lr = LinearRegression()
# Fit model
lr.fit(X_train, y_train)
# Get predictions
test_predict = lr.predict(X_test)
train_predict = lr.predict(X_train)
# Score and compare
print(r2_score(y_test, test_predict))
print(r2_score(y_train, train_predict))
```
Our training data is less than our testing data so we know it is not overfitting. We could probably do a thing or two to make it better, but for the scope of what we were trying to accomplish; it's simply good enough.

## Conclusion
We have seen that out of the columns, we only need a couple to fit our model and get a pretty decent score right off the bat.


<sub><sub>
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

Cylinder (engine). From Wikipedia, the free encyclopedia. 15 February 2019, at 04:47 (UTC). https://en.wikipedia.org/wiki/Cylinder_(engine)

Engine displacement. From Wikipedia, the free encyclopedia. 12 February 2019, at 00:04 (UTC). https://en.wikipedia.org/wiki/Engine_displacement.

Horsepower. From Wikipedia, the free encyclopedia. 9 March 2019, at 23:50 (UTC). https://en.wikipedia.org/wiki/Horsepower.

Meredith Barricella — 2001. Student of Glenn Elert, edited by Glenn Elert. Acceleration Of A Car. https://hypertextbook.com/facts/2001/MeredithBarricella.shtml
</sub></sub>
