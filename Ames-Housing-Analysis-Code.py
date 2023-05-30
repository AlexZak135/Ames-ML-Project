# Title: Ames Housing Statistical Analysis
# Author: Alexander Zakrzeski
# Date: January 17, 2023

# Import the necessary libraries and modules

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt
import seaborn as sns
import plotnine as pn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import janitor as jan
import movecolumn as mc

##### Part A - Data Preprocessing ##### 

# Take the following steps: 
  # 1. Create a list
  # 2. Import the data
  # 3. Consolidate the column names
  # 4. Rename a specific column
  # 5. Create new columns
  # 6. Modifying existing columns
  # 7. Drop columns
   
cols1_drop = ["pid", "ms_subclass", "year_built", "year_remod_add",
              "fireplaces", "garage_yr_blt", "mo_sold", "yr_sold"]  

data1_all = (
    jan.clean_names(pd.read_csv("Ames_Data.csv"))
       .rename(columns = {"order": "id"})
       .assign(age = lambda x: x["yr_sold"] - x["year_built"])
       .assign(age = lambda x: np.where(x["age"] < 0, 0, x["age"]),
               overall_cond = lambda x: 
                   np.where(x["overall_cond"] < 5, "below",
                   np.where(x["overall_cond"] == 5, "average",                  
                                                      "above")),
               overall_qual = lambda x:
                   np.where(x["overall_qual"] < 5, "below",
                   np.where(x["overall_qual"] == 5, "average", 
                                                      "above")),
               kitch_qual = lambda x:
                   np.where((x["kitchen_qual"] == "Po") | 
                            (x["kitchen_qual"] == "Fa"), "below",
                   np.where(x["kitchen_qual"] == "TA", "average", 
                                                         "above")),
               fplace = lambda x: np.where(x["fireplaces"] > 0, "yes", "no"))
       .drop(columns = cols1_drop))
      
# Take the following steps:
  # 1. Select columns with certain data types
  # 2. Change the data type of a specific column
  # 3. Replace missing values with 0
  
data1_num = (data1_all
             .select_dtypes(include = np.number)
             .astype({"id": str})
             .fillna(0))

# Take the following steps:
  # 1. Calculate the Pearson correlation coefficients
  # 2. Sort by the selected column
  # 3. Filter appropriately
  # 4. Select the necessary column
  # 5. Get the index of the column as a list
  
corr1 = (data1_num
         .corr(numeric_only = True)
         .sort_values(by = "price", ascending = False)
         .query("price > 0.45 | price < -0.45")
         .price
         .index
         .tolist())

# Define the name of the extra column 

cols1_extra = "id"

# Append the column

if cols1_extra not in corr1:
    corr1.append(cols1_extra)

# Select the columns that are in the list

data1_num = data1_num[corr1]

# Take the following steps:
  # 1. Create a list
  # 2. Change the data type of a specific column
  # 3. Select columns with certain data types
  # 4. Select the necessary columns
   
cols1_keep = ["id", "overall_cond", "overall_qual", "kitch_qual", "fplace"]  
  
data1_str = (data1_all
             .astype({"id": str})
             .select_dtypes(include = object)
             [cols1_keep])
   
# Take the following steps:
  # 1. Create a list
  # 2. Merge the data frames
  # 3. Rename certain columns
  # 4. Drop a specific column
  # 5. Reorder the columns

cols1_order = ["price", "abv_grd_sf", "garage_cars", "garage_sf", 
               "bsmt_sf","x1st_flr_sf", "full_bath", "mas_area", 
               "totrms_abvgrd", "age", "kitch_qual", "fplace", 
               "overall_cond", "overall_qual"]
   
data2 = (data1_num
         .merge(data1_str, on = "id", how = "left")
         .rename(columns = {"area": "abv_grd_sf",
                            "garage_area": "garage_sf",
                            "total_bsmt_sf": "bsmt_sf",
                            "mas_vnr_area": "mas_area"})
         .drop(columns = "id")
         [cols1_order])
          
##### Part B - Exploratory Data Analysis ##### 

# Select columns with certain data types

data2_num = data2.select_dtypes(include = np.number)

# Take the following steps:
  # 1. Create a dictionary
  # 2. Generate the descriptive statistics
  # 3. Rename index
  
index1_rename = {"count": "Count", 
                 "mean": "Mean", 
                 "std": "Standard Deviation", 
                 "min": "Minimum", 
                 "max": "Maximum"}

ds1 = (data2_num
       .agg(["count", "mean", "std", "min", "max"])
       .rename(index = index1_rename))

# Create a new row for the median

ds1.loc["Median"] = data2_num.median()

# Create a new row for the mode

ds1.loc["Mode"] = data2_num.mode().iloc[0]

# Take the following steps:
  # 1. Create a dictionary
  # 2. Rename columns
  # 3. Transpose the data frame 
  # 4. Modify the position of columns

cols1_rename = {"price": "Price", "abv_grd_sf": "Abv. Grd. SF",
                "garage_cars": "Garage Doors", "garage_sf": "Garage SF",
                "bsmt_sf": "Basement SF",  "x1st_flr_sf": "1st Flr. SF",
                "full_bath": "Bathrooms", "mas_area": "Masonry SF",
                "totrms_abvgrd": "Rooms Abv. Grd.", "age": "Age"} 

ds1 = (ds1
       .rename(columns = cols1_rename)
       .T
       .pipe(mc.MoveToN, "Median", 3)
       .pipe(mc.MoveToN, "Mode", 4))

# Take the following steps:
  # 1. Select columns with certain data types
  # 2. Rename all the columns
  # 3. Calculate the Pearson correlation coefficients  
  
corr2 = (data2
         .select_dtypes(include = np.number)
         .rename(columns = cols1_rename)
         .corr())
  
# Create a correlation matrix displaying the Pearson correlation coefficients 

plt.figure(figsize = (8, 6))
color = sns.diverging_palette(10, 240, as_cmap = True)
cmat = sns.heatmap(corr2, vmin = -1, vmax = 1, annot = True, cmap = color,
                   linewidths = 0.5, linecolor = "black", xticklabels = True, 
                   yticklabels = True, annot_kws = {"color": "black"})
cmat.tick_params(bottom = False, left = False)
cmat.set_title("Figure 1 - Correlation Matrix of Numeric Analysis Variables", 
               fontdict = {"fontsize": 18}, pad = 12)

plt.show()

# Take the following steps:
  # 1. Modifying existing column
  # 2. Select the necessary columns
  # 3. Calculate Spearman's rank correlation coefficient
   
corr3 = (data2
         .assign(kitch_qual = 
                 np.where(data2["kitch_qual"] == "below", 1,
                 np.where(data2["kitch_qual"] == "average", 2, 
                                                            3)))
         [["price", "kitch_qual"]])

corrtest3 = stats.spearmanr(corr3["kitch_qual"], corr3["price"])

# Take the following steps:
  # 1. Modifying existing column
  # 2. Select the necessary columns
  # 3. Calculate the point biserial correlation coefficient 
    
corr4 = (data2
         .assign(fplace = np.where(data2["fplace"] == "yes", 1, 0))
         [["price", "fplace"]])

corrtest4 = stats.pointbiserialr(corr4["fplace"], corr4["price"])
        
# Take the following steps:
  # 1. Modifying existing column
  # 2. Select the necessary columns
  # 3. Calculate Spearman's rank correlation coefficient

corr5 = (data2
         .assign(overall_cond = 
                 np.where(data2["overall_cond"] == "below", 1,
                 np.where(data2["overall_cond"] == "average", 2, 
                                                              3))) 
         [["price", "overall_cond"]])

corrtest5 = stats.spearmanr(corr5["overall_cond"], corr5["price"])

# Take the following steps:
  # 1. Modifying existing column
  # 2. Select the necessary columns
  # 3. Calculate Spearman's rank correlation coefficient

corr6 = (data2
         .assign(overall_qual = 
                 np.where(data2["overall_qual"] == "below", 1,
                 np.where(data2["overall_qual"] == "average", 2, 
                                                              3))) 
         [["price", "overall_qual"]])

corrtest6 = stats.spearmanr(corr6["overall_qual"], corr6["price"])

# Create a histogram displaying the distribution

(pn.ggplot(data = data2, mapping = pn.aes(x = "price")) + 
 pn.geom_histogram(fill = "#005288", color = "black", bins = 25) + 
 pn.scale_x_continuous(labels = lambda x: ["${:,.0f}".format(i) for i in x]) + 
 pn.labs(title = "Figure 2 - Distribution of Price", 
         x = "Price", y = "Frequency") + 
 pn.theme_bw() +
 pn.theme(plot_title = pn.element_text(hjust = 0.5), 
          text = pn.element_text(size = 14)))

# Create a new variable

data2["log_price"] = np.log(data2["price"])

# Create a histogram displaying the distribution

(pn.ggplot(data = data2, mapping = pn.aes(x = "log_price")) + 
 pn.geom_histogram(fill = "#c0c2c4", color = "black", bins = 25) + 
 pn.labs(title = "Figure 3 - Distribution of the Log of Price", 
         x = "Log of Price", y = "Frequency") + 
 pn.theme_bw() +
 pn.theme(plot_title = pn.element_text(hjust = 0.5), 
          text = pn.element_text(size = 14)))

# Create a scatter plot displaying the relationship between the variables

(pn.ggplot(data = data2, mapping = pn.aes(x = "abv_grd_sf", y = "price")) + 
 pn.geom_point(color = "#005288", size = 0.6) +  
 pn.geom_smooth(method = "lm", se = False, size = 0.7) + 
 pn.scale_x_continuous(labels = lambda x: ["{:,.0f}".format(i) for i in x]) + 
 pn.scale_y_continuous(labels = lambda x: ["${:,.0f}".format(i) for i in x]) + 
 pn.labs(title = "Figure 4 - Price vs. Above Ground Sq. Ft.", 
         x = "Above Ground Sq. Ft.", y = "Price") + 
 pn.theme_bw() +
 pn.theme(plot_title = pn.element_text(hjust = 0.5), 
          text = pn.element_text(size = 14)))

# Create a function that identifies outliers using the IQR method

def get_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers

# Use function to get outliers using the IQR method for two different variables

outliers1 = get_outliers(data2, "abv_grd_sf")

outliers2 = get_outliers(data2, "price")
    
# Filter appropriately to produce a new data frame

data3 = data2.query("abv_grd_sf < 4000 & price > 15000")

# Create a scatter plot displaying the relationship between the variables

(pn.ggplot(data = data3, mapping = pn.aes(x = "abv_grd_sf", y = "log_price")) + 
 pn.geom_point(color = "#0078ae", size = 0.6) + 
 pn.geom_smooth(method = "lm", se = False, size = 0.7) + 
 pn.scale_x_continuous(labels = lambda x: ["{:,.0f}".format(i) for i in x]) + 
 pn.labs(title = "Figure 5 - Log of Price vs. Above Ground Sq. Ft.", 
         x = "Above Ground Sq. Ft.", y = " Log of Price") + 
 pn.theme_bw() +
 pn.theme(plot_title = pn.element_text(hjust = 0.5), 
          text = pn.element_text(size = 14)))

# Create a scatter plot displaying the relationship between the variables

(pn.ggplot(data = data2, mapping = pn.aes(x = "age", y = "price")) + 
 pn.geom_point(color = "#005288", size = 0.6) + 
 pn.geom_smooth(method = "lm", se = False, color = "black", size = 0.7) + 
 pn.scale_y_continuous(labels = lambda x: ["${:,.0f}".format(i) for i in x]) + 
 pn.labs(title = "Figure 6 - Price vs. Age", x = "Age", y = "Price") + 
 pn.theme_bw() +
 pn.theme(plot_title = pn.element_text(hjust = 0.5), 
          text = pn.element_text(size = 14)))

# Create a scatter plot displaying the relationship between the variables

(pn.ggplot(data = data3, mapping = pn.aes(x = "age", y = "log_price")) + 
 pn.geom_point(color = "#0078ae", size = 0.6) + 
 pn.geom_smooth(method = "lm", formula = "y ~ x + I(x**2)", se = False, 
                color = "black", size = 0.7) + 
 pn.labs(title = "Figure 7 - Log of Price vs. Age", 
         x = "Age", y = "Log of Price") + 
 pn.theme_bw() +
 pn.theme(plot_title = pn.element_text(hjust = 0.5), 
          text = pn.element_text(size = 14)))

# Create a new variable

data3["age2"] = np.power(data2["age"], 2)

##### Part C - Statistical Modeling #####       

# Take the following steps:
  # 1. Convert categorical variables to dummy variables
  # 2. Drop columns 
  # 3. Modify the position of the columns

data3 = (data3
         .join(pd.get_dummies(data3[["kitch_qual", "fplace", 
                                     "overall_cond", "overall_qual"]],
                              prefix = ["kitch_qual", "fplace", 
                                        "overall_cond", "overall_qual"]))
         .drop(columns = ["kitch_qual", "fplace", 
                          "overall_cond", "overall_qual"])
         .pipe(mc.MoveToN, "log_price", 2)
         .pipe(mc.MoveToN, "age2", 12))

### Polynomial Regression

# Take the following steps:
  # 1. Create a list
  # 2. Create a data frame with the necessary independent variables
  # 3. Create a series with just the dependent variable
  # 4. Use hold-out validation to split data into training and testing sets
  # 5. Save function as an object and fit the model to the training data
  # 6. Generate regression output

cols2_keep = ["abv_grd_sf", "garage_sf", "bsmt_sf", "age", "age2", 
              "kitch_qual_below", "kitch_qual_average", "fplace_yes",
              "overall_cond_below", "overall_cond_average",
              "overall_qual_below", "overall_qual_average"]
            
iv1 = data3[cols2_keep]

dv1 = data3["log_price"]

iv1_train, iv1_test, dv1_train, dv1_test = train_test_split(
    iv1, dv1, test_size = 0.3, shuffle = True, random_state = 100)

polreg1 = LinearRegression()

polreg1.fit(iv1_train, dv1_train)

polreg1_output = sm.OLS(dv1_train, sm.add_constant(iv1_train)).fit().summary()

# Take the following steps:
  # 1. Make predictions on the training data
  # 2. Generate the model's residuals
  # 3. Create a data frame with the fitted values and residuals
  # 4. Produce residuals vs. fitted plot
  # 5. Produce a histogram of the residuals
  # 6. Use custom function to get VIFs

polreg1_train_pred = polreg1.predict(iv1_train)

polreg1_train_resid = dv1_train - polreg1_train_pred 

rfplot1_df = pd.DataFrame({"fitted": polreg1_train_pred, 
                           "resid": polreg1_train_resid})

(pn.ggplot(data = rfplot1_df, mapping = pn.aes(x = "fitted", y = "resid")) +
 pn.geom_point(color = "#005288", size = 0.6) +
 pn.geom_hline(yintercept = 0, color = "grey", linetype = "dashed") +
 pn.labs(title = "Residuals vs. Fitted Plot",
         x = "Fitted Values", y = "Residuals") +
 pn.theme_bw() +
 pn.theme(plot_title = pn.element_text(hjust = 0.5), 
          text = pn.element_text(size = 14)))

(pn.ggplot(data = rfplot1_df, mapping = pn.aes(x = "resid")) + 
 pn.geom_histogram(fill = "#c0c2c4", color = "black", bins = 25) + 
 pn.labs(title = "Distribution of Residuals", 
         x = "Residuals", y = "Frequency") +
 pn.theme_bw() +
 pn.theme(plot_title = pn.element_text(hjust = 0.5), 
          text = pn.element_text(size = 14)))

def compute_vifs(data):
    iv_with_intercept = sm.add_constant(data)
    vifs = [variance_inflation_factor(iv_with_intercept.values, i) 
            for i in range(iv_with_intercept.shape[1])]
    vif_series = pd.Series(vifs[1:], index = data.columns)
    return vif_series

compute_vifs(iv1_train.drop(columns = "age2"))

# Take the following steps:
  # 1. Get r-squared
  # 2. Make predictions on the test data
  # 3. Get the RMSE

polreg1.score(iv1_test, dv1_test)

polreg1_test_pred = polreg1.predict(iv1_test)

rmse1 = np.sqrt(mean_squared_error(dv1_test, polreg1_test_pred))

### Lasso Regression

# Take the following steps:
  # 1. Create a data frame with the necessary independent variables
  # 2. Create a series with just the dependent variable
  # 3. Use hold-out validation to split data into training and testing sets
  # 4. Standardize the necessary variables
  # 5. Use 5-fold cross-validation to find optimal hyperparameter  
  # 6. Save function as an object and fit the model to the training data
  # 7. Get the r-squared

cols3_keep = ["abv_grd_sf", "garage_cars", "garage_sf", "bsmt_sf",
              "x1st_flr_sf", "full_bath", "mas_area", "totrms_abvgrd",
              "age", "age2", "kitch_qual_below", "kitch_qual_average",
              "fplace_yes", "overall_cond_below", "overall_cond_average",
              "overall_qual_below", "overall_qual_average"]

iv2 = data3[cols3_keep]

dv2 = data3["log_price"]

iv2_train, iv2_test, dv2_train, dv2_test = train_test_split(
    iv2, dv2, test_size = 0.3, shuffle = True, random_state = 100)

def standardize_df(df):
    df_stand = df.copy()
    num_cols = (df_stand.select_dtypes(include = np.number)
                        .select_dtypes(exclude = "uint8").columns)
    scaler = StandardScaler()
    scaler.fit(df_stand[num_cols])
    df_stand[num_cols] = scaler.transform(df_stand[num_cols])
    return df_stand

iv2_train = standardize_df(iv2_train)

iv2_test = standardize_df(iv2_test)

lasso1_cv = LassoCV(cv = 5, random_state = 100)

lasso1_cv.fit(iv2_train, dv2_train) 

lasso1 = Lasso(alpha = lasso1_cv.alpha_)

lasso1.fit(iv2_train, dv2_train)

lasso1.score(iv2_train, dv2_train)

# Take the following steps:
  # 1. Get r-squared
  # 2. Make predictions on the test data
  # 3. Get the RMSE

lasso1.score(iv2_test, dv2_test)

lasso1_pred = lasso1.predict(iv2_test)

rmse2 = np.sqrt(mean_squared_error(dv2_test, lasso1_pred))

### K-Nearest Neighbors

# Take the following steps:
  # 1. Use hold-out validation to split data into training and testing sets
  # 2. Save function as an object and perform hyperparameter tuning
  # 3. Fit the model to the training data
  # 4. Get the r-squared

iv3_train, iv3_test, dv3_train, dv3_test = train_test_split(
    iv2, dv2, test_size = 0.3, shuffle = True, random_state = 100)

knn1 = KNeighborsRegressor()

hp1_grid = {"n_neighbors": [4, 8, 12, 16, 20]}

knn1_cv = GridSearchCV(knn1, param_grid = hp1_grid, cv = 5)

knn1_cv.fit(iv3_train, dv3_train)

knn1 = knn1_cv.best_estimator_

knn1.fit(iv3_train, dv3_train)

knn1.score(iv3_train, dv3_train)

# Take the following steps:
  # 1. Get r-squared
  # 2. Make predictions on the test data
  # 3. Get the RMSE
  
knn1.score(iv3_test, dv3_test)

knn1_test_pred = knn1.predict(iv3_test)

rmse3 = np.sqrt(mean_squared_error(dv3_test, knn1_test_pred))

### Random Forest

# Take the following steps:
  # 1. Use hold-out validation to split data into training and testing sets
  # 2. Perform hyperparameter tuning
  # 3. Save function as an object and fit the model to the training data
  # 4. Get the r-squared

iv4_train, iv4_test, dv4_train, dv4_test = train_test_split(
    iv2, dv2, test_size = 0.3, shuffle = True, random_state = 100)

hp2_grid = {"n_estimators": [200, 250, 500],
            "max_depth": [20, 21, 22],
            "min_samples_split": [4]}

rf1_cv = GridSearchCV(RandomForestRegressor(random_state = 100), 
                      hp2_grid, cv = 5)

rf1_cv.fit(iv4_train, dv4_train)

rf1_bp = rf1_cv.best_params_

rf1 = RandomForestRegressor(n_estimators = rf1_bp["n_estimators"], 
                            max_depth = rf1_bp["max_depth"],
                            min_samples_split = rf1_bp["min_samples_split"],
                            random_state = 100)

rf1.fit(iv4_train, dv4_train)

rf1.score(iv4_train, dv4_train)

# Take the following steps:
  # 1. Get r-squared
  # 2. Make predictions on the test data
  # 3. Get the RMSE

rf1.score(iv4_test, dv4_test)

rf1_test_pred = rf1.predict(iv4_test)

rmse4 = np.sqrt(mean_squared_error(dv4_test, rf1_test_pred))