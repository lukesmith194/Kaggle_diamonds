# Machine Learning project on Diamonds

The purpose of this project is to fine the best ML method to get the best Root mean squared error score for a Kaggle competition. 
### Content: 
In this repo you will find various folders with different  
 - Models : Contains all CSV from exploration and cleaning as well as exported CSV´s to upload to Kaggle.
 - Notebooks : Jupyter notebooks showing exploration, cleaning and running of ML models

## Method
First of all i explored the dataset we were given for this competition and found various columns where catergorical. I changed these into numerical columns as they were all a scale which was easily convertible. Then the correlation was checked and the new CSV was exported in order to start with the predictive models. A train test split was also implemented with the train being 80% of our data to be able to do a comparison.


## Algorithms 
- Linear regression [Here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)  
    - This was the first basic model that was used in order to check how the data functioned to this model. 
- Decision tree [Here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
    - Secondly a decision tree was used in order to check the data more in depth and find a more accurate model. Various different depths were used, finding the optimal one to be that of 5.
- Random Forrest [Here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
    - Various random forrests were used. _n_estimators_, _max_features_, _max_depth_ and _min_samples_split_ were the main parameters used, finding the optimal forrest the one that was the largest. 
- XGBoost [Here](https://xgboost.readthedocs.io/en/latest/)
    - XGBoost is a powerful algorithm that iterates its tries over the data and learns from its mistakes. One of the best predictors.

- CatBoost [Here](https://xgboost.readthedocs.io/en/latest/)
    - Also a powerful algorithm, does similar to XGboost but you have the posibility to see how it learns over time and are able to choose parameters such as _iterations_,_learning_rate_ and _depth_ .

## Conclusions
As expected CatBoost was the best at predicting the root mean square error between my test and train sets.


### Libraries
-	Pandas [Here](https://pandas.pydata.org/docs/)
-	numpy [Here](https://numpy.org/doc/)
-	Seaborn [Here](https://seaborn.pydata.org/)
-   Skilearn [Here](https://scikit-learn.org/0.21/documentation.html)
-   Catboost [Here](https://catboost.ai/docs/concepts/about.html)
-   XGboost [Here](https://xgboost.readthedocs.io/en/latest/)
