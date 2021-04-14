
import pandas as pd
data=pd.read_csv("/home/abhi/Downloads/train.csv")
data.head()


x=data.iloc[:,2:]
y=data.iloc[:,1]

x.head()
y.head()

import evalml
x_train,x_test,y_train,y_test=evalml.preprocessing.split_data(x,y,problem_type="binary")

x_train.head()


evalml.problem_types.ProblemTypes.all_problem_types


from evalml.automl import AutoMLSearch
automl=AutoMLSearch(X_train=x_train,y_train=y_train,problem_type="binary")
automl.search()


#getting the best pipeline

automl.best_pipeline
best_pipeline=automl.best_pipeline

# detailed description

automl.describe_pipeline(automl.rankings.iloc[0]['id'])


best_pipeline.score(x_train,y_train,objectives=["auc","f1","Precision","Recall"])
