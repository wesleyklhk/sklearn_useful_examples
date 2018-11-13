from sklearn.model_selection import PredefinedSplit
import pandas as pd
import numpy as np
import pdb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss,make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.scorer import neg_log_loss_scorer

########dataset
X1 = np.arange(1,11)
X2 = X1**2

y = np.array(['LOW'] * 5 + ['HI'] * 5)


X = pd.DataFrame({'X1':X1,'X2':X2}).values
print('dataset')
print('X')
print(X)

print('y')
print(y)
print()
print()
print('------------------------------')

########manually splitting into training and validation set
test_fold = [-1] * 2 + [0] * 2 + [-1] * 2 + [0] * 2 + [-1] * 2
ps = PredefinedSplit(test_fold)




########search for parameters using own scoring function
le = LabelEncoder()
y = le.fit_transform(y)

def my_own_scorer(clf, X, y_true):
    class_labels = clf.classes_
    loss = log_loss(y_true,clf.predict_proba(X),class_labels)
    return loss

Cs = [0.1,0.3,0.5,1,3,5]
tols = [0.01,0.03]


gs = GridSearchCV(
        estimator=LogisticRegression(random_state=0), ######machine learning algorithm
        param_grid={'C': Cs,'tol':tols},#####list of parameters to search for
        cv=ps.split(), #######evaluate performance on training and validation set
        verbose=True,
        scoring=my_own_scorer######evaluate the performance using this scoring function
        )

model = gs.fit(X,y)

print(pd.DataFrame(model.cv_results_) )



########search for parameters using sklearn predefined scorer
le = LabelEncoder()
y = le.fit_transform(y)

Cs = [0.1,0.3,0.5,1,3,5]
tols = [0.01,0.03]


gs = GridSearchCV(
        estimator=LogisticRegression(random_state=0), ######machine learning algorithm
        param_grid={'C': Cs,'tol':tols},#####list of parameters to search for
        cv=ps.split(), #######evaluate performance on training and validation set
        verbose=True,
        scoring=neg_log_loss_scorer######evaluate the performance using this scoring function
        )

model = gs.fit(X,y)

print( pd.DataFrame(model.cv_results_) )

