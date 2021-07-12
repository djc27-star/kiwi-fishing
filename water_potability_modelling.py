
# load libraries
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns

# read in dataset
dataset = read_csv("water_potability.csv")

# check for na 
if dataset.isnull().values.any() != False:
    print('null values present')

dataset.describe()
dataset.isnull().sum()
# ph 491 nas, sulfate 781 nas Trihalomethanes 162 nas

# remove nas with mean for each output
dataset['ph'] = dataset['ph'].fillna(dataset.groupby('Potability')
                                                     ['ph'].transform('mean'))

dataset['Sulfate'] = dataset['Sulfate'].fillna(dataset.groupby('Potability')
                                                               ['Sulfate'].transform('mean'))

dataset['Trihalomethanes'] = dataset['Trihalomethanes'].fillna(dataset.groupby('Potability')
                                                        ['Trihalomethanes'].transform('mean'))

# assess each variable has values in correct range and examine of the data
dataset.head(5)
dataset.describe()
dataset.columns

for i in dataset['ph']:
    assert isinstance(i,float) and i > 0
sns.distplot( dataset['ph'])
plt.show()
sns.boxplot(x = 'Potability', y = 'ph',data =dataset)
plt.show()

for i in dataset['Hardness']:
    assert isinstance(i,float) and i > 0
sns.distplot( dataset['Hardness'])
plt.show()
sns.boxplot(x = 'Potability', y = 'Hardness',data =dataset)
plt.show()

for i in dataset['Solids']:
    assert isinstance(i,float) and i > 0
sns.distplot( dataset['Solids'])
plt.show()
sns.boxplot(x = 'Potability', y = 'Solids',data =dataset)
plt.show()

for i in dataset['Chloramines']:
    assert isinstance(i,float) and i > 0
sns.distplot( dataset['Chloramines'])
plt.show()
sns.boxplot(x = 'Potability', y = 'Chloramines',data =dataset)
plt.show()

for i in dataset['Sulfate']:
    assert isinstance(i,float) and i > 0
sns.distplot( dataset['Sulfate'])
plt.show()
sns.boxplot(x = 'Potability', y = 'Sulfate',data =dataset)
plt.show()

for i in dataset['Conductivity']:
    assert isinstance(i,float) and i > 0
sns.distplot( dataset['Conductivity'])
plt.show()
sns.boxplot(x = 'Potability', y = 'Conductivity',data =dataset)
plt.show()

for i in dataset['Organic_carbon']:
    assert isinstance(i,float) and i > 0
sns.distplot( dataset['Organic_carbon'])
plt.show()
sns.boxplot(x = 'Potability', y = 'Organic_carbon',data =dataset)
plt.show()

for i in dataset['Trihalomethanes']:
    assert isinstance(i,float) and i > 0
sns.distplot( dataset['Trihalomethanes'])
plt.show()
sns.boxplot(x = 'Potability', y = 'Trihalomethanes',data =dataset)
plt.show()

for i in dataset['Turbidity']:
    assert isinstance(i,float) and i > 0
sns.distplot( dataset['Turbidity'])
plt.show()
sns.boxplot(x = 'Potability', y = 'Turbidity',data =dataset)
plt.show()

for i in dataset['Potability']:
    assert i in [0,1]
print(dataset.groupby('Potability').size())

sns.pairplot(dataset, hue = "Potability")
plt.show()

# all data has correct values 
# can move onto variable selection

# load libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from pandas import DataFrame

x = dataset.drop('Potability',axis = 1)
y = dataset['Potability']

# standardise numerical data
names = x.columns
st_x= StandardScaler()    
x= st_x.fit_transform(x)  
X = DataFrame(x,columns = names)

# evaluate performance of variables
model = RandomForestClassifier()
model.fit(X,y)
plt.bar(X.columns,model.feature_importances_)
plt.xticks(rotation=90)
plt.show()
variable_importance = [(X.columns[list(model.feature_importances_).index(i)],i) for i in 
sorted(model.feature_importances_, reverse = True)]

# all variables appear to be important features

no_feat = []
data = []
for i in range(len(variable_importance)):
    data.append(variable_importance[i][0])
    X_train, X_test, Y_train, Y_test = train_test_split(X[data],y,test_size=0.2, random_state=1)
    model = GaussianNB()
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    no_feat.append((i+1,cv_results.mean())) 

# no significant drop off in accuracy when adding variables
# we will keep all variables in the moddelling

# Split-out validation dataset
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# Spot Check Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()  

# cart is clearly strongest model, suggests random forest will be strong

# try ensemble models
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

# build more complex algorithms
models2 = []
models2.append(("BC",BaggingClassifier()))
models2.append(("RF",RandomForestClassifier()))
models2.append(("ETC",ExtraTreesClassifier()))
models2.append(("ABC",AdaBoostClassifier()))
models2.append(("GBC",GradientBoostingClassifier()))
# evaluate each complex model in turn
results2 = []
names2 = []
for name, model in models2:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results2.append(cv_results)
	names2.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
plt.boxplot(results2, labels=names2)
plt.title('Algorithm Comparison')
plt.show() 

# GBC and RF are the strongest models
# shall optimise both and pick strongest 

# optimise RF
from sklearn.model_selection import GridSearchCV

rs = {'bootstrap': [True, False],
 'max_features': ['auto', 'sqrt','log2'],
 'criterion' : ['entropy', 'gini'],
 'n_estimators': [100,300,500]}
grid = GridSearchCV(estimator = RandomForestClassifier(), param_grid = rs,cv =10)
grid.fit(X_train,Y_train)
grid.best_params_

kfold = StratifiedKFold(n_splits=9, random_state=1, shuffle=True)
cv_results = cross_val_score(RandomForestClassifier(bootstrap = True, 
            max_features = 'sqrt', criterion = 'gini', n_estimators = 300),
                             X_train, Y_train, cv=kfold, 
                             scoring='accuracy')
print("rf optimised", " ", cv_results.mean()) # 0.7927369538723866

# optimise GBC
rs = {"n_estimators":[500],
    "max_depth":[5,7,9],
    "learning_rate":[0.1,1,5,10],
    "loss" : ['deviance','exponential']}
grid = GridSearchCV(estimator = GradientBoostingClassifier(), param_grid = rs,cv =10)
grid.fit(X_train,Y_train)
grid.best_params_

kfold = StratifiedKFold(n_splits=9, random_state=1, shuffle=True)
cv_results = cross_val_score(GradientBoostingClassifier(learning_rate = 1,
                                                        loss = 'exponential',
                                                        max_depth = 7,
                                                        n_estimators = 500),
                             X_train, Y_train, cv=kfold, 
                             scoring='accuracy')
print("sgb optimised", " ", cv_results.mean())

# RF appears to be our strongest model so will now validate it 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


model = RandomForestClassifier(bootstrap = True, max_features = 'sqrt', 
                               criterion = 'gini', n_estimators = 300)
model.fit(X_train,Y_train)
predictions = model.predict(X_validation)

print(accuracy_score(Y_validation, predictions)) # 0.8033536585365854
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
