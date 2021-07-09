
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

# all known values of ph in dataset are safe by who standards
# take mean for nas but prepare to remove column if variable is unimportant 
dataset['ph'].fillna(value=dataset['ph'].mean(), inplace=True)

# nearly a quatre of sulfates are missing 
# take average to replace nas
dataset['Sulfate'].fillna(value=dataset['Sulfate'].mean(), inplace=True)

for i in dataset['Trihalomethanes']:
    if i > 80:
        print('value greater than 80')
        break
    
# values in column surpass who safe levels so provides information
# take mean to replace nas
dataset['Trihalomethanes'].fillna(value=dataset['Trihalomethanes'].mean(), inplace=True)

# assess each variable has values in correct range and examine of the data
dataset.head(5)
dataset.describe()
dataset.columns

for i in dataset['Hardness']:
    assert isinstance(i,float) and i > 0
dataset['Hardness'].plot(kind='box')
plt.show()

for i in dataset['Solids']:
    assert isinstance(i,float) and i > 0
dataset['Solids'].plot(kind='box')
plt.show()

for i in dataset['Chloramines']:
    assert isinstance(i,float) and i > 0
dataset['Chloramines'].plot(kind='box')
plt.show()

for i in dataset['Sulfate']:
    assert isinstance(i,float) and i > 0
dataset['Sulfate'].plot(kind='box')
plt.show()

for i in dataset['Conductivity']:
    assert isinstance(i,float) and i > 0
dataset['Conductivity'].plot(kind='box')
plt.show()

for i in dataset['Organic_carbon']:
    assert isinstance(i,float) and i > 0
dataset['Organic_carbon'].plot(kind='box')
plt.show()

for i in dataset['Trihalomethanes']:
    assert isinstance(i,float) and i > 0
dataset['Trihalomethanes'].plot(kind='box')
plt.show()

for i in dataset['Turbidity']:
    assert isinstance(i,float) and i > 0
dataset['Turbidity'].plot(kind='box')
plt.show()

for i in dataset['Potability']:
    assert i in [0,1]
print(dataset.groupby('Potability').size())

sns.pairplot(dataset)
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

# all variables seem useful but we can check with a simple model
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
models.append(('SVC', SVC(gamma='auto')))
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

# svc appears to be best model

# try ensemble models
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier

# build more complex algorithms
models2 = []
models2.append(("BDT",BaggingClassifier()))
models2.append(("RF",RandomForestClassifier()))
models2.append(("ETC",ExtraTreesClassifier()))
models2.append(("ABC",AdaBoostClassifier()))
models2.append(("SGB",GradientBoostingClassifier()))
models2.append(("VOTE",VotingClassifier([models[0],models[1],models[2],models[4],models[5]])))
# evaluate each complex model in turn
results2 = []
names2 = []
for name, model in models2:
	kfold = StratifiedKFold(n_splits=9, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results2.append(cv_results)
	names2.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
plt.boxplot(results2, labels=names2)
plt.title('Algorithm Comparison')
plt.show() 

# svc still appears to be best model but rf is close
# will optimise both to find optimal model

# optimise svm
from sklearn.model_selection import GridSearchCV
rs = {"C" : [0.1, 1, 10, 100, 1000],
      "kernel" : ['linear', 'poly', 'rbf'],
      "gamma" :[0.1,1,10,100],
      "degree" : [2, 3, 4, 5, 6]}
grid = GridSearchCV(estimator = SVC(), param_grid = rs)
grid.fit(X_train,Y_train)
grid.best_params_
