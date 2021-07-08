<<<<<<< HEAD
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


