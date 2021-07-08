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
# provides no infomation so delete column
del dataset['ph']

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