# Machine Learning - Mean Median Mode
# Mean
import numpy
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
x = numpy.mean(speed)
print(x)
print()

# Median
x = numpy.median(speed)
print(x)
print()

# Mode
from scipy import stats  
x = stats.mode(speed)
print(x)
print()

# Machine Learning - Standard Deviation
x = numpy.std(speed)
print(x)
print()

# Variance
x = numpy.var(speed)
print(x)
print()

# Machine Learning - Percentiles
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
x = numpy.percentile(ages, 75)
print(x)
print()

x = numpy.percentile(ages, 95)
print(x)
print()

# Machine Learning - Data Distribution
x = numpy.random.uniform(0.0, 100.0, 50000)
print(x)
print()

# Histogram
import matplotlib.pyplot as plt
x = numpy.random.uniform(0.0, 100.0, 50000)
# plt.hist(x, 500)
# plt.show()

# Machine Learning - Normal Data Distribution
x = numpy.random.normal(15.0, 1.0, 1000000)
print(x)
# plt.hist(x, 1000)
# plt.show()
print()

# Machine Learning - Scatter Plot
x = [5,7,8,7,2,17,2,9,4,11,12,9,6,15,20]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86,92,100]
# plt.scatter(x,y)
# plt.show()

# Random Data Distributions
x = numpy.random.normal(50.0, 10.0, 10000)
y = numpy.random.normal(100.0, 20.0, 10000)
# plt.scatter(x, y)
# plt.show()

# Machine Learning - Linear Regression
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

# plt.scatter(x, y)
# plt.plot(x, mymodel)
# plt.show()

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(y, x)


mymodel = list(map(myfunc, y))

# plt.scatter(y, x)
# plt.plot(y, mymodel)
# plt.show()

# R for Relationship
print()
print(r)
print()
print(p)
print()
print(std_err)
print()

# Predict Future Values
speed = myfunc(100)
print(speed)

# Bad Fit?
x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

# plt.scatter(x, y)
# plt.plot(x, mymodel)
# plt.show()
print()
print(r)
print()

# Machine Learning - Polynomial Regression
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
myline = numpy.linspace(1, 22, 100)

# plt.scatter(x, y)
# plt.plot(myline, mymodel(myline))
# plt.show()
print()

# R-Squared
from sklearn.metrics import r2_score
print(r2_score(y, mymodel(x)))
print()

# Predict Future Values
speed = mymodel(17)
print(speed)
print()

# Bad Fit?
x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
myline = numpy.linspace(2, 95, 100)
# plt.scatter(x, y)
# plt.plot(myline, mymodel(myline))
# plt.show()
print(r2_score(y, mymodel(x)))
print()

# Machine Learning - Multiple Regression
import pandas
from sklearn import linear_model
df = pandas.read_csv("cars.csv")
X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])
print(predictedCO2)
print()


# Coefficient
print(regr.coef_)
print()

# Machine Learning - Train/Test
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x
# plt.scatter(x, y)
# plt.show()

# Split Into Train/Test
train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]
# plt.scatter(train_x, train_y)
# plt.show()

# Display the Testing Set
# plt.scatter(test_x, test_y)
# plt.show()

# Fit the Data Set

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
myline = numpy.linspace(0, 6, 100)
# plt.scatter(train_x, train_y)
# plt.plot(myline, mymodel(myline))
# plt.show()

# R2
r2 = r2_score(test_y, mymodel(test_x))
print(r2)
print()

# Predict Values
print(mymodel(5))

# Machine Learning - Decision Tree
import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.image as pltimg
df = pandas.read_csv("shows.csv")
print(df)
print()

d = {'UK' : 0,'USA' : 1, 'N' : 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES' : 1, 'NO' : 0}
df['Go'] = df['Go'].map(d)
print(df)
print()

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

print(X)
print(y)
print()

# Create a Decision Tree, save it as an image, and show the image:
import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

df = pandas.read_csv("shows.csv")

d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()
