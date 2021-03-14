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
plt.hist(x, 500)
plt.show()

# Machine Learning - Normal Data Distribution
x = numpy.random.normal(15.0, 1.0, 1000000)
print(x)
plt.hist(x, 1000)
plt.show()
print()

# Machine Learning - Scatter Plot
x = [5,7,8,7,2,17,2,9,4,11,12,9,6,15,20]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86,92,100]
plt.scatter(x,y)
plt.show()

# Random Data Distributions
x = numpy.random.normal(50.0, 10.0, 10000)
y = numpy.random.normal(100.0, 20.0, 10000)
plt.scatter(x, y)
plt.show()