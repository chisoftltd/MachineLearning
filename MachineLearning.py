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
print( )