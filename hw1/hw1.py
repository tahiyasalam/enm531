import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

''' EXERCISE 3 '''
num_samples = 1000

# p(x,y)
mean_x_y = [0, 2]
covariance_x_y = [[0.3, -1], [-1, 5]]
x_y = np.random.multivariate_normal(mean_x_y, covariance_x_y, num_samples)

xmin = x_y[:, 0].min()
xmax = x_y[:, 0].max()
ymin = x_y[:, 1].min()
ymax = x_y[:, 1].max()

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])

kernel = stats.gaussian_kde(x_y.T)
Z = np.reshape(kernel(positions).T, X.shape)

fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
ax.imshow(X, Y, x_y.pdf(positions),cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])

ax.plot(x_y[:, 0], x_y[:, 1], 'k.', markersize=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Distribution of p(x,y)')
plt.show()

# p(x)
mean_x = [0]
covariance_x = [[0.3]]
x = np.random.multivariate_normal(mean_x, covariance_x, num_samples)

xmin = x[:].min()
xmax = x[:].max()
X = np.linspace(xmin, xmax, 100)

mu = 0
variance = covariance_x[0][0]
sigma = np.sqrt(variance)
x_range = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

fig, ax = plt.subplots()
ax.plot(x_range,mlab.normpdf(x_range, mu, sigma), 'g')

kernel = stats.gaussian_kde(x.T)
ax.plot(X, kernel(X), 'r')
ax.set_title('Distribution of p(x)')
ax.legend(['True distribution', 'Estimated distribution'])
plt.show()

# p(x | y = -1)
mean_x_given_y = [0.6]
covariance_x_given_y = [[0.9]]
x_given_y = np.random.multivariate_normal(mean_x_given_y, covariance_x_given_y, num_samples)

xmin = x_given_y[:].min()
xmax = x_given_y[:].max()
X = np.linspace(xmin, xmax, 100)

mu = mean_x_given_y[0]
variance = covariance_x_given_y[0][0]
sigma = np.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
fig, ax = plt.subplots()
ax.plot(x,mlab.normpdf(x, mu, sigma), 'g')

kernel = stats.gaussian_kde(x_given_y.T)
ax.plot(X, kernel(X), 'b')
ax.legend(['True distribution', 'Estimated distribution'])
ax.set_title('Distribution of p(x|y=-1)')
plt.show()


''' EXERCISE 4 '''
num_samples = 5000

mean_x1_x2 = [0, 0]
covariance_x1_x2 = [[1, 0.9], [0.9, 1]]
x1_x2 = np.random.multivariate_normal(mean_x1_x2, covariance_x1_x2, num_samples)

x1 = x1_x2[:, 0]
x2 = x1_x2[:, 1]
a = 1.15
b = 0.5

y1 = a*x1
y2 = x2/a + b*(np.square(x2) + a**2)

y1min = y1.min()
y1max = y1.max()
y2min = y2.min() - 1
y2max = y2.max()

Y1, Y2 = np.mgrid[y1min:y1max:100j, y2min:y2max:100j]
positions = np.vstack([Y1.ravel(), Y2.ravel()])

y1_y2 = np.vstack([y1.T, y2.T]).T
kernel = stats.gaussian_kde(y1_y2.T)
Z = np.reshape(kernel(positions).T, Y1.shape)

fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[y1min, y1max, y2min, y2max])

ax.plot(y1_y2[:, 0], y1_y2[:, 1], 'k.', markersize=2)
ax.set_xlim([y1min, y1max])
ax.set_ylim([y2min, y2max])
ax.set_xlabel('y1')
ax.set_ylabel('y2')
ax.set_title('Distribution of Y')
plt.show()

empirical_mean = np.average(y1), np.average(y2)
print(empirical_mean)