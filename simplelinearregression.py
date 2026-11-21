import numpy as np
import matplotlib.pyplot as plt

X=np.array([1,2,3,4,5])
y=np.array([2,4,5,4,5])

mean_x=np.mean(X)
mean_y=np.mean(y)

m=np.sum((X-mean_x)*(y-mean_y))/np.sum((X-mean_x)**2)
c=mean_y-m*mean_x

y_pred=m*X+c

plt.scatter(X,y)
plt.plot(X,y_pred)
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")
plt.title("Simple Linear Regression")
plt.show()
