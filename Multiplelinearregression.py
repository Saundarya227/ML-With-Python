import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X=np.array([[2,60,75],
           [4,70,75],
           [6,80,80],
           [8,85,85],
           [10,90,95]])
Y=np.array([60,65,75,80,90])

model=LinearRegression()
model.fit(X,Y)

prediction=model.predict([[7,75,78]])
print("Final Marks:",prediction[0])

print("Intercept:",model.intercept_)
print("Coefficients:",model.coef_)
print("R^2:",model.score(X,Y))

plt.scatter(X[:,0],Y)
plt.plot(X,model.predict(X))
plt.xlabel("Student Performance")
plt.ylabel("Final Marks")
plt.title("Multiple Linear Regression ")
plt.show()
