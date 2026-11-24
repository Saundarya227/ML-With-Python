import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

hours=np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
marks=np.array([15,30,45,55,65,75,85,90])

model=LinearRegression()
model.fit(hours,marks)

predicted_marks=model.predict([[7.8]])
print("Predicted marks for 7.8 hours of study:",predicted_marks[0])

print("Slope(m):",model.coef_[0])
print("Intercept(c):",model.intercept_)

plt.scatter(hours,marks)
plt.plot(hours,model.predict(hours))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study hours vs Marks using linear regression")
plt.show()