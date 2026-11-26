import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

hours=np.array([2,3,4,5,6,7,8]).reshape(-1,1)
result=np.array([0,0,0,0,0,1,1])

model=LogisticRegression()
model.fit(hours,result)

prediction =model.predict([[7.5]])
print("Predicted Result(1=Pass,0=Fail):",prediction[0])

accuracy=model.score(hours,result)
print("Model Accuracy:",accuracy)

plt.scatter(hours,result)
plt.plot(hours,model.predict(hours))
plt.xlabel("Study Hours")
plt.ylabel("Result")
plt.title("Student Pass Prediction using Logistic Regression")
plt.show()