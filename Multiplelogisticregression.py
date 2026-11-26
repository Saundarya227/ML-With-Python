import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X=np.array([[2,50,60],
            [4,60,65],
            [6,70,75],
            [8,80,85],
            [10,90,95]
            ])
y=np.array([0,0,0,1,1])

model=LogisticRegression()
model.fit(X,y)

prediction=model.predict([[7,75,78]])
print("Predicted class:",prediction[0])

y_pred=model.predict(X)
print("Accuracy:",accuracy_score(y,y_pred))

plt.scatter(X[:,0],y)
plt.plot(X,model.predict(X))
plt.xlabel("Student Performance")
plt.ylabel("Fianl Result")
plt.title("Student Pass Prediction using Multiple Logistic Regression")
plt.show()