import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay ,classification_report

data={'hours_study': [2,4,6,1,5,7,3,8,9,10],
    'attendance': [60,70,80,55,75,90,65,95,98,99],
    'passed':     [0,1,1,0,1,1,0,1,1,1]
}
df=pd.DataFrame(data)
#Visualizing  the data
plt.scatter(df["hours_study"],df["passed"])
plt.xlabel("hours_studied")
plt.ylabel("passed")
plt.title("You get what you deserve not what you want")
plt.show()
#Heatmaps- correlations
sns.heatmap(df.corr(),annot=True)
plt.title("Correlation Heatmap")
plt.show()
#Splitting the Data
X=df[["hours_study","attendance"]]
y=df["passed"]
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3,random_state=42)
#Train the Model
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
#Evaluaating
ConfusionMatrixDisplay.from_estimator(model,X_test,y_test)
plt.title("Confusion Matrix")
plt.show()
#Make predictions
y_predict=model.predict(X_test)
print(classification_report(y_test, y_predict))
# Feature importance
plt.bar(['hours_study','attendance'], model.feature_importances_)
plt.title("Feature Importance")
plt.show()
