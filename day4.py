from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# load the data
data = load_iris()
X = data.data
y = data.target

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#train a model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

#Test it
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy ", accuracy)
print(data.feature_names)
print(data.target_names)