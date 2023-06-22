import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# بارگیری مجموعه داده
url = 'rr.py'
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
data = pd.read_csv(url, names=names)

# بررسی ابعاد داده‌ها
print("Shape of dataset:", data.shape)

# بررسی تعداد مقادیر null در هر ستون
print("Number of null values in each column:\n", data.isnull().sum())

# جدا کردن ویژگی‌ها و برچسب‌ها
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# تقسیم داده‌ها به داده‌های آموزشی و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# انجام مقیاس‌بندی داده‌ها
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ایجاد مدل درخت تصمیم
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# پیش‌بینی برچسب‌ها با استفاده از داده‌های تست
y_pred = classifier.predict(X_test)

# ارزیابی عملکرد مدل
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print("Confusion matrix:\n", cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
