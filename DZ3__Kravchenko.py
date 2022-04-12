#1. Импортируйте библиотеки pandas и numpy.
import numpy as np
import pandas as pd

#Загрузите "Boston House Prices dataset" из встроенных наборов данных библиотеки sklearn. 
from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()

#Создайте датафреймы X и y из этих данных.
data = boston["data"]
data.shape

feature_names = boston["feature_names"]
feature_names

target = boston["target"]
target[:10]

X = pd.DataFrame(data, columns=feature_names)
X.head()

X.info()

y = pd.DataFrame(target, columns=["price"])
y.info()

#Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test) 
#с помощью функции train_test_split так, чтобы размер тестовой выборки составлял 30% от 
#всех данных, при этом аргумент random_state должен быть равен 42. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Создайте модель линейной регрессии под названием lr с помощью класса 
#LinearRegression из модуля sklearn.linear_model. 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

#Обучите модель на тренировочных данных (используйте все признаки) и сделайте предсказание на тестовых.
lr.fit(X_train, y_train)


y_pred = lr.predict(X_test)
y_pred.shape

check_test = pd.DataFrame({
    "y_test": y_test["price"],
    "y_pred": y_pred.flatten(),
})

check_test.head(10)

#Вычислите R2 полученных предказаний с помощью r2_score из модуля sklearn.metrics.
check_test["error"] = check_test["y_pred"] - check_test["y_test"]
check_test.head()

from sklearn.metrics import r2_score
r2_score_1=r2_score(check_test["y_pred"], check_test["y_test"])
r2_score_1

#                                                 Задание 2.
# Создайте модель под названием model с помощью RandomForestRegressor из модуля sklearn.ensemble.
#Сделайте агрумент n_estimators равным 1000, max_depth должен быть равен 12 и random_state сделайте равным 42. 
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)

#Обучите модель на тренировочных данных аналогично тому, как вы обучали модель LinearRegression, 
#но при этом в метод fit вместо датафрейма y_train поставьте y_train.values[:, 0], чтобы получить 
#из датафрейма одномерный массив Numpy, так как для класса RandomForestRegressor в данном методе 
#для аргумента y предпочтительно применение массивов вместо датафрейма.
model.fit(X_train, y_train.values[:, 0])

#Сделайте предсказание на тестовых данных и посчитайте R2.
y_pred = model.predict(X_test)
y_pred.shape

check_test = pd.DataFrame({
    "y_test": y_test["price"],
    "y_pred": y_pred.flatten(),
})

check_test.head(10)

r2_score_2=r2_score(check_test["y_pred"], check_test["y_test"])
r2_score_2

#Сравните с результатом из предыдущего задания.
#Напишите в комментариях к коду, какая модель в данном случае работает лучше.
r2_score_1<r2_score_2

# модель RandomForestRegressor работает лучше, чем модель LinearRegression

#                                   Задание 3.
# Вызовите документацию для класса RandomForestRegressor,
?RandomForestRegressor

#найдите информацию об атрибуте feature_importances. 
#feature_importances_ : array of shape = [n_features]
#The feature importances (the higher, the more important the feature).
    

#С помощью этого атрибута найдите сумму всех показателей важности, 
print(model.feature_importances_)

model.feature_importances_.sum()

#установите, какие два признака показывают наибольшую важность.
max_value_idx1=model.feature_importances_.argmax()
max_value_idx1

max_value_idx2=0
max_value=model.feature_importances_[max_value_idx2]
for i in range(model.n_features_):
    if max_value<model.feature_importances_[i] and i!=max_value_idx1:
        max_value=model.feature_importances_[i]
        max_value_idx2=i
print(max_value_idx2)







