import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("rosamanhattandata.csv")

data.sort_values(["Registration Date"], axis=0, ascending=[True], inplace=False)

# print(data["Registration Date"])
x = data["Registration Date"]
x = x.array.shape(-1,1)

x = x.apply(pd.to_datetime)

y = data["Derived Carpet Area Rate"]

# plt.scatter(x,y)
# plt.show()

model = LinearRegression(fit_intercept=True).fit(x,y)

r_sq = model.score(x,y)
print(f"coefficient of determination: {r_sq}")
