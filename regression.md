# Simple Ordinary Least Squares Linear Regression

* **Problem Statement**

We want to predict **Ice Cream Revenue** based on **Temperature**.

* **Independent variable (X):** Temperature
* **Dependent variable (Y):** Revenue
* **Type of problem:** Supervised Learning → Regression
* **Model:** Simple Linear Regression
* **Training data:** First 50 observations
* **Testing data:** Next 50 observations

Our objective is to find the **best straight line** that represents the relationship between temperature and revenue.

---

# 1. What is Simple Linear Regression?

Simple Linear Regression is a supervised machine learning algorithm used to predict a **continuous numerical value** using **one independent variable**.

The model assumes that the relationship between `X` and `Y` can approximately be represented by a straight line.

The general equation is:

$$
\hat{Y} = b_0 + b_1X
$$

Where:

* `X` → Input / independent variable
* `Ŷ` → Predicted output
* `b₀` → Intercept
* `b₁` → Slope

For our problem:

$$
\text{Predicted Revenue} = b_0 + b_1(\text{Temperature})
$$

---

# 2. What are we trying to find?

The main task is to find the two parameters:

* **Slope (`b₁`)**
* **Intercept (`b₀`)**

Once we find them, we have our regression equation.

For example, suppose we obtain:

$$
b_0 = -50
$$

$$
b_1 = 20
$$

Then our model becomes:

$$
\hat{Y} = -50 + 20X
$$

For a temperature of `30°C`:

$$
\hat{Y} = -50 + 20(30)
$$

$$
\hat{Y} = 550
$$

So the model predicts a revenue of **550**.

---

# 3. What does "Ordinary Least Squares" mean?

The model should not simply draw **any** line through the data.

There are many possible lines:

```text
Revenue
   |
   |       •
   |     •
   |   •
   | •
   |________________ Temperature
```

We want the line that is **closest to all the observed data points**.

For every observation, there is a difference between:

* **Actual value**
* **Predicted value**

This difference is called the **residual** or **error**.

$$
e_i = y_i - \hat{y}_i
$$

Where:

* `yᵢ` → Actual revenue
* `ŷᵢ` → Predicted revenue
* `eᵢ` → Residual

---

# 4. Why do we square the errors?

Suppose we have three errors:

```text
+10
-20
+5
```

If we simply add them:

$$
10 - 20 + 5 = -5
$$

Positive and negative errors can cancel each other.

Instead, we square every error:

$$
10^2 + (-20)^2 + 5^2
$$

$$
= 100 + 400 + 25
$$

$$
= 525
$$

Therefore, Ordinary Least Squares minimizes the **Sum of Squared Errors (SSE)**.

$$
SSE = \sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

The best regression line is the line that produces the **minimum SSE**.

---

# 5. How do we calculate the slope?

For simple linear regression, the slope is calculated using:

$$
b_1 =
\frac{\sum (x_i-\bar{x})(y_i-\bar{y})}
{\sum (x_i-\bar{x})^2}
$$

Where:

* `xᵢ` → Individual temperature
* `yᵢ` → Individual revenue
* `x̄` → Mean temperature
* `ȳ` → Mean revenue

The numerator measures how **X and Y vary together**.

The denominator measures how much **X varies around its mean**.

---

# 6. How do we calculate the intercept?

Once the slope is known:

$$
b_0 = \bar{y} - b_1\bar{x}
$$

Therefore, the complete process is:

```text
Calculate mean X
       ↓
Calculate mean Y
       ↓
Calculate slope (b₁)
       ↓
Calculate intercept (b₀)
       ↓
Build regression equation
       ↓
Make predictions
```

---

# 7. Manual Implementation Using a User-Defined Function

Instead of using `sklearn`, let's implement Simple Linear Regression ourselves.

This helps us understand what happens **inside the algorithm**.

### Step 1 — Calculate the slope

```python
def calculate_slope(X, Y):
    
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    
    numerator = np.sum((X - x_mean) * (Y - y_mean))
    denominator = np.sum((X - x_mean) ** 2)
    
    slope = numerator / denominator
    
    return slope
```

### Step 2 — Calculate the intercept

```python
def calculate_intercept(X, Y, slope):
    
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    
    intercept = y_mean - slope * x_mean
    
    return intercept
```

### Step 3 — Create the complete model

```python
def train_linear_regression(X, Y):
    
    slope = calculate_slope(X, Y)
    intercept = calculate_intercept(X, Y, slope)
    
    return slope, intercept
```

---

# 8. Train the Model

We already have:

```python
X_train = training_data['Temperature']
Y_train = training_data['Revenue']
```

Now train our manually implemented model:

```python
slope, intercept = train_linear_regression(X_train, Y_train)

print("Slope:", slope)
print("Intercept:", intercept)
```

Our regression equation is:

```python
print(f"Revenue = {intercept:.2f} + {slope:.2f} × Temperature")
```

So conceptually:

$$
\boxed{\hat{Y}=b_0+b_1X}
$$

---

# 9. Understanding the Slope

The slope has an important interpretation.

If:

```text
slope = 20
```

then:

> For every **1°C increase in temperature**, the model predicts an average increase of approximately **20 units of revenue**.

This makes the model much more interpretable than treating it as a black box.

---

# 10. Create a Prediction Function

Now that we have `slope` and `intercept`, we can predict revenue.

```python
def predict(X, slope, intercept):
    return intercept + slope * X
```

Test it:

```python
temperature = 30

predicted_revenue = predict(
    temperature,
    slope,
    intercept
)

print("Predicted Revenue:", predicted_revenue)
```

The model is answering:

> **"Based on the relationship learned from the training data, what revenue should we expect at 30°C?"**

---

# 11. Predict for the Complete Test Dataset

We haven't used the testing data to train the model.

That's important.

```python
X_test = testing_data['Temperature']
Y_test = testing_data['Revenue']
```

Now generate predictions:

```python
Y_pred = predict(X_test, slope, intercept)

print(Y_pred)
```

We now have:

```text
Actual Revenue       Predicted Revenue
     ↓                       ↓
    Y_test                  Y_pred
```

---

# 12. Visualize the Regression Line

First, plot the training data.

```python
plt.scatter(
    X_train,
    Y_train,
    color='green',
    label='Training Data'
)

plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.title('Temperature vs Revenue')

plt.grid(True)
plt.legend()
plt.show()
```

Now calculate the predicted values for the training data:

```python
Y_train_pred = predict(
    X_train,
    slope,
    intercept
)
```

Plot the regression line:

```python
plt.scatter(
    X_train,
    Y_train,
    color='green',
    label='Actual Data'
)

plt.plot(
    X_train,
    Y_train_pred,
    color='red',
    label='Regression Line'
)

plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.title('Simple Linear Regression')

plt.grid(True)
plt.legend()
plt.show()
```

The visualization now shows:

```text
Revenue
  |
  |                  •
  |             •  /
  |          •   /
  |       •    /
  |    •     /
  | •       /
  |________/________________ Temperature
```

The red line represents the relationship learned by our model.

---

# 13. Understanding Residuals

A residual is:

$$
Residual = Actual - Predicted
$$

We can calculate it using:

```python
residuals = Y_train - Y_train_pred
```

Visualizing residuals can help us understand whether the model is making systematic errors.

```python
plt.scatter(
    X_train,
    residuals
)

plt.axhline(
    y=0,
    color='red',
    linestyle='--'
)

plt.xlabel('Temperature')
plt.ylabel('Residual')
plt.title('Residual Plot')

plt.grid(True)
plt.show()
```

Ideally, residuals should be scattered around zero without a clear pattern.

---

# 14. Evaluate the Model

A simple metric for regression is **Mean Squared Error (MSE)**.

$$
MSE =
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$

We can implement it ourselves:

```python
def mean_squared_error(Y_actual, Y_predicted):
    
    errors = Y_actual - Y_predicted
    
    squared_errors = errors ** 2
    
    mse = np.mean(squared_errors)
    
    return mse
```

Calculate the test MSE:

```python
mse = mean_squared_error(
    Y_test,
    Y_pred
)

print("Test MSE:", mse)
```

---

# 15. RMSE — Easier to Interpret

We can take the square root of MSE:

$$
RMSE = \sqrt{MSE}
$$

```python
rmse = np.sqrt(mse)

print("Test RMSE:", rmse)
```

If:

```text
RMSE = 50
```

we can roughly interpret it as:

> The model's predictions typically have an error magnitude of around **50 revenue units**.

Unlike MSE, RMSE has the **same unit as the target variable**.

---

# 16. Alternative: Using Scikit-Learn

Now that we understand what happens mathematically, we can use the industry-standard implementation.

Scikit-learn provides:

```python
LinearRegression
```

Import it:

```python
from sklearn.linear_model import LinearRegression
```

---

# 17. Prepare X and Y

There is one important difference.

Our `X_train` is currently a Pandas Series:

```python
X_train = training_data['Temperature']
```

Scikit-learn expects the feature matrix to have **2 dimensions**.

Therefore:

```python
X_train = training_data[['Temperature']]
Y_train = training_data['Revenue']
```

Notice the difference:

```python
training_data['Temperature']
```

returns a Series.

```python
training_data[['Temperature']]
```

returns a DataFrame with one feature.

---

# 18. Create the Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

---

# 19. Train the Model

```python
model.fit(X_train, Y_train)
```

The `fit()` method learns:

* Slope
* Intercept

from the training data.

---

# 20. Get the Learned Parameters

```python
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
```

Compare these values with our manually calculated values:

```python
print("Manual Slope:", slope)
print("Sklearn Slope:", model.coef_[0])

print("Manual Intercept:", intercept)
print("Sklearn Intercept:", model.intercept_)
```

They should be essentially the same, apart from tiny floating-point differences.

This is an excellent point to emphasize to students:

> **Scikit-learn isn't performing a completely different algorithm. We manually implemented the same Ordinary Least Squares idea that `LinearRegression` provides conveniently.**

---

# 21. Predict Using Scikit-Learn

Prepare the test data:

```python
X_test = testing_data[['Temperature']]
Y_test = testing_data['Revenue']
```

Predict:

```python
Y_pred_sklearn = model.predict(X_test)
```

For example:

```python
temperature = [[30]]

prediction = model.predict(temperature)

print("Predicted Revenue:", prediction[0])
```

---

# 22. Evaluate the Scikit-Learn Model

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(
    Y_test,
    Y_pred_sklearn
)

rmse = np.sqrt(mse)

print("Test MSE:", mse)
print("Test RMSE:", rmse)
```

---

# 23. Plot Actual vs Regression Line

```python
plt.scatter(
    X_train,
    Y_train,
    color='green',
    label='Training Data'
)

plt.plot(
    X_train,
    model.predict(X_train),
    color='red',
    label='Regression Line'
)

plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.title('Temperature vs Revenue - Linear Regression')

plt.grid(True)
plt.legend()
plt.show()
```

---

# 24. Manual vs Scikit-Learn

| Step                | Manual Implementation   | Scikit-Learn           |
| ------------------- | ----------------------- | ---------------------- |
| Calculate slope     | `calculate_slope()`     | `model.fit()`          |
| Calculate intercept | `calculate_intercept()` | `model.fit()`          |
| Create model        | Our UDF                 | `LinearRegression()`   |
| Prediction          | `predict()`             | `model.predict()`      |
| MSE                 | Our UDF                 | `mean_squared_error()` |
| RMSE                | `np.sqrt(MSE)`          | `np.sqrt(MSE)`         |

The **concept and mathematics are the same**.

The major difference is that scikit-learn gives us a reliable, optimized implementation without requiring us to manually calculate every parameter.

---

# 25. Complete Manual Implementation

For students who want to see the complete algorithm together:

```python
import numpy as np

def train_linear_regression(X, Y):
    
    # Calculate means
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    
    # Calculate slope
    numerator = np.sum(
        (X - x_mean) * (Y - y_mean)
    )
    
    denominator = np.sum(
        (X - x_mean) ** 2
    )
    
    slope = numerator / denominator
    
    # Calculate intercept
    intercept = y_mean - slope * x_mean
    
    return slope, intercept


def predict(X, slope, intercept):
    
    return intercept + slope * X


def mean_squared_error(Y_actual, Y_predicted):
    
    return np.mean(
        (Y_actual - Y_predicted) ** 2
    )


# Train
slope, intercept = train_linear_regression(
    X_train,
    Y_train
)

# Predict
Y_pred = predict(
    X_test,
    slope,
    intercept
)

# Evaluate
mse = mean_squared_error(
    Y_test,
    Y_pred
)

rmse = np.sqrt(mse)

print("Slope:", slope)
print("Intercept:", intercept)
print("MSE:", mse)
print("RMSE:", rmse)
```

---

# 26. Complete Scikit-Learn Implementation

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Prepare data
X_train = training_data[['Temperature']]
Y_train = training_data['Revenue']

X_test = testing_data[['Temperature']]
Y_test = testing_data['Revenue']

# Create model
model = LinearRegression()

# Train
model.fit(X_train, Y_train)

# Parameters
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Prediction
Y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)
```

---
