from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
########################     ########################################
# attempt to create classifier that check if feature help to the training of the model or not
########################     ########################################
def feature_regression(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Perform Recursive Feature Elimination with Cross-Validation
    regr = LinearRegression()
    rfe = RFECV(estimator=regr, step=1, cv=5)
    rfe.fit(X_train_scaled, y_train)

    # Select the most important features
    X_train_selected = rfe.transform(X_train_scaled)
    X_test_selected = rfe.transform(X_test_scaled)

    # Train and evaluate the regressor with the selected features
    regr.fit(X_train_selected, y_train)
    y_pred = regr.predict(X_test_selected)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Display the selected features
    selected_features = [i for i, mask in enumerate(rfe.support_) if mask]
    print(f"Selected features: {selected_features}")