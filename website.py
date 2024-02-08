import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
# Additional imports for performance metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
# Set the 'fivethirtyeight' style
plt.style.use('fivethirtyeight')


# Assuming 'reg' is your trained model
# Modify this based on your actual feature names and model
# For this example, let's assume 'reg' is a trained model using features like 'dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year'

df = pd.read_csv(r"C:\Users\PRO\Downloads\archive\PJME_hourly.csv")
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)


# Function to make predictions
train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']
def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)
train = create_features(train)
test = create_features(test)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'PJME_MW'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)
def make_predictions(start_date, end_date):
    new_data = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
    new_data[['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']] = new_data.index.to_series().apply(lambda x: pd.Series([x.dayofyear, x.hour, x.dayofweek, x.quarter, x.month, x.year]))
    new_data['prediction'] = reg.predict(new_data[['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']])
    return new_data
def make_prediction(prediction_datetime):
    # Create a DataFrame with a single row for the prediction date and time
    prediction_data = pd.DataFrame(index=[prediction_datetime])

    # Extract relevant features from the prediction date and time
    prediction_data[['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']] = prediction_data.index.to_series().apply(lambda x: pd.Series([x.dayofyear, x.hour, x.dayofweek, x.quarter, x.month, x.year]))

    # Make prediction for the specific day and time
    predicted_power_consumption = reg.predict(prediction_data[['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']])

    # Return the predicted power consumption value
    return predicted_power_consumption[0]


# Function to make a single prediction for a specific date
def make_single_prediction(prediction_date):
    # Create a DataFrame with a single row for the prediction date
    prediction_data = pd.DataFrame(index=[prediction_date])

    # Extract relevant features from the prediction date
    prediction_data[['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']] = prediction_data.index.to_series().apply(lambda x: pd.Series([x.dayofyear, x.hour, x.dayofweek, x.quarter, x.month, x.year]))

    # Make prediction for the specific day
    predicted_power_consumption = reg.predict(prediction_data[['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']])

    # Return the predicted power consumption value
    return predicted_power_consumption[0]

def make_predictions_for_analysis(start_date, end_date):
    new_data = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
    new_data[['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']] = new_data.index.to_series().apply(lambda x: pd.Series([x.dayofyear, x.hour, x.dayofweek, x.quarter, x.month, x.year]))
    new_data['prediction'] = reg.predict(new_data[['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']])
    return new_data




# Streamlit app

def main():
    st.title("Power Consumption Prediction App")

    # User input for date range
    start_date = st.date_input("Select start date", pd.to_datetime('2019-01-01'))
    end_date = st.date_input("Select end date", pd.to_datetime('2019-01-01'))

    # Make predictions
    predictions = make_predictions(start_date, end_date)

    # Display predictions table
    st.subheader("Power Consumption Predictions")
    st.write(predictions)

    # Find the date of the most power consumed day in a week
    most_power_consumed_day = predictions.groupby(predictions.index.isocalendar().week)['prediction'].idxmax()

    # Display most power consumed day for each week
    st.subheader("Most Power Consumed Day in Each Week")
    for week, date in most_power_consumed_day.items():
        st.write(f"Week {week}: {date.date()}")


        # Plotting with Matplotlib
    fig, ax = plt.subplots(figsize=(15, 5))
    predictions['prediction'].plot(ax=ax, title='Power Consumption Prediction for the Entire Date Range')
    ax.set_xlabel('Date')
    ax.set_ylabel('Power Consumption')

    # Display the Matplotlib plot in Streamlit
    st.pyplot(fig)



    st.subheader("Power Consumption on specific time of a day")

# User input for the prediction date and time
    prediction_date_str = st.date_input("Enter the prediction date:", pd.to_datetime("today"))
    prediction_time_str = st.time_input("Enter the prediction time:")

# Convert input strings to datetime objects
    prediction_datetime = pd.to_datetime(f"{prediction_date_str} {prediction_time_str}")

# Make prediction
    predicted_power_consumption = make_prediction(prediction_datetime)

# Display the predicted power consumption value in the Streamlit app
    st.write(f"Predicted Power Consumption on {prediction_datetime}= {predicted_power_consumption}")
 # Most Power Consumed Quarter of a Year
    st.subheader("Most Power Consumed Quarter of a Year")

    # User input for the date range to find the most power consumed quarter
    quarter_start_date = st.date_input("Select start date for quarter analysis", pd.to_datetime('2019-01-01'))
    quarter_end_date = st.date_input("Select end date for quarter analysis", pd.to_datetime('2019-01-01'))

    # Convert the date input to datetime objects
    quarter_start_date = pd.to_datetime(quarter_start_date)
    quarter_end_date = pd.to_datetime(quarter_end_date)

  # Make predictions for the specified date range using a new DataFrame
    quarter_predictions_df = make_predictions_for_analysis(quarter_start_date, quarter_end_date)

    # Identify the quarter with the highest power consumption in the new DataFrame
    most_power_consumed_quarter = quarter_predictions_df.groupby(quarter_predictions_df.index.quarter)['prediction'].sum().idxmax()

    # Display the most power consumed quarter for the specified date range
    st.write(f"Most Power Consumed Quarter from {quarter_start_date} to {quarter_end_date}: Quarter {most_power_consumed_quarter}")






if __name__ == "__main__":
    main()
