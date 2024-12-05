import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the data (replace with your actual file path)
data = pd.read_csv(r"./Conversion_data_PC.csv")

# Ensure 'dateCreated' is parsed correctly, assuming format is 'DD/MM/YYYY'
def parse_date(date_string):
    formats = ['%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']
    for fmt in formats:
        try:
            return pd.to_datetime(date_string, format=fmt, errors='raise')
        except (ValueError, TypeError):
            continue
    return pd.NaT  # Return NaT if all parsing attempts fail

# Apply date parsing function
data['dateCreated'] = data['dateCreated'].apply(parse_date)

# Drop rows with invalid dates
data = data.dropna(subset=['dateCreated'])

# Apply initial filters:
filtered_data = data[
    (data['sessions'] >= 300) &  # Filter sessions greater than or equal to 300
    (data['quotes'] >= 80) &     # Filter quotes greater than or equal to 80
    (~data['hours'].between(1, 7))  # Filter out hours between 1 AM and 7 AM (inclusive)
]

# Create filters for other columns (e.g., hours, sessions, etc.)
selected_dates = st.sidebar.date_input(
    "Select dates",
    min_value=filtered_data['dateCreated'].min().date(),  # Convert to date for comparison
    max_value=filtered_data['dateCreated'].max().date(),  # Convert to date for comparison
    value=(filtered_data['dateCreated'].min().date(), filtered_data['dateCreated'].max().date()),  # Convert to date for comparison
    key="date_filter"
)

# Filter the data by selected dates (make sure we convert to date before comparison)
filtered_data = filtered_data[filtered_data['dateCreated'].dt.date.between(selected_dates[0], selected_dates[1])]



# Define the fixed hour range for the slider (8 AM to 12 AM)
hour_range = st.sidebar.slider(
    "Select hour range",
    min_value=8,  # Start from 8 AM
    max_value=24,  # End at midnight (24 hours format)
    value=(8, 24)  # Default selection
)

# Filter the data based on the selected hour range
filtered_data = filtered_data[
    (filtered_data['hours'] >= hour_range[0]) & (filtered_data['hours'] <= hour_range[1])
]


# Add sliders for sessions, quotes, and transactions
min_sessions, max_sessions = int(filtered_data['sessions'].min()), int(filtered_data['sessions'].max())
min_quotes, max_quotes = int(filtered_data['quotes'].min()), int(filtered_data['quotes'].max())
min_transactions, max_transactions = int(filtered_data['transactions'].min()), int(filtered_data['transactions'].max())

# Filters for sessions, quotes, and transactions
session_filter = st.sidebar.slider(
    "Select session range",
    min_value=min_sessions, max_value=max_sessions, value=(min_sessions, max_sessions)
)
quote_filter = st.sidebar.slider(
    "Select quote range",
    min_value=min_quotes, max_value=max_quotes, value=(min_quotes, max_quotes)
)
transaction_filter = st.sidebar.slider(
    "Select transaction range",
    min_value=min_transactions, max_value=max_transactions, value=(min_transactions, max_transactions)
)



# Add a checkbox in the sidebar to exclude weekends
exclude_weekends = st.sidebar.checkbox("Exclude Weekends", value=False)

# Add a column to identify weekends (Saturday = 5, Sunday = 6)
filtered_data['is_weekend'] = filtered_data['dateCreated'].dt.weekday.isin([5, 6])

# Apply the filter to exclude weekends if the checkbox is selected
if exclude_weekends:
    filtered_data = filtered_data[~filtered_data['is_weekend']]








# Apply the session filter
filtered_data_sessions = filtered_data[(
    filtered_data['sessions'] >= session_filter[0]) & (filtered_data['sessions'] <= session_filter[1])
]

# Apply the transaction filter globally to both sessions and quotes data
filtered_data_transactions_sessions = filtered_data_sessions[(
    filtered_data_sessions['transactions'] >= transaction_filter[0]) & (
    filtered_data_sessions['transactions'] <= transaction_filter[1])
]

# Show the data in a table with a slider to control how many rows are displayed
rows_to_show = st.slider("Select number of rows to display", 1, len(filtered_data), 5)

# Display the first "n" rows based on the slider value
st.dataframe(filtered_data_transactions_sessions.head(rows_to_show))  # Show filtered data in a table

# Create the polynomial regression model
poly = PolynomialFeatures(degree=2)  # Degree 2 for quadratic
model_sessions = LinearRegression()

# Prepare the data for regression (using 'sessions' as the independent variable)
X_sessions = filtered_data_transactions_sessions[['sessions']]
y_sessions = filtered_data_transactions_sessions['quoteConversion']

# Fit the model to the data
X_poly_sessions = poly.fit_transform(X_sessions)
model_sessions.fit(X_poly_sessions, y_sessions)

# Generate evenly spaced values of 'sessions' for smooth plotting
x_range_sessions = np.linspace(filtered_data_transactions_sessions['sessions'].min(), filtered_data_transactions_sessions['sessions'].max(), 100).reshape(-1, 1)
x_range_poly_sessions = poly.transform(x_range_sessions)

# Predict the corresponding 'quoteConversion' values
y_range_sessions_pred = model_sessions.predict(x_range_poly_sessions)

# Create scatter plot for Sessions vs Quote Conversion
fig1 = px.scatter(
    filtered_data_transactions_sessions,
    x='sessions',
    y='quoteConversion',
    color='hours',
    title="Sessions vs Quote Conversion"
)

# Add the quadratic regression line
fig1.add_scatter(
    x=x_range_sessions.flatten(),  # Use the new range of sessions
    y=y_range_sessions_pred,        # Predicted quoteConversion values
    mode='lines',
    name='Quadratic Fit',
    line=dict(color='red', dash='dash', width=3)  # Thicker line
)

# Update y-axis to show percentage with two decimals (e.g., 25.00%)
fig1.update_layout(
    yaxis_tickformat='.2%'  # Format percentage with 2 decimal places
)

# **Only apply quotes filter for the Quotes vs Quote Conversion graph**
# Filter data for Quotes vs Quote Conversion
filtered_data_quotes = filtered_data[filtered_data['quotes'] >= 80]
filtered_data_quotes = filtered_data_quotes[~filtered_data_quotes['hours'].isin([1, 2, 3, 4, 5, 6, 7])]  # Filter out 1am - 7am

# Apply the quote filter here (this filter will only apply to Quotes vs Quote Conversion)
filtered_data_quotes = filtered_data_quotes[(
    filtered_data_quotes['quotes'] >= quote_filter[0]) & (filtered_data_quotes['quotes'] <= quote_filter[1])
]

# Apply the transaction filter to Quotes vs Quote Conversion
filtered_data_quotes_transactions = filtered_data_quotes[(
    filtered_data_quotes['transactions'] >= transaction_filter[0]) & (
    filtered_data_quotes['transactions'] <= transaction_filter[1])
]

# Prepare the data for regression (using 'quotes' as the independent variable)
X_quotes = filtered_data_quotes_transactions[['quotes']]  # Define X_quotes
y_quotes = filtered_data_quotes_transactions['quoteConversion']

# Fit the model to the data
X_poly_quotes = poly.fit_transform(X_quotes)
model_quotes = LinearRegression()
model_quotes.fit(X_poly_quotes, y_quotes)

# Generate evenly spaced values of 'quotes' for smooth plotting
x_range_quotes = np.linspace(filtered_data_quotes_transactions['quotes'].min(), filtered_data_quotes_transactions['quotes'].max(), 100).reshape(-1, 1)
x_range_poly_quotes = poly.transform(x_range_quotes)

# Predict the corresponding 'quoteConversion' values
y_range_quotes_pred = model_quotes.predict(x_range_poly_quotes)

# Create scatter plot for Quotes vs Quote Conversion
fig2 = px.scatter(
    filtered_data_quotes_transactions,
    x='quotes',
    y='quoteConversion',
    color='hours',
    title="Quotes vs Quote Conversion"
)

# Add the quadratic regression line
fig2.add_scatter(
    x=x_range_quotes.flatten(),  # Use the new range of quotes
    y=y_range_quotes_pred,       # Predicted quoteConversion values
    mode='lines',
    name='Quadratic Fit',
    line=dict(color='red', dash='dash', width=3)  # Thicker line
)

# Update y-axis to show percentage with two decimals (e.g., 25.00%)
fig2.update_layout(
    yaxis_tickformat='.2%'  # Format percentage with 2 decimal places
)

# Display both plots in the Streamlit app
st.plotly_chart(fig1)  # First scatter plot (Sessions vs Quote Conversion)
st.plotly_chart(fig2)  # Second scatter plot (Quotes vs Quote Conversion)

# Display sessionConversion and quoteConversion as percentages in the sidebar
st.sidebar.write(f"Session Conversion (percentage): {data['sessionConversion'].mean() * 100:.2f}%")
st.sidebar.write(f"Quote Conversion (percentage): {data['quoteConversion'].mean() * 100:.2f}%")
