from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

DATA_FILE = "sales_forecasting_dataset.csv"

def get_base_data():
    if not os.path.exists(DATA_FILE):
        return None, f"File {DATA_FILE} not found"
    try:
        df = pd.read_csv(DATA_FILE)
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Month'] = df['Order Date'].dt.month
        return df, None
    except Exception as e:
        return None, str(e)

def get_forecast_logic(df):
    monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
    
    if len(monthly_sales) < 2:
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        empty = [{"label": m, "value": 0} for m in month_names]
        return empty, empty

    X = monthly_sales[['Month']]
    y = monthly_sales['Sales']
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_months = np.array(range(1, 13)).reshape(-1, 1)
    predictions = model.predict(future_months)
    
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    forecast_data = [
        {"label": month_names[int(m)-1], "value": round(float(s), 2)} 
        for m, s in zip(range(1, 13), predictions)
    ]
    
    historical_data = [
        {"label": month_names[int(m)-1], "value": round(float(s), 2)} 
        for m, s in zip(monthly_sales['Month'], monthly_sales['Sales'])
    ]
    
    return forecast_data, historical_data

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    df, error = get_base_data()
    if error:
        return f"Error: {error}", 500
    
    available_products = sorted(df['Product'].unique().tolist())
    available_regions = sorted(df['Region'].unique().tolist())

    search_query = request.args.get('search', '').strip().lower()
    
    filtered_df = df.copy()
    if search_query:
        mask = (
            df['Product'].str.lower().str.contains(search_query, na=False) | 
            df['Region'].str.lower().str.contains(search_query, na=False)
        )
        filtered_df = df[mask]

    forecast, historical = get_forecast_logic(df)
    

    table_data = filtered_df.sort_values(by='Order Date', ascending=False).head(50).to_dict('records')
    for row in table_data:
        row['Order Date'] = row['Order Date'].strftime('%Y-%m-%d')

    stats = {
        "total_sales": f"${df['Sales'].sum():,.2f}",
        "avg_sales": f"${df['Sales'].mean():,.2f}",
        "total_orders": f"{len(df):,}",
        "top_region": df.groupby('Region')['Sales'].sum().idxmax(),
        "top_product": df.groupby('Product')['Sales'].sum().idxmax()
    }
    
    data = {
        "forecast": forecast,
        "historical": historical,
        "stats": stats,
        "results": table_data,
        "total_count": len(filtered_df),
        "search_term": search_query,
        "products": available_products,
        "regions": available_regions,
    }
    
    return render_template("dashboard.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)