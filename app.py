import pyodbc
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix

# --- Configuration & Constants ---

# Database connection settings (Using details primarily from clv_dashboard.py for robustness)
SERVER = 'newretailserver123.database.windows.net'
DATABASE = 'RetailDB'
USERNAME = 'azureuser'
PASSWORD = 'YourStrongP@ssw0rd' # Replace with your actual password
DRIVER = '{ODBC Driver 18 for SQL Server}' # Consistent driver format

# --- Helper Functions ---

# Password Hashing (from clv_dashboard.py)
def make_hashes(password):
    """Generates a SHA256 hash for a given password."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    """Checks if a password matches a stored hash."""
    return make_hashes(password) == hashed_text

# Database Connection (Combined approach, favoring clv_dashboard.py's robustness)
def get_connection():
    """Establishes and returns a database connection."""
    try:
        conn_str = (
            f'DRIVER={DRIVER};'
            f'SERVER={SERVER};'
            f'DATABASE={DATABASE};'
            f'UID={USERNAME};'
            f'PWD={PASSWORD};'
            'Encrypt=yes;' # Added for security
            'TrustServerCertificate=yes;' # Adjust based on your server cert setup
            'Connection Timeout=30;' # Added for reliability
        )
        conn = pyodbc.connect(conn_str)
        return conn
    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        st.error(f"Database connection error: {sqlstate}. Please check connection details and network.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during connection: {e}")
        return None


# Data Loading Functions

# Function to load data for a specific HSHD_NUM (from search_household.py)
def load_data_for_household(hshd_num):
    """Loads transaction and product data for a specific household number."""
    query = """
    SELECT
        H.HSHD_NUM,
        T.BASKET_NUM,
        T.PURCHASE_ AS Date, -- Using PURCHASE_ for date
        P.PRODUCT_NUM,
        P.DEPARTMENT,
        P.COMMODITY
    FROM dbo.households H
    JOIN dbo.transactions T ON H.HSHD_NUM = T.HSHD_NUM
    LEFT JOIN dbo.products P ON T.PRODUCT_NUM = P.PRODUCT_NUM
    WHERE H.HSHD_NUM = ?
    ORDER BY H.HSHD_NUM, T.BASKET_NUM, T.PURCHASE_, P.PRODUCT_NUM, P.DEPARTMENT, P.COMMODITY;
    """
    conn = get_connection()
    if conn:
        try:
            df = pd.read_sql(query, conn, params=[hshd_num])
            conn.close()
            return df
        except Exception as e:
            st.error(f"Error fetching data for household {hshd_num}: {e}")
            conn.close()
            return pd.DataFrame() # Return empty dataframe on error
    else:
        return pd.DataFrame() # Return empty dataframe if connection failed

# Cached function to load all data from DB (from clv_dashboard.py)
@st.cache_data(ttl=600) # Cache data for 10 minutes
def load_all_data_from_db():
    """Loads all transactions, households, and products data from the database."""
    conn = get_connection()
    if conn:
        try:
            query_transactions = "SELECT * FROM Transactions"
            query_households = "SELECT * FROM Households"
            query_products = "SELECT * FROM Products"

            df_transactions = pd.read_sql(query_transactions, conn)
            df_households = pd.read_sql(query_households, conn)
            df_products = pd.read_sql(query_products, conn)
            conn.close()

            # Clean column names
            df_transactions.columns = df_transactions.columns.str.strip()
            df_households.columns = df_households.columns.str.strip()
            df_products.columns = df_products.columns.str.strip()

            # Rename columns for consistency (using lowercase)
            df_transactions.rename(columns={'HSHD_NUM': 'hshd_num', 'PRODUCT_NUM': 'product_num'}, inplace=True)
            df_households.rename(columns={'HSHD_NUM': 'hshd_num'}, inplace=True)
            df_products.rename(columns={'PRODUCT_NUM': 'product_num'}, inplace=True)

            # Convert date components to datetime
            try:
                df_transactions['date'] = pd.to_datetime(df_transactions['YEAR'].astype(str) + df_transactions['WEEK_NUM'].astype(str) + '0', format='%Y%U%w', errors='coerce')
            except KeyError:
                st.warning("Could not find 'YEAR' or 'WEEK_NUM' columns for date conversion in Transactions.")
            except Exception as e:
                 st.warning(f"Error converting date columns: {e}")


            return df_transactions, df_households, df_products
        except Exception as e:
            st.error(f"Error loading data from database: {e}")
            conn.close()
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    else:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- Authentication ---

# Initialize session state variables
if 'user_db' not in st.session_state:
    st.session_state.user_db = {} # In-memory user store (for demo purposes)
    # You might want to replace this with a more persistent store
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None

def login_signup():
    """Handles the login and signup logic in the sidebar."""
    if not st.session_state.authenticated:
        auth_option = st.sidebar.selectbox("Login or Signup", ["Login", "Signup"], key="auth_choice")

        if auth_option == "Signup":
            st.sidebar.subheader("Create New Account")
            new_user = st.sidebar.text_input("Username", key="signup_user")
            # new_email = st.sidebar.text_input("Email", key="signup_email") # Email field from original code
            new_password = st.sidebar.text_input("Password", type='password', key="signup_pass")

            if st.sidebar.button("Signup", key="signup_button"):
                if new_user and new_password:
                    if new_user in st.session_state.user_db:
                        st.sidebar.error("Username already exists.")
                    else:
                        hashed_pw = make_hashes(new_password)
                        # Using username as key, storing hashed password
                        st.session_state.user_db[new_user] = {"password": hashed_pw}
                        st.sidebar.success("Signup successful! Please login.")
                else:
                    st.sidebar.error("Username and password cannot be empty.")

        elif auth_option == "Login":
            st.sidebar.subheader("Login to Your Account")
            username = st.sidebar.text_input("Username", key="login_user")
            password = st.sidebar.text_input("Password", type='password', key="login_pass")

            if st.sidebar.button("Login", key="login_button"):
                user_data = st.session_state.user_db.get(username)
                if user_data and check_hashes(password, user_data['password']):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun() # Rerun the script to reflect logged-in state
                else:
                    st.sidebar.error("Invalid username or password.")
    else:
        # Show logged-in user and logout button
        st.sidebar.success(f"Logged in as: {st.session_state.username}")
        if st.sidebar.button("Logout", key="logout_button"):
            st.session_state.authenticated = False
            st.session_state.username = None
            # Clear cached data on logout if desired
            # st.cache_data.clear()
            st.rerun()


# --- Streamlit App Main Function ---

def main_app():
    """Defines the main structure and logic of the Streamlit dashboard."""
    st.set_page_config(page_title="🍭️ Retail Insights Dashboard", layout="wide")

    # --- Authentication Check ---
    login_signup()

    if not st.session_state.authenticated:
        st.warning("Please login or signup using the sidebar to access the dashboard.")
        st.stop() # Stop execution if not authenticated

    # --- Title ---
    st.title("📊 Retail Customer Analytics Dashboard")
    st.write(f"Welcome, {st.session_state.username}!")


    # --- Data Loading Section (Sidebar Option) ---
    st.sidebar.title("Data Source")
    data_source_option = st.sidebar.radio(
        "Choose data source:",
        ('Load from Database', 'Upload CSV Files', 'Search Single Household'),
        key="data_source"
    )

    # Initialize dataframes in session state if they don't exist
    if 'transactions_df' not in st.session_state:
        st.session_state['transactions_df'] = pd.DataFrame()
    if 'households_df' not in st.session_state:
        st.session_state['households_df'] = pd.DataFrame()
    if 'products_df' not in st.session_state:
        st.session_state['products_df'] = pd.DataFrame()
    if 'full_df' not in st.session_state:
        st.session_state['full_df'] = pd.DataFrame()
    if 'single_household_df' not in st.session_state:
        st.session_state['single_household_df'] = pd.DataFrame()


    # --- Handle Data Loading Based on Selection ---

    if data_source_option == 'Load from Database':
        if st.sidebar.button("Load/Refresh Database Data", key="load_db_button"):
            with st.spinner("Loading data from database..."):
                tdf, hdf, pdf = load_all_data_from_db()
                if not tdf.empty and not hdf.empty and not pdf.empty:
                    st.session_state['transactions_df'] = tdf
                    st.session_state['households_df'] = hdf
                    st.session_state['products_df'] = pdf
                    # Merge dataframes immediately after loading
                    st.session_state['full_df'] = tdf.merge(hdf, on='hshd_num', how='left')
                    st.session_state['full_df'] = st.session_state['full_df'].merge(pdf, on='product_num', how='left')
                    st.sidebar.success("Database data loaded successfully!")
                    # Clear single household data if DB is loaded
                    st.session_state['single_household_df'] = pd.DataFrame()
                    st.rerun() # Rerun to update dashboard with new data
                else:
                    st.sidebar.error("Failed to load data from database.")

    elif data_source_option == 'Upload CSV Files':
        st.sidebar.subheader("Upload Your Datasets")
        uploaded_transactions = st.sidebar.file_uploader("Upload Transactions CSV", type="csv", key="upload_trans")
        uploaded_households = st.sidebar.file_uploader("Upload Households CSV", type="csv", key="upload_house")
        uploaded_products = st.sidebar.file_uploader("Upload Products CSV", type="csv", key="upload_prod")

        if uploaded_transactions and uploaded_households and uploaded_products:
            try:
                tdf = pd.read_csv(uploaded_transactions)
                hdf = pd.read_csv(uploaded_households)
                pdf = pd.read_csv(uploaded_products)

                 # Basic validation/cleaning (similar to DB load)
                tdf.columns = tdf.columns.str.strip()
                hdf.columns = hdf.columns.str.strip()
                pdf.columns = pdf.columns.str.strip()

                # Check for essential columns before renaming and merging
                required_trans_cols = {'HSHD_NUM', 'PRODUCT_NUM', 'YEAR', 'WEEK_NUM'}
                required_hh_cols = {'HSHD_NUM'}
                required_prod_cols = {'PRODUCT_NUM'}

                if not required_trans_cols.issubset(tdf.columns):
                    st.sidebar.error(f"Transactions CSV missing required columns: {required_trans_cols - set(tdf.columns)}")
                elif not required_hh_cols.issubset(hdf.columns):
                     st.sidebar.error(f"Households CSV missing required columns: {required_hh_cols - set(hdf.columns)}")
                elif not required_prod_cols.issubset(pdf.columns):
                     st.sidebar.error(f"Products CSV missing required columns: {required_prod_cols - set(pdf.columns)}")
                else:
                    # Rename columns for consistency
                    tdf.rename(columns={'HSHD_NUM': 'hshd_num', 'PRODUCT_NUM': 'product_num'}, inplace=True)
                    hdf.rename(columns={'HSHD_NUM': 'hshd_num'}, inplace=True)
                    pdf.rename(columns={'PRODUCT_NUM': 'product_num'}, inplace=True)

                    # Convert date components
                    try:
                         tdf['date'] = pd.to_datetime(tdf['YEAR'].astype(str) + tdf['WEEK_NUM'].astype(str) + '0', format='%Y%U%w', errors='coerce')
                    except Exception as e:
                         st.warning(f"Could not derive 'date' column from CSV: {e}")


                    st.session_state['transactions_df'] = tdf
                    st.session_state['households_df'] = hdf
                    st.session_state['products_df'] = pdf

                    # Merge dataframes
                    st.session_state['full_df'] = tdf.merge(hdf, on='hshd_num', how='left')
                    st.session_state['full_df'] = st.session_state['full_df'].merge(pdf, on='product_num', how='left')

                    st.sidebar.success("CSV files loaded and merged!")
                    # Clear single household data if CSVs are loaded
                    st.session_state['single_household_df'] = pd.DataFrame()
                    # Optionally rerun if uploads should immediately trigger dashboard update
                    st.rerun()

            except Exception as e:
                st.sidebar.error(f"Error processing uploaded CSV files: {e}")


    elif data_source_option == 'Search Single Household':
        st.sidebar.subheader("Fetch Data for One Household")
        hshd_num_input = st.sidebar.text_input("Enter Household Number (HSHD_NUM):", key="hshd_search_input")

        if st.sidebar.button("Search Household", key="hshd_search_button"):
            if hshd_num_input:
                try:
                    hshd_num = int(hshd_num_input)
                    with st.spinner(f"Fetching data for HSHD_NUM: {hshd_num}..."):
                        data = load_data_for_household(hshd_num)
                        if not data.empty:
                            st.session_state['single_household_df'] = data
                            st.sidebar.success(f"Data loaded for HSHD_NUM {hshd_num}.")
                            # Clear full dataframes if single household is loaded
                            st.session_state['transactions_df'] = pd.DataFrame()
                            st.session_state['households_df'] = pd.DataFrame()
                            st.session_state['products_df'] = pd.DataFrame()
                            st.session_state['full_df'] = pd.DataFrame()
                            st.rerun() # Rerun to display single household data
                        else:
                            st.sidebar.warning("No data found for the entered Household Number or error fetching data.")
                            st.session_state['single_household_df'] = pd.DataFrame() # Clear previous results
                except ValueError:
                    st.sidebar.error("Please enter a valid numeric Household Number.")
                except Exception as e:
                    st.sidebar.error(f"An error occurred during search: {e}")
            else:
                st.sidebar.warning("Please enter a Household Number.")

    # --- Display Data Previews (Optional) ---
    if st.sidebar.checkbox("Show Data Previews", key="show_previews"):
         if not st.session_state['transactions_df'].empty:
              st.sidebar.write("Transactions Preview:", st.session_state['transactions_df'].head())
         if not st.session_state['households_df'].empty:
              st.sidebar.write("Households Preview:", st.session_state['households_df'].head())
         if not st.session_state['products_df'].empty:
              st.sidebar.write("Products Preview:", st.session_state['products_df'].head())
         if not st.session_state['full_df'].empty:
              st.sidebar.write("Merged Data Preview:", st.session_state['full_df'].head())
         if not st.session_state['single_household_df'].empty:
              st.sidebar.write("Single Household Data Preview:", st.session_state['single_household_df'].head())


    # --- Main Dashboard Area ---

    # Display Single Household Data OR Full Dashboard
    if not st.session_state['single_household_df'].empty:
        # Display data for the single household that was searched
        st.header(f"Data for Household Number: {st.session_state['single_household_df']['HSHD_NUM'].iloc[0]}")
        st.dataframe(st.session_state['single_household_df'])

    elif not st.session_state['full_df'].empty:
        # Display the full dashboard using the merged dataframe ('full_df')

        # Use the merged dataframe
        full_df = st.session_state['full_df']
        df_transactions = st.session_state['transactions_df'] # Keep separate for specific analyses if needed
        df_households = st.session_state['households_df']
        df_products = st.session_state['products_df']

        # --- Dashboard Tabs ---
        tab_titles = ["📈 Overview & Demographics", "🧺 Basket & Products", "❓ Customer Behavior (CLV & Churn)", "🤖 ML Predictions"]
        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

        with tab1:
            st.header("📈 Customer Engagement Over Time")
            # Ensure 'date' and 'SPEND' columns exist
            if 'date' in df_transactions.columns and 'SPEND' in df_transactions.columns:
                # Calculate weekly engagement
                 weekly_engagement = df_transactions.copy()
                 weekly_engagement['date_period'] = weekly_engagement['date'].dt.to_period('W')
                 weekly_spend = weekly_engagement.groupby('date_period')['SPEND'].sum().reset_index()
                 weekly_spend['ds'] = weekly_spend['date_period'].dt.start_time # Use start time for plotting

                 if not weekly_spend.empty:
                     st.line_chart(weekly_spend.set_index('ds')['SPEND'])
                 else:
                     st.warning("No weekly spending data available to plot.")
            else:
                st.warning("Required columns ('date', 'SPEND') not found in transaction data for engagement chart.")


            st.header("👨‍👩‍👧 Demographics and Spending")
            # Ensure necessary columns exist in full_df
            demo_options = [col for col in ['INCOME_RANGE', 'AGE_RANGE', 'CHILDREN'] if col in full_df.columns]
            if demo_options and 'SPEND' in full_df.columns:
                selected_demo = st.selectbox("Segment Spending by:", demo_options, key="demo_select")
                if selected_demo:
                    # Handle potential NaN values before grouping
                    demo_spending = full_df.dropna(subset=[selected_demo, 'SPEND']).groupby(selected_demo)['SPEND'].sum().reset_index()
                    st.bar_chart(demo_spending.rename(columns={selected_demo: 'index'}).set_index('index'))
            else:
                 st.warning("Required columns for demographic segmentation ('INCOME_RANGE', 'AGE_RANGE', 'CHILDREN', 'SPEND') not found.")

            st.header("🏆 Top 10 Customers by Spending")
            if 'hshd_num' in full_df.columns and 'SPEND' in full_df.columns:
                 top_customers = full_df.groupby('hshd_num')['SPEND'].sum().reset_index().sort_values(by='SPEND', ascending=False)
                 st.dataframe(top_customers.head(10))
            else:
                 st.warning("Required columns ('hshd_num', 'SPEND') not found for top customer analysis.")


        with tab2:
            st.header("🧺 Basket Analysis - Top Products & Categories")
            # Ensure required columns exist
            if all(col in df_transactions.columns for col in ['BASKET_NUM', 'product_num', 'SPEND']) and \
               all(col in df_products.columns for col in ['product_num', 'COMMODITY', 'DEPARTMENT']):

                basket = df_transactions.groupby(['BASKET_NUM', 'product_num'])['SPEND'].sum().reset_index()
                top_products = basket.groupby('product_num')['SPEND'].sum().nlargest(20).reset_index() # Show top 20
                top_products = top_products.merge(df_products[['product_num', 'COMMODITY', 'DEPARTMENT']], on='product_num', how='left')

                # Aggregate by Commodity
                commodity_spending = top_products.groupby('COMMODITY')['SPEND'].sum().reset_index().sort_values('SPEND', ascending=False)
                if not commodity_spending.empty:
                    st.subheader("Top Product Commodities by Spend")
                    fig_com = px.bar(commodity_spending.head(10), x='COMMODITY', y='SPEND', title='Top 10 Commodities by Spend')
                    st.plotly_chart(fig_com)

                    st.subheader("Spending Distribution by Commodity (Top 20 Products)")
                    fig_pie_com = px.pie(commodity_spending, values='SPEND', names='COMMODITY', title='Spending Distribution by Commodity')
                    st.plotly_chart(fig_pie_com)
                else:
                    st.warning("No commodity spending data to display.")

                 # Aggregate by Department
                department_spending = top_products.groupby('DEPARTMENT')['SPEND'].sum().reset_index().sort_values('SPEND', ascending=False)
                if not department_spending.empty:
                    st.subheader("Top Product Departments by Spend")
                    fig_dep = px.bar(department_spending.head(10), x='DEPARTMENT', y='SPEND', title='Top 10 Departments by Spend')
                    st.plotly_chart(fig_dep)
                else:
                     st.warning("No department spending data to display.")

            else:
                st.warning("Required columns ('BASKET_NUM', 'product_num', 'SPEND', 'COMMODITY', 'DEPARTMENT') not found for basket analysis.")


        with tab3:
            st.header("💰 Customer Lifetime Value (Total Spend per Customer)")
            if 'hshd_num' in df_transactions.columns and 'SPEND' in df_transactions.columns:
                 clv = df_transactions.groupby('hshd_num')['SPEND'].sum().reset_index().sort_values(by='SPEND', ascending=False)
                 st.dataframe(clv.head(20)) # Show top 20
                 # Basic CLV distribution plot
                 fig_clv_hist = px.histogram(clv, x='SPEND', title='Distribution of Customer Total Spending (CLV Proxy)')
                 st.plotly_chart(fig_clv_hist)
            else:
                 st.warning("Required columns ('hshd_num', 'SPEND') not found for CLV analysis.")


            st.header("📆 Seasonal Spending Patterns")
            if 'date' in df_transactions.columns and 'SPEND' in df_transactions.columns:
                df_transactions['month'] = df_transactions['date'].dt.month_name()
                seasonal = df_transactions.groupby('month')['SPEND'].sum().reset_index()
                # Order months correctly
                month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                seasonal['month'] = pd.Categorical(seasonal['month'], categories=month_order, ordered=True)
                seasonal = seasonal.sort_values('month')
                st.bar_chart(seasonal.set_index('month'))
            else:
                 st.warning("Required columns ('date', 'SPEND') not found for seasonal analysis.")

            st.header("🌟 Loyalty Program Effect")
            # Check if loyalty flag exists in households and spend in full_df
            if 'LOYALTY_FLAG' in df_households.columns and 'LOYALTY_FLAG' in full_df.columns and 'SPEND' in full_df.columns:
                 # Ensure consistent typing if necessary, e.g., convert flag to string or bool
                 full_df['LOYALTY_FLAG'] = full_df['LOYALTY_FLAG'].astype(str) # Example conversion
                 loyalty = full_df.groupby('LOYALTY_FLAG')['SPEND'].agg(['sum', 'mean']).reset_index()
                 st.dataframe(loyalty)
                 # Add a comparison chart
                 fig_loyalty = px.bar(loyalty, x='LOYALTY_FLAG', y=['sum', 'mean'], barmode='group', title='Spending by Loyalty Status')
                 st.plotly_chart(fig_loyalty)

            else:
                st.warning("Required columns ('LOYALTY_FLAG', 'SPEND') not found or not merged correctly for loyalty analysis.")


        with tab4:
            st.header("🤖 Machine Learning Predictions")
            st.info("This section uses calculated features based on historical data.")

            # Check if necessary data is available
            if not df_transactions.empty and all(col in df_transactions.columns for col in ['hshd_num', 'date', 'BASKET_NUM', 'SPEND']):

                # --- Churn Prediction Section ---
                st.subheader("⚠️ Churn Prediction Model (Based on RFM)")
                try:
                    now = df_transactions['date'].max()
                    if pd.isna(now):
                         st.error("Cannot determine the latest transaction date. Churn prediction aborted.")
                    else:
                        rfm = df_transactions.groupby('hshd_num').agg(
                            recency=('date', lambda x: (now - x.max()).days if pd.notna(x.max()) else 9999), # Handle potential NaT in max()
                            frequency=('BASKET_NUM', 'nunique'),
                            monetary=('SPEND', 'sum')
                        ).reset_index()

                        # Define churn (e.g., recency > 84 days)
                        rfm['churn'] = (rfm['recency'] > 84).astype(int)

                        # Ensure we have data after aggregation
                        if not rfm.empty and 'churn' in rfm.columns and rfm['churn'].nunique() > 1:
                            X = rfm[['recency', 'frequency', 'monetary']]
                            y = rfm['churn']

                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

                            # Train Model
                            with st.spinner("Training Churn Prediction model..."):
                                clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # Added class_weight
                                clf.fit(X_train, y_train)
                                y_pred = clf.predict(X_test)
                                y_proba = clf.predict_proba(X_test)[:, 1] # Probability of churn

                            st.success("Churn model trained.")

                            # Display Results
                            st.write("**Model Performance (Test Set):**")
                            st.text(classification_report(y_test, y_pred))
                            st.write("**Confusion Matrix:**")
                            cm = confusion_matrix(y_test, y_pred)
                            st.dataframe(pd.DataFrame(cm,
                                                      columns=['Predicted Not Churn', 'Predicted Churn'],
                                                      index=['Actual Not Churn', 'Actual Churn']))

                            st.write("**Feature Importances:**")
                            feat_imp = pd.Series(clf.feature_importances_, index=X.columns)
                            st.bar_chart(feat_imp.sort_values(ascending=False))

                            # Show customers predicted to churn
                            st.subheader("Customers Predicted to Churn (Test Set)")
                            churn_predictions = X_test.copy()
                            churn_predictions['Actual Churn'] = y_test
                            churn_predictions['Predicted Churn'] = y_pred
                            churn_predictions['Churn Probability'] = y_proba
                            st.dataframe(churn_predictions[churn_predictions['Predicted Churn'] == 1].sort_values('Churn Probability', ascending=False))

                        else:
                            st.warning("Insufficient data or churn variance to train churn model after RFM calculation.")

                except Exception as e:
                     st.error(f"An error occurred during churn prediction: {e}")


                # --- Basket Spend Prediction Section ---
                st.subheader("🧺 Basket Spend Prediction Model")
                try:
                    # Need merged data with product details for features
                    if not full_df.empty and all(col in full_df.columns for col in ['BASKET_NUM', 'SPEND', 'DEPARTMENT', 'COMMODITY', 'BRAND_TY', 'NATURAL_ORGANIC_FLAG']):

                        # Prepare features (handle missing values before one-hot encoding)
                        features_to_encode = ['DEPARTMENT', 'COMMODITY', 'BRAND_TY', 'NATURAL_ORGANIC_FLAG']
                        basket_features_raw = full_df[['BASKET_NUM'] + features_to_encode].copy()

                        # Simple imputation (fill with 'Unknown' or mode) - adjust strategy as needed
                        for col in features_to_encode:
                            basket_features_raw[col].fillna('Unknown', inplace=True)

                        # One-hot encode categorical features
                        basket_features_encoded = pd.get_dummies(basket_features_raw, columns=features_to_encode, prefix=features_to_encode)

                        # Aggregate features by basket
                        X_basket = basket_features_encoded.groupby('BASKET_NUM').sum()

                        # Target variable: total spend per basket
                        y_basket = full_df.groupby('BASKET_NUM')['SPEND'].sum()

                        # Align X and y based on BASKET_NUM index
                        common_indices = X_basket.index.intersection(y_basket.index)
                        X_basket = X_basket.loc[common_indices]
                        y_basket = y_basket.loc[common_indices]

                        if not X_basket.empty and not y_basket.empty:
                             # Split data
                             X_train_bs, X_test_bs, y_train_bs, y_test_bs = train_test_split(X_basket, y_basket, test_size=0.25, random_state=42)

                             # Train Model
                             with st.spinner("Training Basket Spend Prediction model..."):
                                 rf_basket = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # Use more cores
                                 rf_basket.fit(X_train_bs, y_train_bs)
                                 y_pred_bs = rf_basket.predict(X_test_bs)

                             st.success("Basket Spend model trained.")

                             # Display Results
                             r2 = r2_score(y_test_bs, y_pred_bs)
                             mse = mean_squared_error(y_test_bs, y_pred_bs)
                             st.write("**Model Performance (Test Set):**")
                             st.write(f"- **R² Score:** {r2:.3f}")
                             st.write(f"- **Mean Squared Error (MSE):** {mse:.2f}")


                             st.subheader("Predicted vs. Actual Basket Spend (Sample from Test Set)")
                             chart_df = pd.DataFrame({"Actual": y_test_bs.values, "Predicted": y_pred_bs}).sample(min(1000, len(y_test_bs))) # Sample for performance
                             st.line_chart(chart_df.reset_index(drop=True))

                             st.subheader("📄 Actual vs. Predicted Spend Table (Sample)")
                             st.dataframe(chart_df)
                             csv = chart_df.to_csv(index=False).encode('utf-8')
                             st.download_button(
                                 label="📥 Download Sample Prediction Data (CSV)",
                                 data=csv,
                                 file_name='predicted_vs_actual_basket_spend_sample.csv',
                                 mime='text/csv',
                             )


                             # Feature importance
                             st.write("**Top Drivers of Basket Spend:**")
                             importances = pd.Series(rf_basket.feature_importances_, index=X_basket.columns)
                             top_features = importances.sort_values(ascending=False).head(20) # Show top 20 features
                             st.bar_chart(top_features)
                        else:
                             st.warning("Insufficient data after feature preparation for basket spend model.")

                    else:
                        st.warning("Required data (merged transaction and product details) not available for basket spend prediction.")

                except Exception as e:
                    st.error(f"An error occurred during basket spend prediction: {e}")
                    st.exception(e) # Print traceback for debugging

            else:
                 st.warning("Transaction data with required columns ('hshd_num', 'date', 'BASKET_NUM', 'SPEND') is needed for ML predictions.")


    elif data_source_option != 'Search Single Household': # Only show this if not viewing single household AND no data loaded
         st.info("Please load data using one of the options in the sidebar (Load from Database or Upload CSV Files) to view the dashboard.")


# --- Entry Point ---
if __name__ == "__main__":
    main_app()

