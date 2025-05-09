import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Constants
INPUT_CSV = 'results.csv'
TRANSFER_CSV = 'transfer_values.csv'
PREDICT_CSV = 'transfer_predictions.csv'
TRANSFERMARKT_URL = 'https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query='
FOOTBALLTRANSFERS_URL = 'https://www.footballtransfers.com/en/search?q='
SELENIUM_TIMEOUT = 20
WEBDRIVER_WAIT_SLEEP = 3
MAX_WEB_ATTEMPTS = 3
RANDOM_STATE = 42

# Features for the model
FEATURES = ['Age', 'Min', 'Gls', 'Ast', 'xG', 'xAG', 'PrgC', 'PrgP', 'PrgR', 'SoT%', 'SoT/90', 'Tkl', 'TklW', 'Blocks', 'Touches', 'Succ%', 'Fls', 'Fld', 'Won%']

def load_data(input_csv):
    """Loads player data from a CSV file.

    Args:
        input_csv (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame, or None on error.
    """
    try:
        df = pd.read_csv(input_csv, encoding='utf-8-sig')
        print(f"Loaded data from '{input_csv}'")
        return df
    except FileNotFoundError:
        print(f"Error: '{input_csv}' not found.")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def preprocess_data(df):
    """Preprocesses the player data.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: Preprocessed DataFrame, or None if required columns are missing.
    """
    required_cols = ['Player', 'Playing Time Min']
    team_col = next((col for col in ['Team', 'Squad', 'team', 'TEAM'] if col in df.columns), None)

    if not all(col in df.columns for col in required_cols) or not team_col:
        print(f"Error: Required columns missing. Found: {df.columns.tolist()}")
        return None

    df['Playing Time Min'] = pd.to_numeric(df['Playing Time Min'].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['Playing Time Min'])
    players_900 = df[df['Playing Time Min'] > 900][['Player', team_col, 'Playing Time Min']].rename(columns={team_col: 'Team'})
    return players_900

def setup_selenium(headless=True):
    """Sets up the Selenium WebDriver.

    Args:
        headless (bool, optional): Whether to run in headless mode. Defaults to True.

    Returns:
        webdriver.Chrome: The initialized WebDriver.
    """
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--allow-insecure-localhost")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.accept_insecure_certs = True
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(60)
    return driver

def scrape_transfer_value_transfermarkt(driver, player):
    """Scrapes transfer value from Transfermarkt.

    Args:
        driver (webdriver.Chrome): The Selenium WebDriver.
        player (str): The player name.

    Returns:
        str: The transfer value, or None if not found.
    """
    try:
        driver.get(f"{TRANSFERMARKT_URL}{player.replace(' ', '+')}")
        link = WebDriverWait(driver, SELENIUM_TIMEOUT).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "td.hauptlink a[href*='/profil/spieler/']"))
        )
        player_url = link.get_attribute('href')
        driver.get(player_url)
        transfer_value = WebDriverWait(driver, SELENIUM_TIMEOUT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/marktwertverlauf/spieler/']"))
        ).text
        print(f"Scraped {player} from Transfermarkt: {transfer_value}")
        return transfer_value
    except Exception:
        return None

def scrape_transfer_value_footballtransfers(driver, player):
    """Scrapes transfer value from FootballTransfers.

    Args:
        driver (webdriver.Chrome): The Selenium WebDriver.
        player (str): The player name.

    Returns:
        str: The transfer value, or None if not found.
    """
    try:
        driver.get(f"{FOOTBALLTRANSFERS_URL}{player.replace(' ', '+')}")
        transfer_value = WebDriverWait(driver, SELENIUM_TIMEOUT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.market-value"))
        ).text
        print(f"Scraped {player} from FootballTransfers: {transfer_value}")
        return transfer_value
    except Exception:
        return None

def scrape_transfer_values(players_df, headless=True):
    """Scrapes transfer values for a list of players.

    Args:
        players_df (pd.DataFrame): DataFrame with player names.
        headless (bool, optional): Whether to run the browser in headless mode. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with player names and transfer values.
    """
    driver = setup_selenium(headless)
    values = []

    for _, row in players_df.iterrows():
        player = row['Player']
        transfer_value = None

        for attempt in range(MAX_WEB_ATTEMPTS):
            transfer_value = scrape_transfer_value_transfermarkt(driver, player)
            if transfer_value:
                break
            time.sleep(WEBDRIVER_WAIT_SLEEP)

        if not transfer_value:
            for attempt in range(MAX_WEB_ATTEMPTS):
                transfer_value = scrape_transfer_value_footballtransfers(driver, player)
                if transfer_value:
                    break
                time.sleep(WEBDRIVER_WAIT_SLEEP)
            else:
                transfer_value = 'N/a'

        values.append({'player_name': player, 'transfer_value': transfer_value})
        time.sleep(WEBDRIVER_WAIT_SLEEP)  # Add a delay

    driver.quit()
    return pd.DataFrame(values)

def clean_transfer_value(value):
    """Cleans and converts a transfer value string to a numeric value.

    Args:
        value (str): The transfer value string.

    Returns:
        float: The numeric transfer value, or np.nan if conversion fails.
    """
    if value == 'N/a' or pd.isna(value):
        return np.nan
    try:
        value = value.replace('€', '').replace('£', '').strip()
        if 'm' in value.lower():
            return float(value.lower().replace('m', '')) * 1e6
        elif 'k' in value.lower():
            return float(value.lower().replace('k', '')) * 1e3
        return float(value)
    except:
        return np.nan

def merge_data(players_df, transfer_data):
    """Merges player data with transfer values.

    Args:
        players_df (pd.DataFrame): DataFrame with player information.
        transfer_data (pd.DataFrame): DataFrame with transfer values.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    merged_data = pd.merge(players_df, transfer_data, left_on='Player', right_on='player_name', how='left')
    merged_data['transfer_value'] = merged_data['transfer_value'].apply(clean_transfer_value)
    return merged_data

def get_numeric_columns(df, cols):
    """Filters a list of columns and returns only the numeric ones present in the DataFrame

    Args:
        df (pd.DataFrame): The dataframe to check
        cols (list): A list of column names

    Returns:
        list: A list of column names that are both in the dataframe and numeric
    """
    numeric_cols = []
    for col in cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col].astype(str).replace(',', ''), errors='coerce')
                if df[col].notna().sum() > 0:  # Check if column has any non-NaN values
                    numeric_cols.append(col)
            except:
                pass
    return numeric_cols
def train_and_predict(df):
    """Trains a RandomForestRegressor model and makes predictions.

    Args:
        df (pd.DataFrame): DataFrame with player data and transfer values.

    Returns:
        tuple: (mse, r2, predictions_df)
    """
    available_features = get_numeric_columns(df, FEATURES)
    if not available_features:
        print("No valid features for training. Exiting.")
        return None, None, None

    df['transfer_value'] = df['transfer_value']  # Ensure 'transfer_value' is present

    valid_df = df.dropna(subset=available_features + ['transfer_value']) # Drop rows with missing values

    X = valid_df[available_features]
    y = valid_df['transfer_value']
    players = valid_df['Player']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=RANDOM_STATE)

    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.2f}, R^2: {r2:.2f}")

    full_pred = model.predict(X_scaled)  # Predict on the entire valid dataset

    predictions_df = pd.DataFrame({
        'Player': players.values,
        'Actual_Value': y.values,
        'Predicted_Value': full_pred
    })
    return mse, r2, predictions_df

def main():
    """Main function to orchestrate the data loading, scraping, and prediction process.
    """
    # Load data
    df = load_data(INPUT_CSV)
    if df is None:
        exit()

    # Preprocess data
    players_900 = preprocess_data(df)
    if players_900 is None:
        exit()

    # Scrape transfer values
    try:
        transfer_data = scrape_transfer_values(players_900, headless=True)
        transfer_data.to_csv(TRANSFER_CSV, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"Scraping failed: {e}. Using fallback data.")
        transfer_data = pd.DataFrame({'player_name': players_900['Player'], 'transfer_value': [np.nan] * len(players_900)})
        transfer_data.to_csv(TRANSFER_CSV, index=False, encoding='utf-8-sig')

    # Merge data
    merged_data = merge_data(players_900, transfer_data)

    # Train and predict
    mse, r2, predictions_df = train_and_predict(merged_data)
    if predictions_df is not None:
        predictions_df.to_csv(PREDICT_CSV, index=False, encoding='utf-8-sig')
        print(f"Done. Check '{TRANSFER_CSV}', '{PREDICT_CSV}'.")
    else:
        print("Prediction process failed.")

if __name__ == "__main__":
    main()
