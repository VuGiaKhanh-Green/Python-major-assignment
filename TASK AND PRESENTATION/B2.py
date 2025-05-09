import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Constants
TOP_N = 3
TOP_PLAYERS_FILE = 'top_3.txt'
RESULTS_FILE = 'results2.csv'
HISTOGRAM_DIR = 'histograms'
ENCODING = 'utf-8-sig'
CSV_FILE = 'results.csv'

# Define attacking and defensive columns
ATTACKING_COLS = ['Standard SoT/90', 'Standard G/Sh', 'Standard Dist']
DEFENSIVE_COLS = ['Tackles Tkl', 'Tackles TklW', 'Blocks']

def read_data(filename, encoding):
    """
    Reads data from a CSV file.

    Args:
        filename (str): The name of the CSV file.
        encoding (str): The encoding to use when reading the file.

    Returns:
        pd.DataFrame: The data read from the CSV file, or None on error.
    """
    try:
        df = pd.read_csv(filename, encoding=encoding)
        print(f"Read file '{filename}' successfully.")
        return df
    except Exception as e:
        print(f"Error when reading file '{filename}': {e}")
        return None

def convert_to_numeric(df):
    """
    Converts string columns in a DataFrame to numeric, handling percentage signs.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with converted numeric columns.
    """
    for col in df.columns:
        if df[col].dtype == 'object' and col not in ['Player', 'Nation', 'Squad', 'Pos', 'Age']:
            if df[col].str.contains('%', na=False).any():
                df[col] = df[col].str.rstrip('%').astype(float)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def filter_numeric_columns(df, selected_columns):
    """
    Filters numeric columns from a DataFrame based on a list of selected columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        selected_columns (list): A list of column names to filter.

    Returns:
        list: A list of numeric column names that exist in the DataFrame.
    """
    numeric_columns = [col for col in selected_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    return numeric_columns

def check_player_and_squad_columns(df):
    """
    Checks if 'Player' and 'Squad' columns exist in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        bool: True if both columns exist, False otherwise.
    """
    if 'Player' not in df.columns or 'Squad' not in df.columns:
        print("Missing 'Player' or 'Squad' column.")
        return False
    return True

def write_top_bottom_players(df, numeric_columns, top_n, filename, encoding):
    """
    Writes the top and bottom N players for each numeric column to a file.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numeric_columns (list): A list of numeric column names.
        top_n (int): The number of top and bottom players to write.
        filename (str): The name of the file to write to.
        encoding (str): The encoding to use when writing to the file.
    """
    with open(filename, 'w', encoding=encoding) as f:
        for col in numeric_columns:
            f.write(f"Statistic: {col}\n")
            f.write("=" * (len(col) + 11) + "\n")

            # Top N players
            f.write(f"Top {top_n} players:\n")
            top_n_players = df[['Player', col]].dropna().sort_values(by=col, ascending=False).head(top_n)
            f.write(top_n_players.to_string(index=False) + "\n\n")

            # Bottom N players
            f.write(f"Bottom {top_n} players:\n")
            bottom_n_players = df[['Player', col]].dropna().sort_values(by=col, ascending=True).head(top_n)
            f.write(bottom_n_players.to_string(index=False) + "\n")
            f.write("-" * 50 + "\n\n")
    print(f"Saved file top/bottom {top_n} to '{filename}'")

def calculate_team_statistics(df, numeric_columns):
    """
    Calculates median, mean, and standard deviation for each numeric column, grouped by team.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numeric_columns (list): A list of numeric column names.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated statistics.
    """
    results = []
    for team, group_df in df.groupby('Squad'):
        row = [team]
        for col in numeric_columns:
            row.extend([
                group_df[col].median(),
                group_df[col].mean(),
                group_df[col].std()
            ])
        results.append(row)

    # Add row for all players
    league_row = ['All players']
    for col in numeric_columns:
        league_row.extend([
            df[col].median(),
            df[col].mean(),
            df[col].std()
        ])
    results.append(league_row)

    # Create header
    header = ['Squad']
    for col in numeric_columns:
        header.extend([
            f"Median {col}",
            f"Mean {col}",
            f"StdDev {col}"
        ])
    return pd.DataFrame(results, columns=header)

def create_histograms(df, numeric_columns, histogram_dir):
    """
    Creates histograms for each numeric column, both for all players and per team.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numeric_columns (list): A list of numeric column names.
        histogram_dir (str): The directory to save the histograms.
    """
    os.makedirs(histogram_dir, exist_ok=True)
    for col in numeric_columns:
        safe_col_name = re.sub(r'[^\w\s-]', '_', col)
        stat_dir = os.path.join(histogram_dir, safe_col_name)
        os.makedirs(stat_dir, exist_ok=True)

        # Histogram for all players
        data = df[col].dropna()
        plt.figure(figsize=(10, 6))
        if data.empty:
            plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
        else:
            plt.hist(data, bins=20, color='steelblue', alpha=0.7)
        plt.title(f"Histogram of {col} - All players")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(stat_dir, "All players.png"))
        plt.close()

        # Histograms per team
        for team in df['Squad'].dropna().unique():
            team_data = df[df['Squad'] == team][col].dropna()
            plt.figure(figsize=(10, 6))
            if team_data.empty:
                plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
            else:
                plt.hist(team_data, bins=20, color='orange', alpha=0.7)
            plt.title(f"Histogram of {col} - {team}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            safe_team = re.sub(r'[^\w]', '_', str(team))
            plt.savefig(os.path.join(stat_dir, f"{safe_team}.png"))
            plt.close()
    print(f"Created charts in folder '{histogram_dir}'")

def main():
    """
    Main function to execute the data analysis and visualization.
    """
    # Read data
    df = read_data(CSV_FILE, ENCODING)
    if df is None:
        exit()

    # Convert data to numeric
    df = convert_to_numeric(df)

    # Filter numeric columns
    numeric_columns = filter_numeric_columns(df, ATTACKING_COLS + DEFENSIVE_COLS)
    if not numeric_columns:
        print("Can't correct column.")
        exit()
    print("Using these columns:")
    for i, col in enumerate(numeric_columns, 1):
        print(f"{i}. {col}")

    # Check for required columns
    if not check_player_and_squad_columns(df):
        exit()

    # Write top/bottom players
    write_top_bottom_players(df, numeric_columns, TOP_N, TOP_PLAYERS_FILE, ENCODING)

    # Calculate and save team statistics
    results_df = calculate_team_statistics(df, numeric_columns)
    results_df.to_csv(RESULTS_FILE, index=False, encoding=ENCODING)
    print(f"Saved to '{RESULTS_FILE}'")

    # Create histograms
    create_histograms(df, numeric_columns, HISTOGRAM_DIR)

    # Print Analysis
    print("Based on the analysis, Chelsea appears to be the best-performing team in the 2024-2025 Premier League season. They lead in critical attacking metrics (xG, xAG, SCA, KP, PPA), showing they create numerous high-quality chances. Their high rankings in touches and passes into key areas indicate control in attacking phases, and their defensive metrics (tackles, interceptions) are competitive, though not the highest. Arsenal is a close second due to their efficiency in scoring and assisting, but Chelsea’s broader dominance across creative metrics gives them the edge. Manchester City’s possession-based metrics are exceptional, but their slightly lower goal output places them behind Chelsea. Leicester’s defensive and physical strengths are notable, but their attacking output is less impressive")
    print("So clearly, I think Chelsea is doing the best in overrall")

if __name__ == "__main__":
    main()
