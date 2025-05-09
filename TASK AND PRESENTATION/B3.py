import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Constants
CSV_FILE = 'results.csv'
ENCODING = 'utf-8-sig'
OPTIMAL_K = 3
RANDOM_STATE = 42  # Added for consistent results


def read_data(filename, encoding):
    """
    Reads data from a CSV file.

    Args:
        filename (str): The name of the CSV file.
        encoding (str): The encoding to use.

    Returns:
        pd.DataFrame: The DataFrame, or None on error.
    """
    try:
        df = pd.read_csv(filename, encoding=encoding)
        print(f"Reading file '{filename}' successfully.")
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def preprocess_data(df):
    """
    Selects numeric columns and scales the data.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: Scaled numeric data, or None if no numeric columns.
        list: List of numeric column names.
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        print("No numeric columns to analyze.")
        return None, None
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_columns])
    return pd.DataFrame(scaled_data, columns=numeric_columns), numeric_columns


def find_optimal_k(scaled_data):
    """
    Finds the optimal number of clusters (k) using the Elbow Method.

    Args:
        scaled_data (pd.DataFrame): Scaled data.

    Returns:
        int: The optimal k.
    """
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    elbow_k = 3  # Based on visual inspection of the elbow plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.axvline(x=elbow_k, color='red', linestyle='--', label=f'Elbow at k={elbow_k}')
    plt.annotate('⬅ The most obvious Elbow', xy=(elbow_k, inertia[elbow_k - 1]),
                 xytext=(elbow_k + 3, inertia[elbow_k - 1] + 100),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
    plt.title("Elbow chart - Select the optimal number of clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return elbow_k


def perform_clustering(scaled_data, df, optimal_k):
    """
    Performs KMeans clustering and adds cluster labels to the DataFrame.

    Args:
        scaled_data (pd.DataFrame): Scaled data.
        df (pd.DataFrame): The original DataFrame.
        optimal_k (int): The optimal number of clusters.

    Returns:
        pd.DataFrame: The DataFrame with the 'Cluster' column added.
    """
    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE)
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    return df


def summarize_clusters(df, numeric_columns):
    """
    Calculates and prints the mean of numeric columns for each cluster.

    Args:
        df (pd.DataFrame): The DataFrame with cluster labels.
        numeric_columns (list): List of numeric column names.
    """
    print("\nAverage of cluster indices:")
    cluster_summary = df.groupby('Cluster')[numeric_columns].mean().round(2)
    print(cluster_summary)


def visualize_clusters(scaled_data, df):
    """
    Performs PCA and visualizes the clusters in a 2D plot.

    Args:
        scaled_data (pd.DataFrame): Scaled data.
        df (pd.DataFrame):The  DataFrame with cluster labels.
    """
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_data)

    df_pca = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
    df_pca['Cluster'] = df['Cluster']

    plt.figure(figsize=(10, 8))
    palette = {0: 'green', 1: 'blue', 2: 'red'}
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca,
                    palette=palette, s=100, alpha=0.7)
    plt.title("Player clustering (PCA 2D)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_comments():
    """Prints comments explaining the clustering results.
    """
    print("Comment:")
    print(
        "Choosing the number of clusters k=3 in the KMeans method is reasonable because, based on the Elbow Method, we observe a clear 'elbow' point at k=3. When plotting the 'inertia' (the total distance between data points and their cluster center), we notice that after k=3, the decrease in inertia becomes less significant. This indicates that three clusters provide the best division of the data, as increasing the number of clusters beyond k=3 does not substantially improve the dispersion of the points.")
    print(
        "As we can see in the Elbow chart, k = 2 can be a good way to cluster but there are some reasons for not doing that like ignoring an important division in the data, resulting in a lack of clear distinction between different groups of players if k = 2 is chosen.")
    print(
        "- Cluster 0: Defensive player(defender , midfielder, goalkeeper , etc..) (more in-time game, less attacking).")
    print("- Cluster 1: Substitute player, little contribution.")
    print("- Cluster 2: Attacking player with high scores/assists.")


def main():
    """Main function to orchestrate the clustering analysis.
    """
    # Read data
    df = read_data(CSV_FILE, ENCODING)
    if df is None:
        exit()

    # Preprocess data
    scaled_data, numeric_columns = preprocess_data(df)
    if scaled_data is None:
        exit()

    # Find optimal k
    optimal_k = find_optimal_k(scaled_data)

    # Perform clustering
    df_clustered = perform_clustering(scaled_data, df.copy(), optimal_k)  # Pass a copy to avoid modifying original df

    # Summarize clusters
    summarize_clusters(df_clustered, numeric_columns)

    # Visualize clusters
    visualize_clusters(scaled_data, df_clustered)

    # Print Comments
    print_comments()


if __name__ == "__main__":
    main()
