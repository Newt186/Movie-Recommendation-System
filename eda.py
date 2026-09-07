import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(data_path="data/movies.csv"):

    movies = pd.read_csv(data_path)

    print("=" * 50)
    print("MOVIE DATASET EDA")
    print("=" * 50)

    # Dataset shape
    print("\nDataset Shape:")
    print(movies.shape)

    # First records
    print("\nFirst 5 Records:")
    print(movies.head())

    # Dataset information
    print("\nDataset Information:")
    print(movies.info())

    # Missing values
    print("\nMissing Values:")
    print(movies.isnull().sum())

    # Duplicate values
    print("\nDuplicate Rows:")
    print(movies.duplicated().sum())

    # Number of movies
    print("\nNumber of Movies:")
    print(movies["movieId"].nunique())

    # Genre analysis
    movies["genres"] = movies["genres"].fillna("")

    genre_counts = {}

    for genres in movies["genres"]:
        for genre in genres.split("|"):
            if genre and genre != "(no genres listed)":
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

    genre_df = pd.DataFrame(
        list(genre_counts.items()),
        columns=["Genre", "Count"]
    ).sort_values(
        "Count",
        ascending=False
    )

    print("\nTop Genres:")
    print(genre_df.head(10))

    # Plot
    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=genre_df.head(10),
        x="Count",
        y="Genre",
        color="royalblue"
    )

    plt.title("Top 10 Movie Genres")
    plt.xlabel("Number of Movies")
    plt.ylabel("Genre")

    plt.tight_layout()

    plt.savefig("top_genres.png")

    plt.show()


if __name__ == "__main__":
    perform_eda()
