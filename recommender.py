import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """Recommend movies based on their genres."""

    def __init__(self, data_path="data/movies.csv"):
        self.data_path = data_path
        self.movies = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.movie_indices = None

    def load_data(self):
        """Read the movie dataset and clean the data."""

        self.movies = pd.read_csv(self.data_path)

        # Remove duplicate movie records
        self.movies.drop_duplicates(
            subset="movieId",
            inplace=True
        )

        # Movies without a title are not useful for recommendations
        self.movies.dropna(
            subset=["title"],
            inplace=True
        )

        # Replace missing genres with an empty string
        self.movies["genres"] = self.movies["genres"].fillna("")

        # Remove the release year from movie titles
        self.movies["title"] = self.movies["title"].apply(
            self.clean_title
        )

        # Convert "|" between genres into spaces
        self.movies["genres"] = (
            self.movies["genres"]
            .str.replace("|", " ", regex=False)
            .str.replace(
                "(no genres listed)",
                "",
                regex=False
            )
        )

        self.movies.reset_index(drop=True, inplace=True)

        return self.movies

    @staticmethod
    def clean_title(title):
        """Remove the year from a movie title."""

        title = re.sub(r"\(\d{4}\)", "", title)
        title = re.sub(r"\s+", " ", title)

        return title.strip()

    def train_model(self):
        """Convert genres into vectors and calculate similarities."""

        if self.movies is None:
            self.load_data()

        # Convert movie genres into TF-IDF features
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = vectorizer.fit_transform(
            self.movies["genres"]
        )

        # Compare every movie with every other movie
        self.similarity_matrix = cosine_similarity(
            self.tfidf_matrix
        )

        # Store movie titles with their dataframe index
        self.movie_indices = pd.Series(
            self.movies.index,
            index=self.movies["title"].str.lower()
        ).drop_duplicates()

    def recommend(self, movie_title, number_of_recommendations=10):
        """Return movies similar to the movie entered by the user."""

        if self.similarity_matrix is None:
            self.train_model()

        movie_title = movie_title.lower().strip()

        # Look for an exact movie title first
        if movie_title in self.movie_indices:
            movie_index = self.movie_indices[movie_title]

            # Handle duplicate movie titles
            if isinstance(movie_index, pd.Series):
                movie_index = movie_index.iloc[0]

        else:
            # If there is no exact match, try a partial match
            matching_movies = self.movies[
                self.movies["title"]
                .str.lower()
                .str.contains(movie_title, na=False)
            ]

            if matching_movies.empty:
                return pd.DataFrame(
                    columns=["title", "genres", "similarity"]
                )

            movie_index = matching_movies.index[0]

        # Get similarity scores for the selected movie
        scores = list(
            enumerate(
                self.similarity_matrix[movie_index]
            )
        )

        # Put the most similar movies first
        scores.sort(
            key=lambda item: item[1],
            reverse=True
        )

        # Don't recommend the movie the user already selected
        scores = [
            item for item in scores
            if item[0] != movie_index
        ]

        # Select the requested number of recommendations
        top_movies = scores[
            :number_of_recommendations
        ]

        movie_indices = [
            item[0]
            for item in top_movies
        ]

        similarity_scores = [
            item[1]
            for item in top_movies
        ]

        recommendations = self.movies.iloc[
            movie_indices
        ][["title", "genres"]].copy()

        recommendations["similarity"] = similarity_scores

        return recommendations.reset_index(drop=True)


def main():
    """Run the movie recommendation program."""

    recommender = MovieRecommender()

    print("Loading movie data...")
    recommender.load_data()

    print("Preparing recommendation model...")
    recommender.train_model()

    print("\n" + "=" * 45)
    print("       MOVIE RECOMMENDATION SYSTEM")
    print("=" * 45)

    movie_name = input(
        "\nEnter a movie name: "
    ).strip()

    if not movie_name:
        print("Please enter a movie name.")
        return

    recommendations = recommender.recommend(
        movie_name,
        number_of_recommendations=10
    )

    if recommendations.empty:
        print(
            f"\nSorry, I couldn't find a movie "
            f"matching '{movie_name}'."
        )
        return

    print(
        f"\nMovies similar to '{movie_name}':\n"
    )

    for number, (_, movie) in enumerate(
        recommendations.iterrows(),
        start=1
    ):
        print(
            f"{number}. {movie['title']} "
            f"- Similarity: {movie['similarity']:.2f}"
        )


if __name__ == "__main__":
    main()
