import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """
    Content-based Movie Recommendation System.

    Uses movie genres as text features and TF-IDF + cosine similarity
    to recommend movies similar to a selected movie.
    """

    def __init__(self, data_path="data/movies.csv"):
        self.data_path = data_path
        self.movies = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.indices = None

    def load_data(self):
        """Load and clean the MovieLens movie dataset."""

        self.movies = pd.read_csv(self.data_path)

        # Remove duplicate movies
        self.movies.drop_duplicates(subset="movieId", inplace=True)

        # Remove missing titles
        self.movies.dropna(subset=["title"], inplace=True)

        # Fill missing genres
        self.movies["genres"] = self.movies["genres"].fillna("")

        # Clean title
        self.movies["title"] = self.movies["title"].apply(self.clean_title)

        # Convert genres into text
        self.movies["genres"] = (
            self.movies["genres"]
            .str.replace("|", " ", regex=False)
            .str.replace("(no genres listed)", "", regex=False)
        )

        self.movies.reset_index(drop=True, inplace=True)

        return self.movies

    @staticmethod
    def clean_title(title):
        """Remove year from movie title."""

        title = re.sub(r"\(\d{4}\)", "", title)
        title = re.sub(r"\s+", " ", title)

        return title.strip()

    def train(self):
        """Create TF-IDF vectors and cosine similarity matrix."""

        if self.movies is None:
            self.load_data()

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = vectorizer.fit_transform(
            self.movies["genres"]
        )

        self.similarity_matrix = cosine_similarity(
            self.tfidf_matrix,
            self.tfidf_matrix
        )

        self.indices = pd.Series(
            self.movies.index,
            index=self.movies["title"].str.lower()
        ).drop_duplicates()

    def recommend(self, movie_title, number_of_recommendations=10):
        """
        Recommend movies similar to the selected movie.
        """

        if self.similarity_matrix is None:
            self.train()

        movie_title = movie_title.lower().strip()

        # Exact match
        if movie_title not in self.indices:
            # Try partial matching
            matches = self.movies[
                self.movies["title"]
                .str.lower()
                .str.contains(movie_title, na=False)
            ]

            if matches.empty:
                return pd.DataFrame(
                    columns=["title", "genres", "similarity"]
                )

            movie_index = matches.index[0]

        else:
            movie_index = self.indices[movie_title]

            # In case duplicate titles return multiple indices
            if isinstance(movie_index, pd.Series):
                movie_index = movie_index.iloc[0]

        similarity_scores = list(
            enumerate(self.similarity_matrix[movie_index])
        )

        # Sort by similarity score
        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        # Remove the selected movie itself
        similarity_scores = [
            item for item in similarity_scores
            if item[0] != movie_index
        ]

        top_movies = similarity_scores[
            :number_of_recommendations
        ]

        movie_indices = [item[0] for item in top_movies]
        scores = [item[1] for item in top_movies]

        recommendations = self.movies.iloc[
            movie_indices
        ][["title", "genres"]].copy()

        recommendations["similarity"] = scores

        return recommendations.reset_index(drop=True)


if __name__ == "__main__":

    recommender = MovieRecommender("data/movies.csv")

    recommender.load_data()
    recommender.train()

    movie = input("Enter a movie name: ")

    recommendations = recommender.recommend(
        movie,
        number_of_recommendations=10
    )

    if recommendations.empty:
        print("\nMovie not found.")
    else:
        print("\nRecommended Movies:\n")

        for i, row in recommendations.iterrows():
            print(
                f"{i + 1}. {row['title']} "
                f"({row['similarity']:.2f})"
            )
