from flask import Flask, request, jsonify, render_template_string

from recommender import MovieRecommender


app = Flask(__name__)

# Load recommender
recommender = MovieRecommender("data/movies.csv")
recommender.load_data()
recommender.train()


HTML_PAGE = """
<!DOCTYPE html>

<html>

<head>

    <title>Movie Recommendation System</title>

    <style>

        body {
            font-family: Arial, sans-serif;
            background: #141414;
            color: white;
            text-align: center;
            padding: 40px;
        }

        h1 {
            color: #e50914;
        }

        input {
            width: 350px;
            padding: 12px;
            border-radius: 5px;
            border: none;
            font-size: 16px;
        }

        button {
            padding: 12px 20px;
            margin-left: 5px;
            border: none;
            border-radius: 5px;
            background: #e50914;
            color: white;
            cursor: pointer;
        }

        button:hover {
            background: #b20710;
        }

        .movie {
            background: #222;
            margin: 15px auto;
            padding: 15px;
            max-width: 600px;
            border-radius: 8px;
        }

        .movie h3 {
            color: #ffcc00;
        }

    </style>

</head>

<body>

    <h1>🎬 Movie Recommendation System</h1>

    <p>Enter a movie name to get similar movies.</p>

    <form method="GET">

        <input
            type="text"
            name="movie"
            placeholder="Example: Toy Story"
            value="{{ movie }}"
            required
        >

        <button type="submit">
            Recommend
        </button>

    </form>

    {% if recommendations %}

        <h2>Recommended Movies</h2>

        {% for movie in recommendations %}

            <div class="movie">

                <h3>{{ movie.title }}</h3>

                <p>
                    Genres: {{ movie.genres }}
                </p>

                <p>
                    Similarity:
                    {{ "%.2f"|format(movie.similarity) }}
                </p>

            </div>

        {% endfor %}

    {% elif movie %}

        <p>No movie found.</p>

    {% endif %}

</body>

</html>
"""


@app.route("/")
def home():

    movie = request.args.get("movie", "")

    recommendations = []

    if movie:

        result = recommender.recommend(
            movie,
            number_of_recommendations=10
        )

        recommendations = result.to_dict(
            orient="records"
        )

    return render_template_string(
        HTML_PAGE,
        movie=movie,
        recommendations=recommendations
    )


@app.route("/api/recommend")
def api_recommend():

    movie = request.args.get("movie")

    if not movie:

        return jsonify({
            "error": "Please provide a movie name."
        }), 400

    result = recommender.recommend(
        movie,
        number_of_recommendations=10
    )

    if result.empty:

        return jsonify({
            "error": "Movie not found."
        }), 404

    return jsonify(
        result.to_dict(orient="records")
    )


if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
