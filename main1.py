from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

df = pd.read_excel("hackdata.xlsx")

stop_words = set(stopwords.words("english"))


def process_text(text):
    tokens = word_tokenize(text)
    tokens = [
        word.lower()
        for word in tokens
        if word.isalpha() and word.lower() not in stop_words
    ]
    return " ".join(tokens)


df["processed_abstract"] = df["Abstract"].apply(process_text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["processed_abstract"])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    question_vector = vectorizer.transform([process_text(question)])
    df["similarity_score"] = cosine_similarity(tfidf_matrix, question_vector)

    # result_df = df.sort_values(by="similarity_score", ascending=False).head(5)
    result_df = df.sort_values(by="similarity_score", ascending=False)
    result_df = result_df.drop_duplicates(subset=["Abstract"])

    result_df = result_df[["Abstract", "Publication Year", "Title", "URL"]]

    result_df = result_df.head(5)

    result_list = result_df.to_dict(orient="records")

    return jsonify({"results": result_list})


if __name__ == "__main__":
    app.run(debug=True)
