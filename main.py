from flask import Flask, request, jsonify
import pandas as pd
import spacy

app = Flask(__name__)

nlp = spacy.load("en_core_web_md")

df = pd.read_excel("hackdata.xlsx")


def semantic_search(question, dataframe, nlp_model):
    question_doc = nlp_model(question)
    dataframe["similarity_score"] = dataframe["Abstract"].apply(
        lambda x: question_doc.similarity(nlp_model(x))
    )
    result_df = dataframe.sort_values(by="similarity_score", ascending=False)
    return result_df[["Abstract", "similarity_score"]].to_dict(orient="records")


@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    result = semantic_search(question, df, nlp)

    return jsonify({"results": result[:5]})


if __name__ == "__main__":
    app.run(debug=True)
