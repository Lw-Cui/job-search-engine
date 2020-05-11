import traceback

from flask import Flask, jsonify, request
from flask_cors import CORS
from computation import tfidf, bm25f, glove
from google.cloud import storage
from io import StringIO

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

CORS(app)


@app.route('/_ah/warmup')
def warmup():
    return '', 200, {}


def get_query():
    if request.method == "POST":
        json = request.get_json()
        print(json)
        return json.get('query')
    else:
        print(request.args.get("query"))
        return request.args.get("query")


@app.route('/bm25', methods=['GET', 'POST'])
def bm25_query():
    try:
        return jsonify(bm25f.query(get_query()))
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Internal Error"}), 500


@app.route('/tfidf', methods=['GET', 'POST'])
def tfidf_query():
    try:
        return jsonify(tfidf.query(get_query()))
    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Internal Error"}), 500


@app.route('/glove', methods=['GET', 'POST'])
def glove_query():
    try:
        return jsonify(glove.query(get_query()))
    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Internal Error"}), 500


def setup_app():
    tfidf.init('data/data.csv')
    bm25f.init('data/data.csv')
    glove.init('data/data.csv')
    print("finish read file")
    try:
        import googleclouddebugger
        googleclouddebugger.enable()
    except ImportError:
        pass


setup_app()

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
