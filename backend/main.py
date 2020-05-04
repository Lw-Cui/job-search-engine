from flask import Flask, jsonify
from computation import tfidf, bm25f
from google.cloud import storage
from io import StringIO

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/tfidf', methods=['GET', 'POST'])
def query():
    #return jsonify(tfidf.query("compiler"))
    return jsonify(bm25f.query("compiler"))


def setup_app():
    #tfidf.init('amazon_jobs_dataset.csv')
    bm25f.init('amazon_jobs_dataset.csv')
    print("finish read file")


setup_app()

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
