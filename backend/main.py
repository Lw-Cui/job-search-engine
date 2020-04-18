from flask import Flask, jsonify
from computation import tfidf
from google.cloud import storage

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def query():
    return jsonify({'result': [1, 2, 3]})


def setup_app():
    storage_client = storage.Client()
    bucket = storage_client.bucket('en-601-666.appspot.com')
    blob = bucket.blob('tmp.csv')
    tfidf.init(blob.download_as_string())
    print("finish read file")


setup_app()

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
