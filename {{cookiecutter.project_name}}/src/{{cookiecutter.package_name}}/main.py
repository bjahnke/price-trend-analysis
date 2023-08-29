from flask import Flask

app = Flask(__name__)


@app.route('/', methods=['POST'])
def handle_request():
    """
    This is the main function that will be called by the cloud function.
    :return:
    """
    return 'Service executed with no errors'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)