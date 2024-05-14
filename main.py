from flask import Flask
import src.trend_analysis.scanner_engine

app = Flask(__name__)


@app.route('/', methods=['POST'])
def handle_request():
    """
    This is the main function that will be called by the cloud function.
    :return:
    """
    src.trend_analysis.scanner_engine.main(multiprocess=True)
    print('Service executed with no errors')
    return 'Service executed with no errors'


if __name__ == '__main__':
    handle_request()
    app.run(host='0.0.0.0', port=8080)
