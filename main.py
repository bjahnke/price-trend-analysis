import os
from flask import Flask
from sqlalchemy import URL
import src.trend_analysis.scanner_engine
import env

app = Flask(__name__)

connection_string = URL.create(
    'postgresql',
    username='bjahnke71',
    password=os.environ.get('NEON_DB_PASSWORD'),
    host='ep-spring-tooth-474112.us-east-2.aws.neon.tech',
    database='historical-stock-data',
)

@app.route('/', methods=['POST'])
def handle_request():
    """
    This is the main function that will be called by the cloud function.
    :return:
    """
    connection_string = URL.create(
        'postgresql',
        username='bjahnke71',
        password=os.environ.get('NEON_DB_PASSWORD'),
        host='ep-spring-tooth-474112.us-east-2.aws.neon.tech',
        database='historical-stock-data',
    )
    src.trend_analysis.scanner_engine.main(connection_string, multiprocess=False)
    print('Service executed with no errors')
    return 'Service executed with no errors'


if __name__ == '__main__':
    src.trend_analysis.scanner_engine.main(connection_string, multiprocess=False)
    # app.run(host='0.0.0.0', port=8080)
