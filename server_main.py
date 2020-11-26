from flask import *
from flask_cors import CORS

from model.Server.network_server import Server

app = Flask(__name__)
CORS(app, supports_credentials=True)
print("Building Server")
server = Server()
print("Building Finish")


@app.after_request
def af_request(resp):
    """
    #请求钩子，在所有的请求发生后执行，加入headers。
    :param resp:
    :return:
    """
    resp = make_response(resp)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return resp


@app.route('/', methods=['POST'])
def upload():
    print("Get data")
    x = request.form['data']
    try:
        output = server.forward(x)
    except Exception as e:
        print(e)
    return jsonify({'message': 'success',
                    'output': output})

@app.route('/test', methods=['POST'])
def test():
    request_body = request.json
    print(request_body)
    return request_body

if __name__ == '__main__':
    app.run('0.0.0.0', port=8081)
