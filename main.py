from flask import Flask, Response, send_file, jsonify
from StringIO import StringIO
from network.mnist import MnistNetwork, digit

app = Flask(__name__)

@app.route("/")
def hello():
  return "Hello, Flask!"

@app.route("/mnist")
def mnist():
  network = MnistNetwork()
  id, prediction = network.predict()
  response = """
    <p>The neural network thinks <img src="image/%s"/> is %s.</p>
  """ % (id, prediction)
  return response

@app.route("/image/<int:id>", methods = ["GET"])
def getImage(id):
  image = digit(id)
  io = StringIO()
  image.save(io, 'PNG')
  io.seek(0)
  return send_file(io, mimetype = "image/png")

@app.route("/names/<string:name>")
def getName(name):
  data = { 'name': name }
  response = jsonify(data)
  response.status_code = 200
  response.headers['Link'] = 'http://www.backpasture.net'
  return response

if __name__ == "__main__":
  app.run(host = "0.0.0.0", port = 3005, debug = True)
