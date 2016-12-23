from flask import Flask, Response, send_file, jsonify
from StringIO import StringIO
from network.mnist import MnistNetwork, digit

import base64
import cStringIO

app = Flask(__name__)

@app.route("/")
def root():
  return "Hello, Flask!\n"

@app.route("/mnist/random")
def mnist():
  network = MnistNetwork()
  id, prediction = network.predict()
  response = """
    <p>The neural network thinks <img src="image/%s"/> is %s.</p>
  """ % (id, prediction)
  return response

@app.route("/mnist/<int:id>.json", methods = ["GET"])
def getMnistJson(id):
  network = MnistNetwork()
  id, prediction = network.predict(id)
  data = {
    'id': id,
    'prediction': prediction,
    'url': "/mnist/image/%s" % id,
  }
  return jsonify(data)

@app.route("/mnist/image/<int:id>.json")
def getMnistImageJson(id):
  image = digit(id)
  buffer = cStringIO.StringIO()
  image.save(buffer, format="PNG")
  encoded_image = base64.b64encode(buffer.getvalue())
  data = {
    'id': id,
    'image': encoded_image
  }
  return jsonify(data)

@app.route("/mnist/image/<int:id>", methods = ["GET"])
def getMnistImage(id):
  image = digit(id)
  io = StringIO()
  image.save(io, 'PNG')
  io.seek(0)
  return send_file(io, mimetype = "image/png")

if __name__ == "__main__":
  app.run(host = "0.0.0.0", port = 3002, debug = True)
