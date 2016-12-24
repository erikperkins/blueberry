from flask import Flask, Response, send_file, jsonify, request
from StringIO import StringIO
from network.mnist import MnistNetwork, digit, mnist_array, mnist_input


import json
from PIL import Image

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

  input = mnist_input(id)
  id, prediction = network.predict(input)

  image = digit(id)
  buffer = cStringIO.StringIO()
  image.save(buffer, format="PNG")
  encoded_image = base64.b64encode(buffer.getvalue())

  data = {
    'id': id,
    'prediction': prediction,
    'url': "/mnist/image/%s" % id,
    'image': encoded_image
  }
  return jsonify(data)

#######################

@app.route("/mnist/classify.json", methods = ["GET"])
def postMnistJson():
  network = MnistNetwork()

  image_json = request.args.get('image')
  image_ascii = json.dumps(image_json).encode('ascii')
  blob = base64.urlsafe_b64decode(image_ascii)

  input = mnist_array(blob)
  id, result = network.predict(input)

  data = {
    "result": result
  }
  return jsonify(data)

#########################

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
