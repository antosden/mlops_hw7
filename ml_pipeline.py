from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X, y = iris.data, iris.target

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok"
            }).encode())

        elif self.path == "/predict":
            sample = np.array([X[0]])
            pred = model.predict(sample)[0]

            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({
                "prediction": int(pred)
            }).encode())

PORT = 8000
HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()