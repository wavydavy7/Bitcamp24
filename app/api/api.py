from flask import Flask
from datetime import datetime
app = Flask(__name__)

@app.route('/api/ml')
def predict():
    return {'accuracy': datetime.now()}
