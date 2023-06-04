from flask import Flask, url_for, request, Response
import lancedb
from fashion_clip.fashion_clip import FashionCLIP
import pandas as pd
import os

app = Flask(__name__)
print("Loading DB")
uri = "./lancedb"
db = lancedb.connect(uri)
tbl = db.open_table("styledb")
print("Loading Model")
fclip = FashionCLIP('fashion-clip')

@app.route('/recommend', methods=['GET'])
def get_recommendations():
  args = request.args
  description = args.get("text_description")
  num_results = int(args.get("num_results"))
  #description = request.json.get('text_description')
  #num_results = request.json.get('num_results')
  #if num_results is None:
  #  num_results = 5
  print("Commencing Search")
  embedding = fclip.encode_text([description], batch_size=32)[0]
  result_df = tbl.search(embedding).limit(num_results).to_df().sort_values(by='score', ascending=False)
  itm_lst = []
  for i in range(result_df.shape[0]):
    itm_lst.append({'filename': url_for('static',filename=result_df.iloc[i]['filename']),
                    'description': result_df.iloc[i]['detailed_desc'],
                    'score': float(result_df.iloc[i]['score'])})
  response = {}

  return {'item_count': result_df.shape[0],
           'items': itm_lst}

@app.route("/", methods=['GET']):
def index():
   return "Hello"

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=7000)
