from flask import Flask, jsonify
from flask_cors import CORS  
# from run_LaBSE_neighbor import result  
from run import result

app = Flask(__name__)
CORS(app)  

@app.route('/run-script', methods=['GET'])
def run_script():
    try:
        hit1, hit10, mrr = result()
        return jsonify({"Hit@1": hit1, "Hit@10": hit10, "MRR": mrr})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
