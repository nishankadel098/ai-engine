from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import json
import torch
import os

app = Flask(__name__)

# CORS setup: React (port 3000) ko allow karne ke liye
CORS(app, resources={r"/*": {"origins": "*"}})

# 1. AI Model aur Knowledge Load karein
print("--- Starting AI Engine ---")
print("Loading AI Brain (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge base load karne ka safer tareeka
KB_PATH = os.path.join("data", "knowledge_base.json")

try:
    if os.path.exists(KB_PATH):
        with open(KB_PATH, "r") as f:
            knowledge_base = json.load(f)
        print(f"✅ Knowledge Base Loaded! ({len(knowledge_base)} items)")
    else:
        knowledge_base = []
        print("⚠️ Warning: knowledge_base.json nahi mili. Pehle train_knowledge.py chalayein.")
except Exception as e:
    print(f"❌ Error loading JSON: {e}")
    knowledge_base = []

@app.route('/ask', methods=['POST', 'OPTIONS'])
def ask_ai():
    # CORS Pre-flight request handle karne ke liye
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    # Request data nikalna
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"answer": "Kuch toh pucho! (Empty message)"}), 400

    user_query = data.get("message")
    print(f"User Query: {user_query}")

    if not knowledge_base:
        return jsonify({"answer": "Mere paas abhi koi knowledge nahi hai. Pehle files upload karein."})

    # 2. User ki query ko vector mein badalna
    query_vector = model.encode(user_query, convert_to_tensor=True)
    
    # 3. Knowledge Base mein dhoondhna
    best_match = None
    highest_score = -1

    for item in knowledge_base:
        # Vector ko tensor mein badalna similarity check ke liye
        item_vector = torch.tensor(item['vector'])
        score = util.cos_sim(query_vector, item_vector).item()
        
        if score > highest_score:
            highest_score = score
            best_match = item

    # 4. Result dikhana (Confidence threshold 0.35 - 0.40 best hota hai)
    if highest_score > 0.35:
        return jsonify({
            "answer": f"Zaroor! Mujhe '{best_match['subject']}' ke resources mile hain.",
            "file_url": best_match.get('url', '#'),
            "semester": best_match.get('semester', 'N/A'),
            "confidence": round(highest_score * 100, 2)
        })
    else:
        return jsonify({
            "answer": "Maaf kijiye, mujhe aapke subject se milta-julta kuch nahi mila. Kya aap subject ka naam sahi se likh sakte hain?",
            "confidence": round(highest_score * 100, 2)
        })

@app.route('/', methods=['GET'])
def home():
    return "AI Server is Running! React se /ask par POST request bhejein."

if __name__ == '__main__':
    # debug=True development ke liye sahi hai
    app.run(host='0.0.0.0', port=5000, debug=True)