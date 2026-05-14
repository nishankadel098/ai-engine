from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import json
import torch
import os
import random
import gc

# --- RAM OPTIMIZATION: STEP 1 (System Level) ---
# PyTorch ko unnecessary memory reserve karne se rokne ke liye
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

# CORS setup
CORS(app, resources={r"/*": {"origins": "*"}})

def memory_cleanup():
    """Manual garbage collection to free unused RAM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("--- Starting AI Engine (Extreme RAM Optimized) ---")
print("Loading AI Brain (all-MiniLM-L6-v2)...")

# --- RAM OPTIMIZATION: STEP 2 (Model Level) ---
# Model ko memory-efficient tarike se load karna
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
model.max_seq_length = 64  # Input length choti rakhne se RAM bachti hai
memory_cleanup()

# File Paths
KB_PATH = os.path.join("data", "knowledge_base.json")
QUESTIONS_PATH = os.path.join("data", "questions.json")

# Knowledge base load karein
try:
    if os.path.exists(KB_PATH):
        with open(KB_PATH, "r") as f:
            knowledge_base = json.load(f)
        print(f"✅ Knowledge Base Loaded! ({len(knowledge_base)} items)")
    else:
        knowledge_base = []
        print("⚠️ Warning: knowledge_base.json nahi mili.")
except Exception as e:
    print(f"❌ Error loading Knowledge Base: {e}")
    knowledge_base = []

# --- ROUTE 1: Chatbot Logic ---
@app.route('/ask', methods=['POST', 'OPTIONS'])
def ask_ai():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"answer": "Kuch toh pucho!"}), 400

    user_query = data.get("message")
    
    if not knowledge_base:
        return jsonify({"answer": "Mere paas abhi koi data nahi hai."})

    # AI Vector Matching (Optimized with inference_mode to save maximum RAM)
    with torch.inference_mode():
        query_vector = model.encode(user_query, convert_to_tensor=True)
        best_match = None
        highest_score = -1

        for item in knowledge_base:
            item_vector = torch.tensor(item['vector'])
            score = util.cos_sim(query_vector, item_vector).item()
            
            if score > highest_score:
                highest_score = score
                best_match = item
        
        # Cleanup vectors from memory
        del query_vector

    memory_cleanup()

    if highest_score > 0.35:
        return jsonify({
            "answer": f"Zaroor! Mujhe '{best_match['subject']}' ke resources mile hain.",
            "file_url": best_match.get('url', '#'),
            "semester": best_match.get('semester', 'N/A'),
            "confidence": round(highest_score * 100, 2)
        })
    else:
        return jsonify({
            "answer": "Maaf kijiye, mujhe aapke subject se milta-julta kuch nahi mila.",
            "confidence": round(highest_score * 100, 2)
        })

# --- ROUTE 2: Mock Test Logic ---
@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    data = request.get_json()
    branch = data.get('branch')
    sem = str(data.get('sem'))
    subject_input = data.get('subject', '').lower()

    if not os.path.exists(QUESTIONS_PATH):
        return jsonify({"error": "Questions database (questions.json) nahi mila!"}), 404

    try:
        with open(QUESTIONS_PATH, "r") as f:
            all_questions = json.load(f)

        branch_data = all_questions.get(branch, {})
        sem_data = branch_data.get(sem, {})

        if not sem_data:
            return jsonify({"error": f"{branch} Sem {sem} ke liye questions nahi mile."}), 404

        available_subjects = list(sem_data.keys())
        
        # AI Semantic Matching (Optimized)
        with torch.inference_mode():
            user_vec = model.encode(subject_input, convert_to_tensor=True)
            sub_vecs = model.encode(available_subjects, convert_to_tensor=True)
            similarities = util.cos_sim(user_vec, sub_vecs)[0]
            best_idx = torch.argmax(similarities).item()
            
            # Delete large tensors
            del user_vec
            del sub_vecs

        if similarities[best_idx] > 0.5:
            matched_subject = available_subjects[best_idx]
            questions_list = sem_data[matched_subject]
            selected_questions = random.sample(questions_list, min(len(questions_list), 10))
            
            memory_cleanup()
            return jsonify({
                "subject_detected": matched_subject,
                "questions": selected_questions
            })
        else:
            memory_cleanup()
            return jsonify({"error": "Subject match nahi hua. Sahi subject likhein."}), 404

    except Exception as e:
        memory_cleanup()
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "AI Server is Running! React se /ask (Chat) ya /generate-questions (Test) use karein."

if __name__ == '__main__':
    # Render ke liye port fetch karein
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)