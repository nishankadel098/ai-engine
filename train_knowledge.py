import json
import os
from sentence_transformers import SentenceTransformer

print("--- a ---")

json_path = 'data/files.json'

if not os.path.exists(json_path):
    print(f"❌ Error: {json_path} nahi mn.")
    exit()

with open(json_path, 'r') as f:
    rows = json.load(f)

print(f"✅ {len(rows)}  Processing...")


print("Loading AI Model (wait)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ AI Model Loaded!")

knowledge_base = []


for i, row in enumerate(rows):
    
    subject = row.get('subject_name')
    sem = row.get('semester')
    
    if subject:
        print(f"[{i+1}/{len(rows)}] Vectorizing: {subject} (Sem {sem})")
        vector = model.encode(subject).tolist()
        knowledge_base.append({
            "subject": subject,
            "semester": sem,
            "vector": vector
        })


with open("data/knowledge_base.json", "w") as f:
    json.dump(knowledge_base, f)

print("\n--- ✅ SUCCESS ---")
print("AI Knowledge Base : data/knowledge_base.json")