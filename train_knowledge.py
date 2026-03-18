import json
import os
from sentence_transformers import SentenceTransformer

print("--- AI Offline Training Started ---")

# 1. JSON File Load karein
json_path = 'data/files.json'

if not os.path.exists(json_path):
    print(f"❌ Error: {json_path} nahi mili! phpMyAdmin se export karke yahan rakhein.")
    exit()

with open(json_path, 'r') as f:
    rows = json.load(f)

print(f"✅ {len(rows)} rows mil gayi hain. Processing...")

# 2. AI Model Load
print("Loading AI Model (Pehli baar hai toh wait karein)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ AI Model Loaded!")

knowledge_base = []

# 3. Vectorization logic
for i, row in enumerate(rows):
    # Check karein ki row mein subject_name hai ya nahi
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

# 4. Save to Knowledge Base
with open("data/knowledge_base.json", "w") as f:
    json.dump(knowledge_base, f)

print("\n--- ✅ SUCCESS ---")
print("AI Knowledge Base taiyar hai: data/knowledge_base.json")