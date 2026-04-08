import json
import os
from sentence_transformers import SentenceTransformer

print("--- AI Training Started ---")

json_path = 'data/files.json'

if not os.path.exists(json_path):
    print(f"❌ Error: {json_path} nahi mili!")
    exit()

# 1. JSON Load logic (Special for PHPMyAdmin structure)
with open(json_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

rows = []
try:
    # PHPMyAdmin structure: List ka pehla element check karo
    # Usme 'data' naam ki key hogi jisme saari rows hain
    for entry in raw_data:
        if 'data' in entry:
            rows = entry['data']
            break
    
    if not rows:
        print("❌ Error: JSON mein 'data' key nahi mili. Structure check karein.")
        exit()
        
    print(f"✅ {len(rows)} Real items found! Training shuru ho rahi hai...")
except Exception as e:
    print(f"❌ Error parsing JSON: {e}")
    exit()

# 2. Model Loading
print("Loading AI Model (wait)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ AI Model Loaded!")

knowledge_base = []

# 3. Processing Loop
for i, row in enumerate(rows):
    # Aapke JSON ke hisaab se exact keys: subject_name, title, file_url
    subject = row.get('subject_name')
    title = row.get('title', '')
    url = row.get('file_url', '#')
    sem = row.get('semester', 'N/A')

    if subject:
        # Subject + Title ko combine kar rahe hain (e.g. "Applied Physics PYQ-2025")
        full_text = f"{subject} {title}".strip()
        print(f"[{i+1}/{len(rows)}] Vectorizing: {full_text}")
        
        vector = model.encode(full_text).tolist()
        
        knowledge_base.append({
            "subject": subject,
            "title": title,
            "semester": sem,
            "url": url,
            "vector": vector
        })

# 4. Save
with open("data/knowledge_base.json", "w", encoding='utf-8') as f:
    json.dump(knowledge_base, f)

print(f"\n--- ✅ SUCCESS ---")
print(f"Total {len(knowledge_base)} items saved in knowledge_base.json")