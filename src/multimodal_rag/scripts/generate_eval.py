"""
Synthetic Data Generator for RAG Evaluation.
"Trivia Flashcard" Prompting (Max 8 Words) for High-Precision Queries.
"""

import pickle
import random
import sys
from pathlib import Path

# Ensure the script can find your src/ folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from multimodal_rag.resources import get_resources

def generate_synthetic_data(num_samples=10):
    print(f"🔍 Generating {num_samples} Short-Form Trivia Questions...\n")
    
    meta_path = Path("embeddings/meta.pkl")
    if not meta_path.exists():
        print("❌ meta.pkl not found! Ensure your embeddings are in the embeddings/ folder.")
        return

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    
    # Filter for substantial chunks (at least 30 words) for context
    valid_chunks = [m for m in metadata if len(m.get("text", "").split()) > 30]
    samples = random.sample(valid_chunks, min(num_samples, len(valid_chunks)))
    
    resources = get_resources()
    generator = resources.get_generator_model()
    
    eval_data = []
    
    for idx, chunk in enumerate(samples):
        chunk_id = chunk["chunk_index"]
        text = chunk["text"]
        
        print(f"⏳ Generating question {idx+1}/{num_samples} (Reading Chunk #{chunk_id})...")

        # 3. THE TRIVIA PROMPT
        prompt = (
            "Task: Write a highly specific standalone trivia question based on the fact below.\n"
            "The trivia question must include specific details, names, dates, or subjects, and the question must be understandable to anyone without any additional context.\n\n"
            "Fact: The Associated Press report detailed the rise of labor exploitation in agricultural supply chains.\n"
            "Trivia Question: What did the Associated Press report detail?\n\n"
            "Fact: The names of Abraham's two sons are Ishmael and Isaac.\n"
            "Trivia Question: What are the names of Abraham's two sons?\n\n"
            f"Fact: {text[:450]}\n"
            "Trivia Question:"
        )
        
        try:
            # Setting temperature to 0.1 forces the model to be extremely literal
            output = generator(prompt, max_new_tokens=20, do_sample=False, temperature=0.1)
            question = output[0]["generated_text"].strip()
            
            # Clean up: Force short length, remove artifacts
            question = question.split('\n')[0].strip() 
            question = question.replace("Trivia Question:", "").strip()
            
            if not question.endswith("?"):
                question += "?"
                
        except Exception as e:
            print(f"Generation failed for chunk {chunk_id}: {e}")
            continue
        
        eval_data.append({
            "query": question,
            "relevant_chunks": [chunk_id],
            "expected_answer_keywords": []
        })
        
    print("\n✅ DONE! === COPY AND PASTE THIS LIST INTO evaluate.py ===\n")
    print("EVAL_DATA = [")
    for item in eval_data:
        safe_query = item['query'].replace("'", "")
        print(f"    {{")
        print(f"        'query': '{safe_query}',")
        print(f"        'relevant_chunks': {item['relevant_chunks']},")
        print(f"        'expected_answer_keywords': []")
        print(f"    }},")
    print("]\n")

if __name__ == "__main__":
    generate_synthetic_data(10)