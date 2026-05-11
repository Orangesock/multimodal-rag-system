"""
Advanced RAG Evaluation Suite
Upgraded with "LLM-as-a-Judge" Semantic Hit Rate Scoring
"""

import sys
from pathlib import Path

# Ensure the script can find your src/ folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from multimodal_rag.resources import get_resources
from multimodal_rag.generation.rag_pipeline import rag_query

# ==================================
# 1. PASTE GENERATED QUESTIONS HERE
# ==================================
EVAL_DATA = [
    {
        'query': 'What is the name of the verse in the Quran that describes the state of disbelief?',
        'relevant_chunks': [331],
        'expected_answer_keywords': []
    },
    {
        'query': 'What is the framework for a human trafficking protocol in healthcare settings?',
        'relevant_chunks': [60],
        'expected_answer_keywords': []
    },
    {
        'query': 'What is the name of the last prophet sent by God?',
        'relevant_chunks': [643],
        'expected_answer_keywords': []
    },
    {
        'query': 'What did Amr Ibn Al-Hareth say about the Prophet Muhammad?',
        'relevant_chunks': [505],
        'expected_answer_keywords': []
    },
    {
        'query': 'What is the name of the first prophet in Islam?',
        'relevant_chunks': [806],
        'expected_answer_keywords': []
    },
    {
        'query': 'What did Aristotle say about thunder and lightning?',
        'relevant_chunks': [387],
        'expected_answer_keywords': []
    },
    {
        'query': 'What is the name of the Day of Judgment mentioned in the Quran?',
        'relevant_chunks': [566],
        'expected_answer_keywords': []
    },
    {
        'query': 'What is the name of the book that lists the stories of people who have converted to Islam?',
        'relevant_chunks': [593],
        'expected_answer_keywords': []
    },
    {
        'query': 'What language was the Quran revealed in?',
        'relevant_chunks': [454],
        'expected_answer_keywords': []
    },
    {
        'query': 'What was the synthesis of Eastern and Western ideas that brought about great advances in medicine, mathematics, physics?',
        'relevant_chunks': [747],
        'expected_answer_keywords': []
    },
]

def main():
    if not EVAL_DATA:
        print("❌ ERROR: You need to paste your 10 questions into the EVAL_DATA list first!")
        return

    print("🚀 Initializing Pipeline and LLM Judge...")
    resources = get_resources()
    
    # Load the LLM to act as our Judge
    generator = resources.get_generator_model()

    total_queries = len(EVAL_DATA)
    exact_hits = 0
    semantic_hits = 0

    print(f"📊 Starting Evaluation on {total_queries} Queries...\n")
    print("-" * 50)

    for idx, item in enumerate(EVAL_DATA, 1):
        query = item['query']
        expected_chunks = item['relevant_chunks']
        
        print(f"🔍 Q{idx}: {query}")
        
        # 1. RETRIEVAL PHASE (Using your actual production pipeline!)
        try:
            pipeline_result = rag_query(query)
            # Grab the reranked or retrieved chunks, take the top 3
            retrieved_results = pipeline_result.get("reranked") or pipeline_result.get("retrieved") or []
            retrieved_results = retrieved_results[:3]
            
            # Extract chunk IDs safely
            retrieved_ids = [res.get('chunk_index', res.get('id', -1)) for res in retrieved_results]
        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
            continue
        
        # 2. EXACT MATCH GRADING (The Old, Flawed Way)
        exact_match = any(eid in expected_chunks for eid in retrieved_ids)
        if exact_match:
            exact_hits += 1
            
        # 3. LLM-AS-A-JUDGE GRADING (The Professional Way)
        semantic_match = False
        
        for res in retrieved_results:
            chunk_text = res.get('text', '')
            
            judge_prompt = (
                "Task: Evaluate if the Text contains the exact factual answer to the Question.\n"
                f"Question: {query}\n"
                f"Text: {chunk_text[:600]}\n"
                "Does the Text contain the answer? Reply strictly with YES or NO.\n"
                "Answer:"
            )
            
            try:
                output = generator(judge_prompt, max_new_tokens=5, do_sample=False, temperature=0.1)
                judgment = output[0]['generated_text'].strip().upper()
                
                if "YES" in judgment:
                    semantic_match = True
                    break 
            except Exception as e:
                print(f"⚠️ Judge LLM error: {e}")
                continue
                
        if semantic_match:
            semantic_hits += 1
            print("   ✅ Semantic Judge: PASS (Correct information retrieved)")
        else:
            print("   ❌ Semantic Judge: FAIL (Information missing from chunks)")
            
        print("-" * 50)

    # 4. FINAL CALCULATION
    exact_hit_rate = exact_hits / total_queries
    semantic_hit_rate = semantic_hits / total_queries

    print("\n" + "="*40)
    print(" 🏆 FINAL EVALUATION METRICS 🏆")
    print("="*40)
    print(f" Exact Match Hit Rate: {exact_hit_rate:.3f} (The Flawed Metric)")
    print(f" Semantic Hit Rate:    {semantic_hit_rate:.3f} (The True Metric)")
    print("="*40)

if __name__ == "__main__":
    main()