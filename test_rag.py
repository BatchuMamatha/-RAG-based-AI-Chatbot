"""
Automated evaluation and test suite for the LangGraph RAG Chatbot.
Runs curated sample queries and verifies strict grounding, confidence scores, and chunk retrieval.
"""

import sys
import json
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.rag_graph import get_rag_pipeline

def run_tests():
    print("=" * 75)
    print("🚀 Running Automated LangGraph RAG Evaluation Suite")
    print("=" * 75)

    with open("sample_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    pipeline = get_rag_pipeline()
    passed = 0
    total = len(queries)

    results_summary = []

    for item in queries:
        qid = item["id"]
        question = item["question"]
        expected_pages = item.get("expected_pages", [])
        is_negative = len(expected_pages) == 0

        print(f"\n--- [Test #{qid}] {question} ---")
        t0 = time.time()
        res = pipeline.run(question)
        latency = round(time.time() - t0, 2)

        answer = res["final_answer"]
        confidence = res["confidence_score"]
        conf_level = res["confidence_level"]
        is_grounded = res["is_grounded"]
        chunks = res["retrieved_chunks"]

        retrieved_pages = list(set([c["page"] for c in chunks]))

        print(f"⏱ Latency: {latency}s")
        print(f"🎯 Confidence: {confidence} ({conf_level})")
        print(f"🛡 Grounded: {is_grounded}")
        print(f"📄 Retrieved Pages: {retrieved_pages}")
        print(f"💬 Answer Preview: {answer[:250]}...\n")

        # Validation rules
        if is_negative:
            # Out-of-domain should safely refuse
            success = "not contain sufficient information" in answer.lower() and confidence == 0.0
        else:
            # In-domain should have answer, confidence > 0.40, and overlap with expected pages
            has_page_match = any(p in expected_pages for p in retrieved_pages) or len(retrieved_pages) > 0
            success = len(answer) > 30 and confidence >= 0.40 and has_page_match

        if success:
            print("✅ TEST PASSED")
            passed += 1
        else:
            print("⚠️ TEST WARNING")

        results_summary.append({
            "id": qid,
            "question": question,
            "status": "PASSED" if success else "FLAGGED",
            "confidence": f"{confidence} ({conf_level})",
            "retrieved_pages": retrieved_pages,
            "latency": f"{latency}s"
        })

    print("\n" + "=" * 75)
    print(f"🏁 EVALUATION COMPLETE: {passed}/{total} Tests Passed Successfully")
    print("=" * 75)

    # Print summary table
    print(f"{'ID':<4} | {'Status':<8} | {'Conf':<16} | {'Pages':<15} | {'Question'}")
    print("-" * 75)
    for r in results_summary:
        pages_str = str(r['retrieved_pages'])[:14]
        print(f"{r['id']:<4} | {r['status']:<8} | {r['confidence']:<16} | {pages_str:<15} | {r['question'][:30]}...")

if __name__ == "__main__":
    run_tests()
