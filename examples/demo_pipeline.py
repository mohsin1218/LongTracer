"""
LongTracer Demo — Parallel RAG Verification Pipeline.

This is a demo application, NOT part of the longtracer SDK package.

Usage:
    python demo_pipeline.py --ingest './Pdf-docs'
    python demo_pipeline.py --query 'What is Beta-CLIP?'
"""

import sys
import os
from pathlib import Path

# Add both examples/ and repo root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_dotenv():
    """Load .env file."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if not os.environ.get(key.strip()):
                        os.environ[key.strip()] = value.strip()

load_dotenv()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LongTracer - RAG Verification")
    parser.add_argument("--ingest", help="Path to PDF directory to ingest")
    parser.add_argument("--query", help="Question to ask")

    args = parser.parse_args()

    if args.ingest:
        from rag.pdf_parser import load_pdfs_from_directory
        from rag.chunker import chunk_documents
        from rag.embedder import get_embeddings
        from rag.store import VectorStore

        print(f"📂 Ingesting PDFs from: {args.ingest}")
        documents = load_pdfs_from_directory(args.ingest)
        print(f"  📄 Loaded {len(documents)} pages")

        if documents:
            chunks = chunk_documents(documents)
            embeddings = get_embeddings()
            store = VectorStore(embeddings)
            store.add_documents(chunks)
            print("✅ Ingestion complete!")
        return

    if args.query:
        from rag.embedder import get_embeddings
        from rag.store import VectorStore
        from rag.retriever import RAGRetriever
        from longtracer.guard.verifier import CitationVerifier
        from longtracer.guard.context_relevance import ContextRelevanceScorer
        from longtracer.guard.parallel_pipeline import ParallelPipeline
        from longtracer.guard.tracer import Tracer
        from longtracer.guard.trace_report import print_trace_report

        print("  📂 Loading existing vector store...")
        embeddings = get_embeddings()
        store = VectorStore(embeddings)

        print("\n🚀 Initializing Verification Models (Singleton)...")
        retriever = RAGRetriever(store)
        verifier = CitationVerifier()
        relevance_scorer = ContextRelevanceScorer(verbose=True)

        tracer = Tracer(run_name=f"longtracer_parallel: {args.query[:50]}...")
        tracer.start_root(inputs={"query": args.query, "mode": "parallel"})
        verifier.tracer = tracer

        print("\n⚡ PARALLEL MODE ENABLED - Optimized pipeline")
        print("="*70)
        print(f"❓ Question: {args.query}")
        print("="*70)

        print("  🔍 Retrieving + LLM + Relevance (PARALLEL)...")

        pipeline = ParallelPipeline(max_workers=4, tracer=tracer)
        result = pipeline.run(args.query, retriever, verifier, relevance_scorer)

        print(f"\n🤖 LLM ANSWER:")
        print(f"{'─'*70}")
        print(result["answer"])
        print(f"{'─'*70}")

        rel = result["relevance_result"]
        print(f"\n🎯 CONTEXT RELEVANCE (Score A - Bi-Encoder):")
        print(f"  Average Relevance: {rel.get('average_relevance', 0):.2%}")
        print(f"  Top Relevance: {rel.get('top_relevance', 0):.2%}")
        print(f"  Threshold Pass: {'✅ Yes' if rel.get('threshold_pass') else '❌ No'}")
        print(f"  Latency: {rel.get('latency_ms', 0):.0f}ms")

        print(f"\n  📋 Chunk Rankings:")
        for i, chunk in enumerate(rel.get("chunk_rankings", [])[:5], 1):
            score_bar = "█" * int(chunk["score"] * 10) + "░" * (10 - int(chunk["score"] * 10))
            preview = chunk['preview'][:60].replace('\n', ' ')
            print(f"    [{i}] [{score_bar}] {chunk['score']:.2f} - {preview}...")

        v = result["verification_result"]
        print(f"\n{'='*70}")
        print("📊 LONGTRACER VERIFICATION RESULT (PARALLEL)")
        print(f"{'='*70}")
        print(f"  Trust Score: {v['trust_score']:.2%}")
        print(f"  All Supported: {v['all_supported']}")
        print(f"  Total Claims: {len(v['claims'])}")
        print(f"  Flagged Claims: {len(v['flagged_claims'])}")
        print(f"  Hallucinations Detected: {v['hallucination_count']}")

        print(f"\n{'─'*70}")
        print("� CLAIM-TO-SOURCE VERIFICATION:")
        print(f"{'─'*70}")

        for i, claim in enumerate(v['claims'], 1):
            status = "✅" if claim['supported'] else "❌"
            label = ""
            if claim.get('is_meta_statement'):
                label = " ℹ️  HONEST UNCERTAINTY"
            elif claim.get('is_hallucination'):
                label = " 🚨 HALLUCINATION"
            elif claim.get('has_hallucination_pattern'):
                label = " ⚠️  POSSIBLE HALLUCINATION"

            print(f"\n  {status} Claim {i}:{label}")
            claim_text = claim['claim'][:150] + "..." if len(claim['claim']) > 150 else claim['claim']
            print(f"     \"{claim_text}\"")
            print(f"     Score: {claim['score']:.2%}")

            if claim['best_source'] and claim['score'] >= 0.30:
                metadata = claim.get('best_source_metadata', {})
                source_idx = claim.get('best_source_index', 0) + 1
                page = metadata.get('page', 'N/A') if metadata else 'N/A'
                print(f"     ↳ Matched: [Source {source_idx}, Page {page}]")
                print(f"       Best match: \"{claim['best_source'][:150]}...\"")

        if v['hallucination_count'] > 0:
            print(f"\n{'─'*70}")
            print("🚨 HALLUCINATION WARNING:")
            print(f"{'─'*70}")
            print(f"  {v['hallucination_count']} claim(s) appear to be fabricated!")
            for h in v['hallucinations']:
                print(f"    ❌ \"{h['claim'][:80]}...\"")

        print(f"\n{'='*70}")
        if v['all_supported'] and v['hallucination_count'] == 0:
            print("✅ VERDICT: All claims are supported by source documents!")
        elif v['hallucination_count'] > 0:
            print("⚠️  VERDICT: Possible hallucinations detected - verify manually!")
        else:
            print("⚠️  VERDICT: Some claims are not fully supported by sources.")
        print(f"{'='*70}")

        timing = result["timing"]
        latency = v.get('latency_stats', {})
        print(f"\n⏱️  PARALLEL LATENCY STATISTICS:")
        print(f"  Retrieval: {timing['retrieve_ms']:.0f}ms")
        print(f"  Parallel (Relevance + LLM): {timing['parallel_relevance_llm_ms']:.0f}ms")
        print(f"  Batch Verification: {timing['verify_ms']:.0f}ms")
        print(f"  ────────────────────────")
        print(f"  TOTAL PIPELINE: {timing['total_ms']:.0f}ms")
        print(f"\n  Internal Model Stats:")
        print(f"    STS: {latency.get('sts_calls', 0)} batch calls, {latency.get('sts_avg_ms', 0):.1f}ms")
        print(f"    NLI: {latency.get('nli_calls', 0)} calls (parallel), {latency.get('nli_avg_ms', 0):.1f}ms avg")
        print(f"    NLI Skipped: {latency.get('nli_skipped', 0)}")

        print(f"\n📚 SOURCE CITATIONS (Retrieved Context):")
        for i, source in enumerate(result['sources'][:5], 1):
            source_name = source.metadata.get('source', 'Unknown')
            page = source.metadata.get('page', 'N/A')
            print(f"\n  [{i}] Source: {Path(source_name).name if source_name != 'Unknown' else 'Unknown'}")
            print(f"      Page: {page}")
            preview = source.page_content[:150].replace('\n', ' ')
            print(f"      Content: {preview}...")

        tracer.end_root(outputs={
            "trust_score": v["trust_score"],
            "context_relevance": rel.get("average_relevance", 0),
            "hallucination_count": v["hallucination_count"],
            "verdict": "PASS" if v["all_supported"] else "FAIL",
            "mode": "parallel",
            "total_ms": timing["total_ms"]
        })

        print_trace_report(tracer)

        if tracer.root_run:
            print(f"\n🔗 Trace ID: {tracer.root_run['trace_id']}")
            print(f"   View with: python view_trace.py --id {tracer.root_run['trace_id']}")

        return

    print("Usage:")
    print("  python main.py --ingest '../Pdf docs'")
    print("  python main.py --query 'What is Beta-CLIP?'")


if __name__ == "__main__":
    main()
