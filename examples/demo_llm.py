#!/usr/bin/env python3
"""
kore-bridge demo: Full cognitive cycle with a REAL LLM.

Requirements:
  1. Ollama running: https://ollama.com
  2. A model pulled: ollama pull llama3.2
  3. pip install kore-mind kore-bridge

100% local. Zero API keys. Zero cloud.
"""

import os
import sys
import tempfile
import json
import urllib.request

from kore_mind import Mind
from kore_bridge import Bridge, OllamaProvider


def header(text):
    print(f"\n{'='*64}")
    print(f"  {text}")
    print(f"{'='*64}\n")


def check_ollama(model="llama3.2"):
    """Check Ollama is running and model is available."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        models = [m["name"].split(":")[0] for m in data.get("models", [])]
        if model not in models:
            print(f"  Model '{model}' not found. Available: {models}")
            print(f"  Run: ollama pull {model}")
            return False
        return True
    except Exception:
        print("  Ollama is not running.")
        print("  Install: https://ollama.com")
        print("  Then: ollama serve & ollama pull llama3.2")
        return False


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "llama3.2"

    header("KORE-BRIDGE: LLM-Powered Cognitive Demo")
    print(f"  Model: {model} (via Ollama, 100% local)")

    if not check_ollama(model):
        return

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    mind = Mind(db_path)
    llm = OllamaProvider(model=model)
    bridge = Bridge(mind=mind, llm=llm)

    # ── Phase 1: Seed the mind ─────────────────────────────────────────

    header("PHASE 1 — Seeding experiences")

    experiences = [
        ("User is a researcher working on P vs NP and proof complexity",
         "semantic", "carlos", ["research", "complexity", "P-vs-NP"]),
        ("User prefers concise, direct answers — no filler or small talk",
         "semantic", "carlos", ["preference", "style"]),
        ("User codes in Python, runs experiments on A100 GPUs",
         "semantic", "carlos", ["python", "GPU", "hardware"]),
        ("User is building an algebraic proof engine called AIP",
         "semantic", "carlos", ["AIP", "proofs", "project"]),
        ("User works late at night, timezone GMT+1, based in Spain",
         "episodic", "carlos", ["schedule", "location"]),
        ("User mentioned interest in open source infrastructure tools",
         "semantic", "carlos", ["OSS", "infrastructure"]),
    ]

    for content, type_, source, tags in experiences:
        bridge.observe(content, source=source, type=type_, tags=tags)
        print(f"  + {content[:60]}...")

    print(f"\n  Mind: {mind.count} memories stored.\n")

    # ── Phase 2: LLM-powered reflect ──────────────────────────────────

    header("PHASE 2 — Reflect (LLM generates identity)")
    print("  The LLM analyzes all memories and generates an identity...")
    print("  This is NOT a summary. It's EMERGENT understanding.\n")

    identity = bridge.reflect()

    print(f"  IDENTITY SUMMARY:")
    print(f"  {identity.summary}\n")
    if identity.traits:
        print(f"  TRAITS:")
        for trait, score in sorted(identity.traits.items(), key=lambda x: -x[1])[:5]:
            bar = "█" * int(score * 20)
            print(f"    {bar} {score:.2f} {trait}")
        print()
    if identity.relationships:
        print(f"  RELATIONSHIPS:")
        for name, desc in identity.relationships.items():
            print(f"    {name}: {desc}")
        print()

    # ── Phase 3: Think with context ────────────────────────────────────

    header("PHASE 3 — Think (LLM responds WITH memory)")

    questions = [
        "What should I focus on next in my research?",
        "Suggest a good Python library for sparse matrix operations.",
        "How would you describe our working relationship so far?",
    ]

    for q in questions:
        print(f"  USER: {q}\n")
        response = bridge.think(q, user="carlos")
        # Wrap long lines
        words = response.split()
        line = "  AI: "
        for w in words:
            if len(line) + len(w) > 70:
                print(line)
                line = "      "
            line += w + " "
        print(line)
        print()

    print(f"  Mind now has {mind.count} memories (grew from conversations).\n")

    # ── Phase 4: Re-reflect after conversations ────────────────────────

    header("PHASE 4 — Re-reflect (identity evolves)")
    print("  After 3 conversations, the identity should be richer...\n")

    identity = bridge.reflect()
    print(f"  UPDATED IDENTITY:")
    print(f"  {identity.summary}\n")

    # ── Summary ────────────────────────────────────────────────────────

    header("WHAT YOU JUST SAW")

    print(f"""  1. SEEDED 6 experiences into the mind (no LLM needed)
  2. LLM REFLECTED on memories → generated emergent identity
  3. LLM RESPONDED to questions WITH full cognitive context
  4. Each conversation was AUTO-OBSERVED (mind grew to {mind.count} memories)
  5. RE-REFLECTED → identity evolved with new experiences

  All from a {os.path.getsize(db_path)} byte SQLite file.
  100% local. Zero API keys. Zero cloud.

  Model: {model}
  Database: {db_path}
""")

    mind.close()
    os.unlink(db_path)


if __name__ == "__main__":
    main()
