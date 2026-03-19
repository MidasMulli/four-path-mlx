#!/usr/bin/env python3
"""
Real-World Four-Path Benchmark
================================

Pulls actual SEC filings from EDGAR and benchmarks four-path speculative
decoding against stock generation on five escalating tasks.

Usage:
    # First: start stock mlx-lm server on port 8898
    #   ~/.mlx-env/bin/python3.11 -m mlx_lm server --model mlx-community/Qwen3.5-9B-MLX-4bit --port 8898 --chat-template-args '{"enable_thinking":false}'
    # Then: start four-path server on port 8899
    #   ~/.mlx-env/bin/python3.11 four_path_server.py
    # Then:
    ~/.mlx-env/bin/python3.11 benchmark_realworld.py
"""

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── EDGAR Fetcher ────────────────────────────────────────────────

EDGAR_HEADERS = {
    "User-Agent": "PhantomBenchmark/1.0 (nick@phantom.local)",
    "Accept-Encoding": "gzip, deflate",
}

CACHE_DIR = Path(__file__).parent / "edgar-cache"
CACHE_DIR.mkdir(exist_ok=True)


def edgar_fetch(url: str, cache_name: str) -> str:
    """Fetch a URL with caching."""
    cache_file = CACHE_DIR / cache_name
    if cache_file.exists():
        return cache_file.read_text()

    print(f"  Fetching {url[:80]}...")
    req = urllib.request.Request(url, headers=EDGAR_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            # Handle gzip
            if resp.headers.get("Content-Encoding") == "gzip":
                import gzip
                data = gzip.decompress(data)
            text = data.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} fetching {url}")
        return ""

    cache_file.write_text(text)
    return text


def get_filing_url(cik: str, filing_type: str = "10-K") -> Optional[str]:
    """Get the latest filing URL from EDGAR."""
    cik_padded = cik.zfill(10)
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{filing_type}%22&dateRange=custom&startdt=2024-01-01&enddt=2026-12-31&forms={filing_type}&entities={cik}"

    # Use the EDGAR filing API
    api_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    text = edgar_fetch(api_url, f"submissions_{cik}.json")
    if not text:
        return None

    data = json.loads(text)
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form == filing_type:
            acc = accessions[i].replace("-", "")
            doc = primary_docs[i]
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"

    return None


def strip_html(html: str) -> str:
    """Basic HTML tag stripping."""
    text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_section(text: str, section_name: str, max_chars: int = 15000) -> str:
    """Extract a named section from a filing."""
    patterns = [
        rf'(?i)(Item\s+\d+[A-Z]?\.\s*{section_name})',
        rf'(?i)({section_name})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            start = match.start()
            # Find the next "Item" header or take max_chars
            next_item = re.search(r'(?i)Item\s+\d+[A-Z]?\.', text[start + 50:])
            if next_item:
                end = start + 50 + next_item.start()
            else:
                end = start + max_chars
            section = text[start:min(end, start + max_chars)]
            return section.strip()
    return text[:max_chars]


# ── Filing Data ──────────────────────────────────────────────────

COMPANIES = {
    "JPM": {"name": "JPMorgan Chase", "cik": "19617", "type": "10-K"},
    "GS":  {"name": "Goldman Sachs", "cik": "886982", "type": "10-K"},
    "ZION": {"name": "Zions Bancorp", "cik": "109380", "type": "10-K"},
    "RDDT": {"name": "Reddit", "cik": "1713445", "type": "S-1"},
    "BAC": {"name": "Bank of America", "cik": "70858", "type": "10-K"},
}


def load_filing(ticker: str) -> str:
    """Load a filing, fetching from EDGAR if needed."""
    info = COMPANIES[ticker]
    cache_name = f"{ticker}_{info['type'].replace('-','')}.txt"
    cache_file = CACHE_DIR / cache_name

    if cache_file.exists():
        print(f"  {ticker} {info['type']}: cached ({cache_file.stat().st_size / 1024:.0f} KB)")
        return cache_file.read_text()

    print(f"  {ticker}: finding latest {info['type']}...")
    url = get_filing_url(info["cik"], info["type"])
    if not url:
        print(f"  WARNING: Could not find {info['type']} for {ticker}")
        return ""

    print(f"  {ticker}: fetching {url[:80]}...")
    raw = edgar_fetch(url, f"{ticker}_raw.htm")
    if not raw:
        return ""

    text = strip_html(raw)
    cache_file.write_text(text)
    print(f"  {ticker}: {len(text):,} chars extracted")
    return text


# ── Benchmark Tasks ──────────────────────────────────────────────

def build_tasks(filings: dict) -> list:
    """Build the five benchmark tasks from loaded filings."""
    tasks = []

    # Task 1: Single filing analysis — JPM Risk Factors
    jpm_risks = extract_section(filings["JPM"], "Risk Factors", max_chars=12000)
    tasks.append({
        "name": "T1: Single Filing Analysis",
        "desc": "JPM 10-K Risk Factors → interest rate exposure classification",
        "messages": [
            {"role": "system", "content": "You are a senior credit analyst at a rating agency."},
            {"role": "user", "content": (
                f"Here is the Risk Factors section from JPMorgan Chase's latest 10-K filing:\n\n"
                f"{jpm_risks}\n\n---\n\n"
                "Identify every risk factor related to interest rate exposure. For each one:\n"
                "1. Classify severity as HIGH / MEDIUM / LOW\n"
                "2. Explain the transmission mechanism (how this risk hits the P&L)\n"
                "3. Flag any hedging disclosures that partially mitigate it\n\n"
                "Be specific. Cite the exact language from the filing."
            )},
        ],
        "max_tokens": 1024,
    })

    # Task 2: Cross-company comparison — JPM vs GS vs ZION
    gs_risks = extract_section(filings["GS"], "Risk Factors", max_chars=8000)
    zion_risks = extract_section(filings["ZION"], "Risk Factors", max_chars=8000)
    tasks.append({
        "name": "T2: Cross-Company Comparison",
        "desc": "JPM vs GS vs ZION Risk Factors → comparative analysis",
        "messages": [
            {"role": "system", "content": "You are a bank equity research analyst."},
            {"role": "user", "content": (
                f"=== JPMorgan Chase Risk Factors ===\n{jpm_risks[:8000]}\n\n"
                f"=== Goldman Sachs Risk Factors ===\n{gs_risks}\n\n"
                f"=== Zions Bancorp Risk Factors ===\n{zion_risks}\n\n---\n\n"
                "Compare the interest rate risk disclosures across these three banks:\n"
                "1. What risks does the regional bank (Zions) face that the G-SIBs don't?\n"
                "2. What risks do the G-SIBs disclose that Zions omits?\n"
                "3. Which bank appears most exposed to a sustained yield curve inversion?\n"
                "4. Rate the quality of each bank's risk disclosure (1-10) with reasoning."
            )},
        ],
        "max_tokens": 1024,
    })

    # Task 3: Adversarial analysis — Reddit S-1
    rddt_business = extract_section(filings.get("RDDT", ""), "Business", max_chars=6000)
    rddt_risks = extract_section(filings.get("RDDT", ""), "Risk Factors", max_chars=8000)
    tasks.append({
        "name": "T3: Adversarial Analysis",
        "desc": "Reddit S-1 → short-seller critique (analytical floor)",
        "messages": [
            {"role": "system", "content": "You are a forensic accountant working for a short-selling research firm."},
            {"role": "user", "content": (
                f"=== Reddit S-1: Business Section ===\n{rddt_business}\n\n"
                f"=== Reddit S-1: Risk Factors ===\n{rddt_risks}\n\n---\n\n"
                "You are evaluating this company pre-IPO for a short report. "
                "Identify the five weakest points in this filing:\n"
                "1. Where is management being evasive?\n"
                "2. Where are the numbers internally inconsistent?\n"
                "3. Where are risks clearly understated?\n"
                "4. What are they NOT saying that a sophisticated investor would notice?\n\n"
                "Be specific. Cite exact language. Explain what they're hiding."
            )},
        ],
        "max_tokens": 1024,
    })

    # Task 4: Document drafting from reference — GS MD&A
    gs_mda = extract_section(filings["GS"], "Management.s Discussion and Analysis", max_chars=12000)
    if len(gs_mda) < 1000:
        gs_mda = extract_section(filings["GS"], "Discussion and Analysis", max_chars=12000)
    if len(gs_mda) < 1000:
        # Fallback: use a larger chunk from the middle of the filing
        mid = len(filings["GS"]) // 3
        gs_mda = filings["GS"][mid:mid + 12000]

    tasks.append({
        "name": "T4: Document Drafting",
        "desc": "GS MD&A template → draft regional bank MD&A (N-gram sweet spot)",
        "messages": [
            {"role": "system", "content": "You are a financial writer drafting SEC filings."},
            {"role": "user", "content": (
                f"Here is the MD&A section from Goldman Sachs' latest 10-K:\n\n"
                f"{gs_mda}\n\n---\n\n"
                "Using this MD&A as a structural template, draft an MD&A section for "
                "Heartland Regional Bancshares, a mid-size regional bank with:\n"
                "- Net interest income declined 12% YoY due to deposit repricing\n"
                "- Fee income grew 8% through wealth management expansion\n"
                "- Total assets: $18.2 billion\n"
                "- CET1 ratio: 11.3%\n"
                "- NPL ratio increased from 0.8% to 1.2%\n\n"
                "Match the tone, structure, and level of detail of the Goldman reference. "
                "This is the headline section of the 10-K — it needs to read like a real filing."
            )},
        ],
        "max_tokens": 2048,
    })

    # Task 5: Batch classification — 10 risk factors from 5 banks
    risk_factors = []
    for ticker in ["JPM", "GS", "ZION", "BAC"]:
        if ticker in filings and filings[ticker]:
            risks = extract_section(filings[ticker], "Risk Factors", max_chars=6000)
            # Extract first two bullet points / paragraphs
            paragraphs = [p.strip() for p in risks.split('\n') if len(p.strip()) > 100][:2]
            for i, p in enumerate(paragraphs):
                risk_factors.append(f"[{ticker}-{i+1}] {p[:500]}")

    # Pad to 10 if needed
    while len(risk_factors) < 10:
        risk_factors.append(f"[GENERIC-{len(risk_factors)+1}] The Company faces risks related to changes in economic conditions and geopolitical uncertainty that may adversely affect its business, financial condition, and results of operations.")

    rf_text = "\n\n".join(risk_factors[:10])
    tasks.append({
        "name": "T5: Batch Classification",
        "desc": "10 risk factors → structured JSON classification",
        "messages": [
            {"role": "system", "content": "You are a risk classification engine. Output valid JSON only."},
            {"role": "user", "content": (
                f"Classify each of these 10 risk factors from bank 10-K filings:\n\n"
                f"{rf_text}\n\n---\n\n"
                "For each risk factor, output a JSON object with:\n"
                "- id: the [TICKER-N] tag\n"
                "- category: one of Credit Risk, Market Risk, Operational Risk, Regulatory Risk, Liquidity Risk, Strategic Risk\n"
                "- sub_risk: specific sub-category being described\n"
                "- confidence: 0.0 to 1.0\n"
                "- reasoning: one sentence explaining the classification\n\n"
                "Output as a JSON array. No markdown, no explanation, just the JSON."
            )},
        ],
        "max_tokens": 1024,
    })

    return tasks


# ── Benchmark Runner ─────────────────────────────────────────────

def run_completion(base_url: str, messages: list, max_tokens: int, temperature: float = 0.3) -> dict:
    """Send a chat completion request and measure performance."""
    body = json.dumps({
        "model": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        return {"error": str(e), "elapsed": time.perf_counter() - t0}
    elapsed = time.perf_counter() - t0

    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

    # Extract four-path stats if present
    x_fp = result.get("x_four_path", {})

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed": elapsed,
        "tok_per_sec": completion_tokens / elapsed if elapsed > 0 else 0,
        "content_preview": content[:200],
        "content_length": len(content),
        "sources": x_fp.get("sources", {}),
        "server_tps": x_fp.get("tok_per_sec", 0),
    }


def check_server(base_url: str, name: str) -> bool:
    """Check if a server is available."""
    try:
        req = urllib.request.Request(f"{base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        print(f"  {name} server at {base_url} is not available")
        return False


def run_benchmark():
    # Single server URL — we'll run stock first, then four-path, sequentially
    SERVER_URL = "http://127.0.0.1:8899"

    print("=" * 100)
    print("REAL-WORLD FOUR-PATH BENCHMARK")
    print("  SEC EDGAR filings → Head-to-head: stock generation vs four-path speculative decode")
    print("=" * 100)

    # Check if four-path server is running
    print("\nChecking server...")
    fourpath_ok = check_server(SERVER_URL, "Four-path")

    if not fourpath_ok:
        print("\nFATAL: Server not running on port 8899")
        print("  Start four-path: ~/.mlx-env/bin/python3.11 four_path_server.py")
        return

    # Check if stock baseline is available (run with --baseline flag or env var)
    stock_ok = os.environ.get("STOCK_URL")
    STOCK_URL = stock_ok or None
    FOURPATH_URL = SERVER_URL

    if stock_ok:
        print(f"  Stock baseline at: {STOCK_URL}")
    else:
        print("  No stock baseline (set STOCK_URL=http://... to enable)")
        print("  Running four-path only — will compare against known baseline (~21 tok/s)\n")

    # Fetch filings
    print("\nFetching SEC filings from EDGAR...")
    filings = {}
    for ticker in ["JPM", "GS", "ZION", "RDDT", "BAC"]:
        filings[ticker] = load_filing(ticker)
        if not filings[ticker]:
            print(f"  WARNING: Failed to fetch {ticker} — tasks using it will be degraded")

    # Build tasks
    print("\nBuilding benchmark tasks...")
    tasks = build_tasks(filings)
    print(f"  {len(tasks)} tasks ready\n")

    # Run benchmarks
    results = []

    for i, task in enumerate(tasks):
        print(f"\n{'━' * 100}")
        print(f"  {task['name']}")
        print(f"  {task['desc']}")
        print(f"  Max tokens: {task['max_tokens']}")
        print(f"{'━' * 100}")

        result = {"task": task["name"], "desc": task["desc"]}

        # Estimate prompt size
        prompt_text = " ".join(m.get("content", "") for m in task["messages"])
        print(f"  Prompt: ~{len(prompt_text):,} chars")

        # Stock baseline
        if STOCK_URL:
            print(f"\n  ▸ Stock (standard generation)...")
            stock_result = run_completion(STOCK_URL, task["messages"], task["max_tokens"])
            if "error" in stock_result:
                print(f"    ERROR: {stock_result['error']}")
                result["stock"] = stock_result
            else:
                print(f"    {stock_result['completion_tokens']} tok / {stock_result['elapsed']:.2f}s = "
                      f"{stock_result['tok_per_sec']:.1f} tok/s "
                      f"(prompt: {stock_result['prompt_tokens']})")
                result["stock"] = stock_result
        else:
            # Use known baseline from prior benchmarks
            result["stock"] = {"tok_per_sec": 21.0, "baseline_note": "estimated from prior benchmarks"}

        # Four-path
        print(f"\n  ▸ Four-path (N-gram + MTP + ANE + GPU)...")
        fp_result = run_completion(FOURPATH_URL, task["messages"], task["max_tokens"])
        if "error" in fp_result:
            print(f"    ERROR: {fp_result['error']}")
            result["four_path"] = fp_result
        else:
            src = fp_result.get("sources", {})
            drafted = sum(src.get(k, 0) for k in ["ngram", "mtp", "ane"])
            total = fp_result["completion_tokens"]
            print(f"    {total} tok / {fp_result['elapsed']:.2f}s = "
                  f"{fp_result['tok_per_sec']:.1f} tok/s "
                  f"(prompt: {fp_result['prompt_tokens']})")
            if src:
                print(f"    Sources: ngram={src.get('ngram',0)} mtp={src.get('mtp',0)} "
                      f"ane={src.get('ane',0)} gpu={src.get('gpu',0)} "
                      f"({drafted}/{total} drafted = {drafted/total*100:.0f}%)")
            if fp_result.get("server_tps"):
                print(f"    Server-measured: {fp_result['server_tps']:.1f} tok/s")
            result["four_path"] = fp_result

        # Speedup
        stock_tps = result.get("stock", {}).get("tok_per_sec", 0)
        if stock_tps > 0 and not fp_result.get("error"):
            speedup = fp_result["tok_per_sec"] / stock_tps
            baseline_label = "measured" if STOCK_URL else "~21 est"
            print(f"\n  ⚡ Speedup: {speedup:.2f}x vs {baseline_label} baseline ({stock_tps:.1f} tok/s)")
            result["speedup"] = speedup

        results.append(result)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")

    header = (f"  {'Task':<30} {'Prompt':>8} {'Tokens':>8} "
              f"{'Baseline':>10} {'4-Path':>10} {'Speedup':>10}"
              f"  │ {'N-gram':>8} {'MTP':>6} {'ANE':>6} {'GPU':>6} {'Draft%':>7}")
    print(header)
    divider = (f"  {'─' * 28} {'─' * 8} {'─' * 8} "
               f"{'─' * 10} {'─' * 10} {'─' * 10}"
               f"  │ {'─' * 8} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 7}")
    print(divider)

    for r in results:
        fp = r.get("four_path", {})
        st = r.get("stock", {})
        src = fp.get("sources", {})
        total = fp.get("completion_tokens", 0)
        drafted = sum(src.get(k, 0) for k in ["ngram", "mtp", "ane"])
        draft_pct = f"{drafted/total*100:.0f}%" if total else "—"
        stock_tps = st.get("tok_per_sec", 0)
        fp_tps = fp.get("tok_per_sec", 0)

        line = (f"  {r['task']:<30} {fp.get('prompt_tokens', '?'):>8} {total:>8} "
                f"{stock_tps:>9.1f} {fp_tps:>9.1f} {r.get('speedup', 0):>9.2f}x"
                f"  │ {src.get('ngram', 0):>8} {src.get('mtp', 0):>6} "
                f"{src.get('ane', 0):>6} {src.get('gpu', 0):>6} {draft_pct:>7}")
        print(line)

    # Save results
    out_path = Path(__file__).parent / "realworld_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    if stock_ok:
        speedups = [r["speedup"] for r in results if "speedup" in r]
        if speedups:
            print(f"\n  Average speedup: {sum(speedups)/len(speedups):.2f}x")
            print(f"  Best: {max(speedups):.2f}x (Task {results[speedups.index(max(speedups))]['task']})")
            print(f"  Floor: {min(speedups):.2f}x (Task {results[speedups.index(min(speedups))]['task']})")


if __name__ == "__main__":
    run_benchmark()
