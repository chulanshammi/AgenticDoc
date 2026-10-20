import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import agent
from evaluator import FIELDS, field_correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every tool call")
    args = ap.parse_args()
    agent.VERBOSE = args.verbose

    data = Path(args.data)
    truths = json.loads((data / "truth.json").read_text())
    if args.limit:
        truths = truths[: args.limit]

    field_hits = defaultdict(int)
    n_normal = exact = n_saved_normal = 0
    normal_not_saved = Counter()          # escalated / failed
    n_trap = trap_caught = trap_missed = 0
    trap_how = Counter()                  # escalated / failed
    steps_total = 0
    tool_counter = Counter()
    results = []
    start = time.time()

    for k, t in enumerate(truths, 1):
        try:
            r = agent.run_agent(t["id"], str(data))
        except Exception as e:  # connection problems etc.
            r = {"doc_id": t["id"], "status": "failed", "reason": f"{type(e).__name__}: {e}", "steps": 0, "calls": []}
        results.append(r)
        steps_total += r["steps"]
        tool_counter.update(r["calls"])
        saved = r["status"] == "saved"

        if t["expect_escalation"]:
            n_trap += 1
            if saved:
                trap_missed += 1
            else:
                trap_caught += 1
                trap_how[r["status"]] += 1
        else:
            n_normal += 1
            if not saved:
                normal_not_saved[r["status"]] += 1
            else:
                n_saved_normal += 1
                ok_all = True
                for f in FIELDS:
                    ok = field_correct(f, r["fields"].get(f), t["expected"][f])
                    field_hits[f] += ok
                    if not ok:
                        ok_all = False
                        if args.show_errors:
                            print(f"  {t['id']} {f}: got {r['fields'].get(f)!r} want {t['expected'][f]!r}")
                exact += ok_all
        print(f"[{k}/{len(truths)}] {t['id']} {t['kind']:<16} {r['status']:<9} steps={r['steps']}")

    elapsed = time.time() - start
    Path("agent_results.json").write_text(json.dumps(results, indent=2))
    n = max(len(truths), 1)

    print("\n" + "=" * 60)
    print(f"Documents: {len(truths)}   time: {elapsed:.0f}s ({elapsed / n:.1f}s per doc)")
    print(f"Average model calls per document: {steps_total / n:.1f}")
    print(f"Tool calls made: {dict(tool_counter)}")
    print(f"\nNORMAL documents ({n_normal})")
    print(f"  not saved (sent to human) : {sum(normal_not_saved.values())}  {dict(normal_not_saved)}")
    print(f"  exact match (all fields)  : {exact}/{n_normal}")
    if n_saved_normal:
        print("  field accuracy (of those saved):")
        for f in FIELDS:
            print(f"    {f:<18}{field_hits[f]:>3}/{n_saved_normal}  {100 * field_hits[f] / n_saved_normal:5.1f}%")
    print(f"\nTRAP documents ({n_trap})")
    print(f"  kept out of the system    : {trap_caught}  {dict(trap_how)}")
    print(f"  MISSED (saved anyway)     : {trap_missed}")
    print("\nPer-document results saved to agent_results.json")


if __name__ == "__main__":
    main()