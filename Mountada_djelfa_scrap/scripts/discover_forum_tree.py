#!/usr/bin/env python
"""CLI: discover djelfa.info's full subforum tree, label every subforum with its majority dialect, and
write

  data/state/forum_tree.json      full tree; each non-category, non-private entry also gets a `dialect`
                                  field (majority class of a sample of its recent posts, or null)
  data/state/scrape_targets.json  flattened list of non-private, non-category forum_ids

Labelling samples the first post page of up to --sample-threads recent threads per subforum (at most
1 + --sample-threads requests each) and classifies up to --sample-posts posts with the shipped DarijaDZ
dialect classifier. It needs `skops` + `scikit-learn`; pass --no-label for a structure-only fetch.

Existing files are copied once to `<name>.prev` before being overwritten (an existing `.prev` is kept, so
the very first copy survives reruns). Non-null labels already present in the existing tree are carried
over, so rerunning after a crash or session expiry resumes the labelling instead of redoing it (null
labels are always retried) -- pass --relabel to redo every subforum.

Requires a bootstrapped session first — see scripts/bootstrap_session.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tqdm import tqdm  # noqa: E402

from darija_forum._atomic import replace_with_retry  # noqa: E402
from darija_forum.dialect import DEFAULT_CLASSIFIER_DIR, DialectClassifier, label_forums  # noqa: E402
from darija_forum.discover import discover_forum_tree  # noqa: E402
from darija_forum.http_client import ForumHttpClient, SessionExpiredError  # noqa: E402
from darija_forum.session import SessionMissingError  # noqa: E402


def _write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    replace_with_retry(tmp_path, path)


def _backup_once(path: Path) -> None:
    prev = path.with_name(path.name + ".prev")
    if path.exists() and not prev.exists():
        shutil.copy2(path, prev)
        print(f"Backed up {path.name} -> {prev.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--max-forums",
        type=int,
        default=None,
        help="Cap total forums considered. Note: djelfa.info's index page renders its "
        "*entire* subforum tree inline (confirmed empirically — recursing into every "
        "individual subforum page found zero additional forums beyond the index alone), "
        "so this only limits how many of the already-seeded forums get an extra "
        "verification visit — it won't shrink the initial index-page result itself.",
    )
    parser.add_argument("--no-label", action="store_true", help="Structure-only fetch: skip the dialect labelling.")
    parser.add_argument("--relabel", action="store_true", help="Redo the labelling even for subforums that already have a label.")
    parser.add_argument("--sample-threads", type=int, default=10, help="Recent threads sampled per subforum (default 10).")
    parser.add_argument("--sample-posts", type=int, default=60, help="Posts classified per subforum (default 60).")
    parser.add_argument("--classifier-dir", default=str(DEFAULT_CLASSIFIER_DIR))
    parser.add_argument("--session-path", default=str(ROOT / "data" / "state" / "session.json"))
    args = parser.parse_args()

    try:
        client = ForumHttpClient(Path(args.session_path))
    except SessionMissingError as exc:
        raise SystemExit(str(exc))

    tree_path = ROOT / "data" / "state" / "forum_tree.json"
    targets_path = ROOT / "data" / "state" / "scrape_targets.json"

    forums = discover_forum_tree(client, max_forums=args.max_forums)

    old_tree = json.loads(tree_path.read_text(encoding="utf-8")) if tree_path.exists() else {}
    if not args.relabel:
        for forum_id, info in forums.items():
            previous = old_tree.get(forum_id, {})
            if previous.get("dialect") is not None and not info["is_category"] and not info["is_private"]:
                info["dialect"] = previous["dialect"]

    _backup_once(tree_path)
    _backup_once(targets_path)
    targets = [fid for fid, info in forums.items() if not info["is_category"] and not info["is_private"]]
    _write_json(forums, tree_path)
    _write_json(targets, targets_path)

    categories = sum(1 for info in forums.values() if info["is_category"])
    private = sum(1 for info in forums.values() if info["is_private"])
    print(f"Discovered {len(forums)} forum entries: {categories} categories, "
          f"{len(forums) - categories} subforums ({private} flagged private).")
    print(f"Scrape targets (non-private, non-category): {len(targets)}")
    print(f"Wrote {tree_path}")
    print(f"Wrote {targets_path}")

    if args.no_label:
        return

    classifier = DialectClassifier(Path(args.classifier_dir))
    todo = sum(
        1 for info in forums.values()
        if not info["is_category"] and not info["is_private"] and (args.relabel or info.get("dialect") is None)
    )
    print(f"Labelling {todo} subforums (up to {args.sample_posts} posts from {args.sample_threads} recent threads each)")
    progress = tqdm(total=todo, desc="labelling subforums", unit="forum")
    try:
        label_forums(
            client, forums, classifier,
            max_threads=args.sample_threads, max_posts=args.sample_posts, relabel=args.relabel,
            save=lambda: _write_json(forums, tree_path), on_progress=lambda: progress.update(1),
        )
    except SessionExpiredError as exc:
        print(f"\nStopped: {exc}\nLabels so far are saved in {tree_path.name}; rerun this script to resume.")
        raise SystemExit(1)
    finally:
        progress.close()

    counts = Counter(info["dialect"] for info in forums.values() if not info["is_category"] and not info["is_private"])
    print("Majority dialect per subforum: " + ", ".join(f"{k or 'unlabelled'}={v}" for k, v in counts.most_common()))


if __name__ == "__main__":
    main()
