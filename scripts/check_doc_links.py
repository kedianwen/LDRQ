#!/usr/bin/env python3
"""Check every relative link in the repository's Markdown: the file must exist and,
for a link to a heading (#anchor) in a .md file, the heading must exist, using
GitHub's anchor rules. Standard library only; run from anywhere.

    python3 scripts/check_doc_links.py      # exit 1 and a list if anything is broken
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def anchors(path, cache={}):
    if path not in cache:
        out, seen = set(), {}
        text = re.sub(r"```.*?```", "", open(path, encoding="utf-8").read(), flags=re.S)
        for line in text.splitlines():
            m = re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
            if not m:
                continue
            a = re.sub(r"[^\w\- ]", "", m.group(1).strip().lower()).replace(" ", "-")
            n = seen.get(a, 0)
            seen[a] = n + 1
            out.add(a if n == 0 else "%s-%d" % (a, n))
        cache[path] = out
    return cache[path]


def main():
    files = subprocess.check_output(["git", "ls-files", "*.md"], cwd=ROOT, text=True).split()
    bad = 0
    for f in files:
        text = open(os.path.join(ROOT, f), encoding="utf-8").read()
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        for m in re.finditer(r"\]\(([^)\s]+)\)", text):
            url = m.group(1)
            if re.match(r"(https?|mailto):", url):
                continue
            path, _, frag = url.partition("#")
            target = os.path.normpath(os.path.join(ROOT, os.path.dirname(f), path)) if path \
                else os.path.join(ROOT, f)
            if not os.path.exists(target):
                print("missing file  %s -> %s" % (f, url))
                bad += 1
            elif frag and target.endswith(".md") and frag not in anchors(target):
                print("missing anchor %s -> %s" % (f, url))
                bad += 1
    print("%d Markdown files, %d broken links" % (len(files), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
