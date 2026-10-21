import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def test_citation_cff_has_the_required_keys():
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text())
    for key in ("cff-version", "message", "title", "authors", "license", "version"):
        assert key in cff, key
    assert cff["license"] == "MIT"
    assert len(cff["authors"]) == 7
    assert cff["preferred-citation"]["type"] == "article"


def test_docs_name_existing_files():
    for doc in ("README.md", "REPRODUCE.md"):
        text = (REPO / doc).read_text()
        for rel in re.findall(r"\]\(([^)#:]+)\)", text):  # local Markdown links
            assert (REPO / rel).exists(), f"{doc}: {rel}"
        for rel in re.findall(r"figures/generate_\w+\.sh", text):
            assert (REPO / rel).is_file(), f"{doc}: {rel}"


def test_reproduce_lists_every_generate_script():
    text = (REPO / "REPRODUCE.md").read_text()
    for path in sorted((REPO / "figures").glob("generate_*.sh")):
        assert f"figures/{path.name}" in text, path.name
