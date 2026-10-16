import pytest

from citekit.config import Settings
from citekit.loaders import Document
from citekit.pipeline import CiteKit

SAMPLE_DOCS = [
    Document(
        id="wifi-1",
        source="data/docs/network_policy.md",
        text=(
            "# Campus Network\n\n"
            "The guest network is called campus-guest and disconnects after twelve hours.\n"
            "The portal is at wifi.example.edu/guest.\n"
            "The secured network uses WPA3-Enterprise with per-user credentials.\n"
        ),
    ),
    Document(
        id="lib-1",
        source="data/docs/library_rules.md",
        text=(
            "# Library\n\n"
            "The library is open from 08:00 to 20:00 Monday to Friday and closes at 20:00.\n"
            "Undergraduate students may borrow up to four books for fourteen days.\n"
            "Overdue books cost Rs 2 per day per book.\n"
        ),
    ),
]


@pytest.fixture
def kit() -> CiteKit:
    return CiteKit(settings=Settings(chunk_size=200, chunk_overlap=40)).index(SAMPLE_DOCS)


@pytest.fixture
def repo_docs_dir() -> str:
    return "data/docs"