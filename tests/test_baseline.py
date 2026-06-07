from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_reasoner.data.synthetic import demo_cases
from pv_reasoner.inference.reviewer import BaselineReviewer
from pv_reasoner.retrieval.label_store import LabelStore, LabelSection


def test_baseline_generates_valid_packet():
    store = LabelStore([
        LabelSection(
            drug="Atorvastatin",
            section="Adverse Reactions",
            text="Rhabdomyolysis and myalgia are described.",
            adverse_reactions=["Rhabdomyolysis", "Myalgia"],
        )
    ])
    reviewer = BaselineReviewer(store)
    packet = reviewer.review(demo_cases()[0])
    assert packet.case_id == "DEMO-001"
    assert packet.seriousness.is_serious is True
    assert packet.meddra_suggestions
    assert packet.labelling_status.status in {"listed", "unlisted", "unknown"}
