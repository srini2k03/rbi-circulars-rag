from __future__ import annotations

import pytest

from rbirag.compliance.clauses import Clause, split_clauses

NUMBERED = """\
SAMPLE BANK LTD. - FRAUD RISK MANAGEMENT POLICY

This Policy was approved by the Board of Directors on 2 April 2025 and is effective
immediately.

1. Governance

1.1 The Board shall approve the Policy and review it at least annually.

1.2 A Special Committee of the Board shall monitor frauds of Rs. 1 crore and above.

2. Reporting of Frauds

2.1 The Bank shall report frauds to the Reserve Bank within the following timelines:
(a) within 14 days of classification as fraud; and
(b) through the CIMS portal.

2.2 Frauds involving amounts below Rs. 1 lakh may be closed at the branch level.
"""


def labels(clauses: list[Clause]) -> list[str]:
    return [c.label for c in clauses]


def test_numbered_policy_with_headings_and_sub_items() -> None:
    clauses = split_clauses(NUMBERED)
    assert labels(clauses) == ["c001", "1.1", "1.2", "2.1", "2.2"]
    preamble, *rest = clauses
    assert preamble.kind == "paragraph" and preamble.number is None
    assert "approved by the Board" in preamble.text
    assert all(c.kind == "numbered" for c in rest)
    title = "SAMPLE BANK LTD. - FRAUD RISK MANAGEMENT POLICY"
    assert preamble.heading_path == [title]  # an all-caps title is the outermost heading
    assert clauses[1].heading_path == [title, "Governance"]
    assert clauses[3].heading_path == [title, "Reporting of Frauds"]
    # lettered sub-items stay inside their parent clause
    assert "(a) within 14 days" in clauses[3].text and "(b) through the CIMS" in clauses[3].text
    assert "2.2" not in clauses[3].text


def test_offsets_round_trip() -> None:
    for c in split_clauses(NUMBERED):
        assert NUMBERED[c.start : c.end] == c.text
        assert c.text == c.text.strip()


def test_markdown_headings_form_the_heading_path() -> None:
    text = (
        "# Deposit Policy\n\n## 4. Interest Rates\n\n"
        "4.1 Interest on savings deposits shall be calculated on a daily product basis.\n\n"
        "4.2 Interest rates shall be uniform across branches for deposits of the same size.\n\n"
        "## 5. Nomination\n\n5.1 Nomination facility shall be offered to all depositors.\n"
    )
    clauses = split_clauses(text)
    assert labels(clauses) == ["4.1", "4.2", "5.1"]
    assert clauses[0].heading_path == ["Deposit Policy", "Interest Rates"]
    assert clauses[2].heading_path == ["Deposit Policy", "Nomination"]


def test_top_level_numbered_clauses_without_headings() -> None:
    text = (
        "Outsourcing Policy\n\n"
        "1. The Bank shall not outsource core management functions such as internal audit.\n"
        "2. The Bank shall conduct due diligence on every service provider before engagement.\n"
        "Clause 3: Service providers shall be reviewed at least once a year.\n"
    )
    clauses = split_clauses(text)
    assert labels(clauses) == ["1", "2", "3"]
    assert clauses[2].text.startswith("Clause 3:")


def test_numbered_heading_needs_children() -> None:
    """``3. Short Title`` without sub-clauses is a clause, not a heading."""
    text = "1. Scope\n\n1.1 Applies to all branches of the bank.\n\n2. Board review annually\n"
    clauses = split_clauses(text)
    assert labels(clauses) == ["1.1", "2"]
    assert clauses[0].heading_path == ["Scope"]


def test_nested_numbering_and_unrelated_headings_are_closed() -> None:
    text = (
        "3. Customer Due Diligence\n\n"
        "3.1 Identity shall be verified using officially valid documents.\n\n"
        "4.1 Records shall be kept for five years after the business relationship ends.\n"
    )
    clauses = split_clauses(text)
    assert labels(clauses) == ["3.1", "4.1"]
    assert clauses[0].heading_path == ["Customer Due Diligence"]
    assert clauses[1].heading_path == []  # "3." does not enclose 4.1


def test_wrapped_line_starting_with_a_number_is_not_a_clause() -> None:
    text = (
        "1.1 The Bank shall comply with the provisions under section\n"
        "5.2 of the Act read with the rules made thereunder.\n\n"
        "1.2 Next clause about something else entirely here.\n"
    )
    clauses = split_clauses(text)
    assert labels(clauses) == ["1.1", "1.2"]
    assert "5.2 of the Act" in clauses[0].text


def test_years_and_amounts_are_not_clause_numbers() -> None:
    text = (
        "2025 Policy Review Summary for the Board of Directors meeting\n\n"
        "The review found 12 open audit points that management will close by June.\n\n"
        "15 days notice shall be given to customers before any change in charges.\n"
    )
    clauses = split_clauses(text)
    assert all(c.number is None for c in clauses)
    assert any("15 days notice" in c.text for c in clauses)


def test_unnumbered_prose_falls_back_to_paragraphs() -> None:
    text = (
        "Customer Grievance Redressal Policy\n\n"
        "Purpose\n"
        "The Bank is committed to resolving every customer complaint fairly and quickly.\n\n"
        "Timelines\n"
        "Complaints shall be acknowledged within 3 working days and resolved within 30 days.\n"
        "Unresolved complaints shall be escalated to the Principal Nodal Officer.\n\n"
        "Ok.\n"
    )
    clauses = split_clauses(text)
    assert [c.kind for c in clauses] == ["paragraph", "paragraph"]
    assert clauses[0].heading_path == ["Purpose"]
    assert clauses[1].heading_path == ["Timelines"]
    assert "escalated" in clauses[1].text  # consecutive lines form one paragraph


def test_numbered_heading_with_intro_paragraph() -> None:
    text = (
        "5. Record Management\n"
        "The following requirements apply to all records maintained by the Bank.\n\n"
        "5.1 Records of transactions shall be preserved for at least five years.\n"
    )
    clauses = split_clauses(text)
    assert labels(clauses) == ["5", "5.1"]
    assert clauses[0].text.startswith("The following requirements")
    assert clauses[1].heading_path == ["Record Management"]


@pytest.mark.parametrize("text", ["", "   \n\n  ", "Title only"])
def test_degenerate_inputs(text: str) -> None:
    assert split_clauses(text) == []


def test_sample_policy_file_splits_into_expected_clauses() -> None:
    from pathlib import Path

    from rbirag.compliance.policy import load_policy

    sample = Path(__file__).parents[1] / "data" / "samples" / "kyc_policy_sample.md"
    clauses = split_clauses(load_policy(sample).text)
    assert labels(clauses) == ["c001", "1.1", "1.2", "2.1", "2.2", "3.1", "3.2", "4.1", "5.1"]
    by_label = {c.label: c for c in clauses}
    assert "(c) such other documents" in by_label["3.1"].text
    assert by_label["3.2"].heading_path[-1] == "Customer Due Diligence"
