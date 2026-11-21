"""Policy-vs-RBI compliance checking built on the retrieval stack.

* :mod:`~rbirag.compliance.policy` loads a bank's policy document (PDF or text);
* :mod:`~rbirag.compliance.clauses` splits it into individually checkable clauses;
* :mod:`~rbirag.compliance.references` links notifications that amend, supersede or
  repeal one another, so stale rules can be recognised;
* :mod:`~rbirag.compliance.sections` cuts long notifications down to the passages that
  matter for a clause;
* :mod:`~rbirag.compliance.regulations` exposes search and fetch over the corpus: the
  tools the compliance agent calls.
"""
