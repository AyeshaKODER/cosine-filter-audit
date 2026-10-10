# Legacy full-model records

This directory preserves the timestamped root-level JSON run records and CSV
tables from the earlier implementation, plus its plotting source. They are
retained for auditability and have not been rewritten.

These artifacts use the historical full-model representation and have
incomplete or unverified seed, partition, code-revision, and task-metric
provenance. Do not combine them with the new delta-space run artifacts or cite
them as paired evidence for the current implementation.

- `json/`: original timestamped run JSON files, filenames unchanged.
- `csv/`: original historical detection CSV files, filenames unchanged.
- `plot_inversion.py`: hard-coded historical plotting source; not executed or
  regenerated as part of the current delta-space study.
