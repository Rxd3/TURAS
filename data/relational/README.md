# Relational data handoff

No CSV rows or database have been created. The authoritative proposed tables,
keys, relationships, graph mapping and access rules are in the
[RDL contract](../../docs/contracts/rdl.md).

M1 owns source tables and vocabulary; M5 owns validated graph snapshots and
learning. Future table bundles require a schema version, per-file hashes,
source/rights provenance, train/validation/test access labels and an audit that
held-out reference-derived facts never enter training/inference. M6 reviews this
boundary. Store only approved small metadata here; bulk CSV/graph/database files
are external artifacts. Never populate a table with invented examples to suggest
the dataset or graph is complete.
