# Portable historical audit support

`deep_v7_support.zip` supplies the historical inputs that replay and comparator
tests formerly read from one developer's personal artifact directory and ignored
`games/` directory. It is independent of Git history, local weights and logs, so
the same tests run from a shallow/clean checkout on Windows and Linux.

The archive contains exact accepted baseline and candidate source bytes, frozen
weights, the original 86-file cohort report and manifest, the accepted shared
source manifest, and the one full historical state history exercised by the
compaction test. Existing source, report and weight hashes remain unchanged.

For the 31 cases already committed in `deep_clipping_real_v7.json`, it includes
only the exact original version, observed-state and diagnostic lines from 12
real logs. The compaction regression also retains the preceding 80 diagnostic
lines required to reconstruct recorded target-selection branches. Original
line numbers are retained with blank lines elsewhere. These
files are explicitly **excerpts**, not complete original logs. Tests continue
to verify original diagnostic-line SHA256 hashes and observable state/context;
`provenance.json` retains the original full-log hashes separately. The archive
does not pretend to recompute full-log integrity from excerpts. Full-corpus
replay still requires the original complete logs supplied explicitly by its
caller and keeps its strict full-log checks.

No unrelated protocol events, tokens or credentials are included. The fixture
builder checked nonempty credential/token fields and credential patterns before
writing this archive. Each packaged file has its own SHA256 in provenance.

`tools.deep_clipping_replay.historical_support()` verifies and extracts the
archive once into the system temporary directory, outside the measured source
tree, and cleans it up at process exit. This does not alter game rules, active
weights, clipping, normalization or coverage exclusions.
