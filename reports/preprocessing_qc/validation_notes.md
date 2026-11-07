# Validation policy

- Download: every file listed in the official 1.0.3 SHA-256 manifest must match,
  including both waveform resolutions and ancillary release files. The manifest
  itself is preserved and its digest recorded in provenance.
- Metadata: unique recording IDs, integral patient IDs and official folds, and
  exact agreement between `RECORDS`, metadata paths, and the release manifest.
- Splits: every patient's records must have one original fold. Unexpected overlap
  stops processing; no folds are repaired or randomized.
- Labels: diagnostic SCP codes are aggregated using official class mappings,
  including likelihood zero. Full-release counts must match the versioned
  documentation. A separate reload check reconstructs each label independently.
- Waveforms: all selected 100 Hz records are read using WFDB and checked for 1000
  samples, 12 leads in header order, 100 Hz, mV units, and finite values. Any failure
  is logged by ID and reason. Arrays are written as float32 without transformations.
- Reload: every saved row ID, source demographic value/missing value, patient ID,
  original fold, partition, binary label and eligibility flag is checked. Every
  serialized waveform value is checked for finiteness, and up to 32 evenly spaced
  records (including the first and last) are compared to raw WFDB physical samples.
- Completion: outputs are staged and promoted only after reload validation passes.
  `COMPLETE.json` indicates success; exceptions return failure and write a traceback.
- Automated tests include corrupt/missing downloads, verified-file preservation,
  likelihood-zero/multiple/absent labels, unknown SCP codes, patient overlap,
  invalid folds, preserved demographic missingness, malformed/missing waveforms,
  exclusion compaction, and detection of corrupted serialized labels.

These checks validate data preparation. No predictive models, fitted imputation,
normalization, feature selection, augmentation, or feature extraction are run.

Source references:

- [Version 1.0.3 release documentation](https://physionet.org/content/ptb-xl/1.0.3/):
  release counts, diagnosis totals, fold recommendations, likelihood-zero semantics,
  and age deidentification.
- [Wagner et al., Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC7248071/):
  metadata definitions, sex encoding (0 male, 1 female), height in cm, weight in kg.
- The downloaded `example_physionet.py` illustrates diagnostic aggregation by
  code presence rather than a likelihood threshold. Its contents are preserved;
  this pipeline does not execute downloaded code.

## Observed official 1.0.3 index defect

The complete release passed all 87,203 official file checksums. Its unmodified
`RECORDS` has 43,597 lines for 43,598 waveform records: the line at the resolution
boundary concatenates `records100/21000/21837_lr` and
`records500/00000/00001_hr`. Both metadata paths and both waveform/header pairs
are present and checksum-verified. The parser recognizes this exact boundary
case in memory and records a QC warning; it does not change the raw file, remove
recordings, change identifiers, or alter folds. Any other index inconsistency
still causes failure. An automated regression test covers this behavior.
