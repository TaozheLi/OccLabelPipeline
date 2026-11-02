# Contributing

Use a new output directory for recipe changes. Keep datasets, checkpoints, CUDA binaries and local configs out of commits. Include the resolved configuration, frame count, external framework revisions, GPU/runtime versions and a short sanitized log when reporting an issue.

```bash
python -m unittest discover -s tests -v
python -m compileall -q occ_label_pipeline
```

Changes to geometry should verify coordinate direction, grid Z shift, unknown/free masks, ego exclusion, dynamic-last priority and measured-ray-only free. Keep English README as the default and update its Chinese counterpart when commands change. Avoid adding automatic downloads, publication, background monitors or browser services to the generation command.
