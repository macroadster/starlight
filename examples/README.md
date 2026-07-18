# Examples

API client examples were removed — this repo is **training / export only**.

Inference serving belongs to Stargate (Go + Trin), not this project.

## Local evaluation

Use the scanner for offline checks:

```bash
python3 scanner.py --help
# or train / export via Makefile targets:
make train
make export-gguf
```

See `docs/GGUF_EXPORT.md` for GGUF export details.
