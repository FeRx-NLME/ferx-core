# File Formats

Reference for the binary and structured output formats produced and consumed by ferx-core.

| Format | Description |
|----|----|
| [`.fitrx` bundle](../file-formats/fitrx.llms.md) | Portable zip archive containing fit results, EBEs, predictions, and the model file. Designed for cross-language interchange (R, Python, CLI). |
| [Check report (`ferx check`)](../file-formats/check-report.llms.md) | Structured JSON output from `ferx check` — one entry per finding with a stable `code`, optional block/line pointer, and a human-readable suggestion. |
