# Contributing

Bug reports and small, focused pull requests are welcome. For anything larger, open an issue first so we can agree
on the change before you write it.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[server,dev]"
```

## Before you open a pull request

```bash
pytest -q
ruff check .
```

Both must pass.

The test suite is torch-free and runs in seconds. It uses a tiny synthetic ONNX graph and the hand-written items in
`tests/fixtures/`, never real model weights or third-party data. Keep it that way.

## Rules

- One change per pull request. Include a test for every behaviour change or bug fix.
- `src/mayaz/text.py` and `src/mayaz/model/render.py` define the exact tokens the released models were trained on.
  Changing them changes model outputs. Pull requests that touch them must show that outputs on the fixtures are
  unchanged, or explain why they are not.
- `web/src/render.js` mirrors the Python renderer line for line. Change both together.
- Server dependencies are pinned in `requirements.txt` and `server/requirements.txt`. Keep them identical. Do not
  add PyTorch or other heavy packages to the server path; a test fails if you do.
- Match the surrounding code: same naming, same comment density, line length 120.
- Commit messages: imperative mood, one subject line, a body if the reason is not obvious.

## Reporting bugs

Open an issue with the tier (`mayaz-17m`, `mayaz-32m` or `mayaz-68m`), the variant, the request that fails and the
response or traceback. Leave out private data.

Security issues and conduct reports go through **Security → Report a vulnerability**, not public issues.

## Licence

By contributing, you agree that your contributions are licensed under the MIT licence of this repository.

All participants follow the [Code of Conduct](CODE_OF_CONDUCT.md).
