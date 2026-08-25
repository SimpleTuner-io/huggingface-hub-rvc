# AGENTS.md

## Project

- Package source lives in `src/huggingface_hub_rvc`.
- Tests live in `tests`.
- Use `python -m pytest -q` for the local test suite.
- Do not commit or push unless explicitly asked.
- Prefer focused changes. Avoid adding broad abstractions unless they remove real duplication.

## Code Style

- Keep the public API close to Hugging Face conventions: `from_pretrained`, `save_pretrained`, and `push_to_hub`.
- New artifact writes should prefer safetensors.
- Legacy `.pth` loading must remain supported.
- Avoid silent fallback paths. If a dependency or artifact is missing, raise a clear error.
- Optional heavyweight behavior such as Demucs separation, FAISS retrieval, and training should be explicit in the API and documented.

## Public Text Privacy

Never publish local machine identity in public text. This includes commit messages, PR titles, PR bodies, issue comments, release notes, package metadata, and generated model cards.

Forbidden public text includes:

- Local absolute home-directory paths
- Local account names or workstation usernames
- Co-author trailers with personal names or emails
- Temporary paths containing local identity
- Raw terminal output containing local identity

Use repo-relative paths and generic commands in public text.

Before any push, release, PR, or package publication:

1. Inspect the staged diff.
2. Inspect commit and PR text.
3. Inspect package metadata and release notes.
4. Confirm none contain local machine identity.

If a privacy guard blocks content, report only: `Blocked: local machine identity was found in public text.`
