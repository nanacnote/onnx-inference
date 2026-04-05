# Releasing the SDK

The SDK version is derived entirely from git tags — there is no version number to edit in code. Publishing a release is a five-step process.

---

## 1. Make sure main is ready

All intended changes merged, CI green, `sync_sdk_protos.py` run if any protos changed.

---

## 2. Tag the commit

Tags follow [Semantic Versioning](https://semver.org): `vMAJOR.MINOR.PATCH`.

```bash
git checkout main
git pull
git tag v0.2.0
git push origin v0.2.0
```

**Guidance on which number to increment:**

| Bump    | When                                                           |
| ------- | -------------------------------------------------------------- |
| `PATCH` | Bug fixes, dependency updates, no API change                   |
| `MINOR` | New method on `InferenceClient`, new model added               |
| `MAJOR` | Breaking change to an existing method signature or return type |

---

## 3. Publish a GitHub release

1. Go to `https://github.com/nanacnote/onnx-inference/releases/new`
2. Choose the tag you just pushed (`v0.2.0`) from the *Choose a tag* dropdown
3. Set the title to the tag name (`v0.2.0`)
4. Write release notes (see [below](#writing-release-notes))
5. Click **Publish release**

Publishing the release fires the `publish-sdk.yml` workflow automatically. Do not use *Save as draft* — that does not trigger the workflow.

---

## 4. Verify the workflow

Go to `https://github.com/nanacnote/onnx-inference/actions` and confirm the *Publish SDK* run completes successfully. On success, two files are attached to the release:

- `onnx_inference-0.2.0-py3-none-any.whl` — the wheel (preferred for installs)
- `onnx_inference-0.2.0.tar.gz` — the source distribution

---

## 5. Share the install command

Once the assets are attached, consumers can install the release with:

```bash
pip install https://github.com/nanacnote/onnx-inference/releases/download/v0.2.0/onnx_inference-0.2.0-py3-none-any.whl
```

Or via git:

```bash
pip install "onnx-inference @ git+https://github.com/nanacnote/onnx-inference.git@v0.2.0#subdirectory=sdk"
```

---

## Writing release notes

Keep notes short and consumer-focused. A useful structure:

```
### What's new
- `client.embed()` now accepts a list of strings for batch encoding (#42)

### Changed
- `OCRResult.box` is now always length 8 even when fewer points were detected

### Fixed
- `InferenceError.code` was `None` on connection timeout — now correctly `UNAVAILABLE`
```

Omit internal refactors, dependency bumps, and CI changes unless they affect consumers.

---

## Fixing a bad release

You cannot overwrite a published GitHub release's assets via the workflow. If the build was wrong:

1. Delete the release on GitHub (this does not delete the tag)
2. Delete the tag locally and remotely:
   ```bash
   git tag -d v0.2.0
   git push origin --delete v0.2.0
   ```
3. Fix the issue, re-tag, and publish a new release from step 2
