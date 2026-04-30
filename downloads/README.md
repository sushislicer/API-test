# Downloads

This directory is the default local artifact cache for heavyweight files that
should not be committed to the API repo.

Suggested layout:

```text
downloads/
  checkpoints/
    lda-1b/
    lingbot-va/
    motus/
    fastwam/
  assets/
    robotwin/
    libero/
    simplerenv/
  datasets/
```

The API does not download these files automatically. Put checkpoints, model
weights, simulator assets, and local datasets here manually, or set
`DOWNLOAD_ROOT` to a larger disk on the remote machine:

```bash
export API_ROOT="$(pwd)"
export DOWNLOAD_ROOT="${API_ROOT}/downloads"
```

If the remote machine has a shared filesystem, use the same variable:

```bash
export DOWNLOAD_ROOT=/mnt/a800_storage/api_downloads
```

Task metadata and training metadata can reference files with the `downloads/`
prefix, for example `downloads/checkpoints/lda-1b/run/checkpoints/step.pt`.
The API resolves that prefix through `DOWNLOAD_ROOT`, so the same config works
when `DOWNLOAD_ROOT` points to repo-local `downloads/` or to a mounted remote
disk.
