# Release Script

## Prerequisites

- Install the GitHub CLI: https://cli.github.com/
- Authenticate with GitHub:

```sh
gh auth login
```

The authenticated account needs permission to create releases in the repository.

## Usage

From the repository root, run:

```sh
./scripts/release.sh v0.10.1 "ATF Chat-arm64.dmg"
```

The first argument is the release version and must start with `v`. The second argument is the release asset path. Both arguments are optional:

```sh
./scripts/release.sh
```

The defaults are `v0.10.0` and `ATF Chat-arm64.dmg`.

The script creates the GitHub release, creates the matching tag at the current commit when needed, generates release notes, and uploads the asset. It stops without changing anything if the release already exists or the asset cannot be found.