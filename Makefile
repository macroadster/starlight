# Starlight monorepo — submodule helpers.
# Submodules pin one known-good snapshot in the superproject, but the
# parent ignores drift (.gitmodules: ignore = all), so pin bumps are
# optional and never dirty `git status`.

.PHONY: sync sync-init

# Fast-forward every submodule to its origin/main tip, merging into the
# currently checked-out branch (no detached HEADs).
sync:
	git submodule update --remote --merge --recursive

# Register and check out submodules after a fresh clone.
sync-init:
	git submodule update --init --recursive
