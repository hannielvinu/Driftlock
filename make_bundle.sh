#!/usr/bin/env bash
# Usage: ./make_bundle.sh <CollegeName_TeamName> [estimated_trajectory.txt] [metadata.json]
# Builds <name>.zip in the layout from the Track 3 "Submission Bundle" slide.
set -euo pipefail
name="${1:?usage: ./make_bundle.sh CollegeName_TeamName}"
traj="${2:-estimated_trajectory.txt}"
meta="${3:-metadata.json}"
tmp="$(mktemp -d)"
mkdir -p "$tmp/$name/src"
cp "$traj" "$tmp/$name/estimated_trajectory.txt"
cp "$meta" "$tmp/$name/metadata.json"
cp src/main.py src/README.md "$tmp/$name/src/"
(cd "$tmp" && zip -qr "$OLDPWD/$name.zip" "$name")
rm -rf "$tmp"
echo "wrote $name.zip"
