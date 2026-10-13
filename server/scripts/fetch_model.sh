#!/bin/sh
# Download the chosen tier's ONNX model at build time and verify its sha256 (a mismatch fails the build).
# Usage: sh fetch_model.sh ARTIFACTS_DIR REGISTRY_JSON
# MAYAZ_MODEL: tier name from the registry (default: the registry's "default", mayaz-32m).
set -eu
dest="$1"
reg="$2"
read_reg() {  # tier field
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); t=sys.argv[2] or r["default"]; e=r["tiers"].get(t)
if e is None: sys.exit("unknown MAYAZ_MODEL %r; choose one of %s" % (t, sorted(r["tiers"])))
print(t if sys.argv[3] == "tier" else (e.get(sys.argv[3]) or ""))' "$reg" "${MAYAZ_MODEL:-}" "$1"
}
tier="$(read_reg tier)"
url="$(read_reg asset_url)"
want="$(read_reg sha256)"
file="$(read_reg file)"
[ -n "$url" ] && [ -n "$want" ] || { echo "$tier has no released model yet (asset_url/sha256 empty)" >&2; exit 1; }
curl -fL --retry 3 -o "$dest/$file" "$url"
got="$(sha256sum "$dest/$file" | cut -d' ' -f1)"
[ "$got" = "$want" ] || { echo "sha256 mismatch for $tier: got $got want $want" >&2; exit 1; }
echo "$tier ok $got"
