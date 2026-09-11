#!/usr/bin/env bash
# Rebuild XGrammar 0.2.5 with the documented one-line R33 Transformers metadata adjustment.
set -euo pipefail

source_input=/source
source_dir=/work/source
out=/out
expected_commit=2ea71da4ccb997a06928c9fb69b99f330da56697
expected_tree=6f118db56d875807366d249dc327cb63dcb91b73

git config --global --add safe.directory "$source_input"
test "$(git -C "$source_input" rev-parse HEAD)" = "$expected_commit"
test "$(git -C "$source_input" rev-parse HEAD^{tree})" = "$expected_tree"
test -z "$(git -C "$source_input" status --porcelain)"
test -z "$(find /work -mindepth 1 -print -quit)"
cp -a "$source_input" "$source_dir"
git config --global --add safe.directory "$source_dir"
python3 - "$source_dir/pyproject.toml" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
source = path.read_text()
old = '"transformers>=4.38.0,<5",'
new = '"transformers>=4.38.0",'
if source.count(old) != 1:
    raise SystemExit("unexpected XGrammar Transformers requirement")
path.write_text(source.replace(old, new))
PY
git -C "$source_dir" add pyproject.toml
result_tree=$(git -C "$source_dir" write-tree)
git -C "$source_dir" diff --cached --binary HEAD -- pyproject.toml > "$out/xgrammar-transformers5.patch"
patch_sha=$(sha256sum "$out/xgrammar-transformers5.patch" | cut -d' ' -f1)

python3 /opt/sparkring-build/verify_torch.py > "$out/foundation-verification.json"
python3 -m pip install 'scikit-build-core>=0.10.0' 'apache-tvm-ffi==0.1.11' build
python3 -m pip freeze > "$out/build-dependencies.txt"
export CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS:-8}
cd "$source_dir"
python3 -m build --wheel --no-isolation --outdir "$out"
wheel=$(find "$out" -maxdepth 1 -name 'xgrammar-0.2.5-*.whl')
test -n "$wheel"

python3 - "$wheel" "$out/wheel-verification.json" <<'PY'
import base64,csv,hashlib,json,struct,sys,zipfile
wheel=sys.argv[1]
with zipfile.ZipFile(wheel) as archive:
    names=archive.namelist(); assert len(names)==len(set(names))
    metadata_name=next(name for name in names if name.endswith('.dist-info/METADATA'))
    metadata=archive.read(metadata_name).decode()
    assert 'Version: 0.2.5\n' in metadata
    assert 'Requires-Dist: transformers>=4.38.0\n' in metadata
    assert 'Requires-Dist: transformers<5,>=4.38.0\n' not in metadata
    record_name=metadata_name.replace('METADATA','RECORD')
    rows={row[0]:row[1:] for row in csv.reader(archive.read(record_name).decode().splitlines())}
    assert set(rows)==set(names)
    for name in set(names)-{record_name}:
        data=archive.read(name); raw=hashlib.sha256(data).digest()
        expected='sha256='+base64.urlsafe_b64encode(raw).rstrip(b'=').decode()
        assert rows[name]==[expected,str(len(data))],name
    native={}
    for name in names:
        if name.endswith(('.so','.so.0')):
            data=archive.read(name); assert data[:4]==b'\x7fELF'
            assert struct.unpack_from('<H',data,18)[0]==183
            native[name]=hashlib.sha256(data).hexdigest()
result={'schema':'sparkring-r33-xgrammar-wheel-verification/v1','record_valid':True,
        'metadata_adjustment':'transformers>=4.38.0','native_aarch64':native,
        'wheel_sha256':hashlib.sha256(open(wheel,'rb').read()).hexdigest()}
open(sys.argv[2],'w').write(json.dumps(result,indent=2,sort_keys=True)+'\n')
PY

sha256sum "$wheel" > "$out/SHA256SUMS"
printf 'status=compiled-cpu-import-and-runtime-qualification-pending\nsource.commit=%s\nsource.base.tree=%s\nsource.result.tree=%s\nmetadata.patch.sha256=%s\nsource.post-build-input-identical=true\n' \
  "$expected_commit" "$expected_tree" "$result_tree" "$patch_sha" > "$out/source-receipt.txt"
test "$(git -C "$source_input" rev-parse HEAD)" = "$expected_commit"
test "$(git -C "$source_input" rev-parse HEAD^{tree})" = "$expected_tree"
test -z "$(git -C "$source_input" status --porcelain)"
