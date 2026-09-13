"""Source archives must contain Git bytes regardless of checkout line endings."""
import importlib.util
from pathlib import Path
import subprocess
import tarfile


def test_archive_ignores_autocrlf(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('prepare_r35', Path(__file__).with_name('prepare_sources.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, output = tmp_path/'source', tmp_path/'output'
    source.mkdir()
    output.mkdir()
    def git(*args):
        return subprocess.check_output(['git','-C',str(source),*args]).decode().strip()
    git('init','-q')
    git('config','core.autocrlf','true')
    git('config','user.name','Fixture')
    git('config','user.email','fixture@example.invalid')
    (source/'b12x').mkdir()
    (source/'b12x/example.py').write_bytes(b'value = 1\r\n')
    git('add','.')
    git('commit','-qm','Fixture')
    monkeypatch.setitem(module.BASES,'b12x',(git('rev-parse','HEAD'),git('rev-parse','HEAD^{tree}')))
    record = module.package('b12x',source,output)
    expected = subprocess.check_output(['git','-C',str(source),'show','HEAD:b12x/example.py'])
    with tarfile.open(output/record['archive']) as archive:
        assert archive.extractfile('b12x/example.py').read() == expected == b'value = 1\n'
