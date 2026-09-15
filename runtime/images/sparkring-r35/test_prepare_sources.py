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


def test_repeated_packaging_has_identical_bytes_and_record(tmp_path, monkeypatch):
    """The same staged tree produces identical archives across clock seconds."""
    import time

    spec = importlib.util.spec_from_file_location(
        'prepare_r35_repeat', Path(__file__).with_name('prepare_sources.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / 'source'
    source.mkdir()

    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args]).decode().strip()

    git('init', '-q')
    git('config', 'user.name', 'Fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    git('config', 'core.autocrlf', 'false')
    (source / 'b12x').mkdir()
    (source / 'b12x/example.py').write_bytes(b'value = 1\n')
    git('add', '.')
    git('commit', '-qm', 'Fixture')
    monkeypatch.setitem(module.BASES, 'b12x',
                        (git('rev-parse', 'HEAD'), git('rev-parse', 'HEAD^{tree}')))
    (source / 'b12x/example.py').write_bytes(b'value = 2\n')
    git('add', '.')
    original_head = git('rev-parse', 'HEAD')
    original_tree = git('write-tree')
    outputs = [tmp_path / 'first', tmp_path / 'second']
    for output in outputs:
        output.mkdir()
    first = module.package('b12x', source, outputs[0])
    time.sleep(1.1)
    # Ambient archive mode settings must not alter the source artifact.
    git('config', 'tar.umask', '0077')
    second = module.package('b12x', source, outputs[1])
    assert first == second
    for name in ('archive', 'patch'):
        assert (outputs[0] / first[name]).read_bytes() == (outputs[1] / second[name]).read_bytes()
    with tarfile.open(outputs[1] / second['archive']) as archive:
        assert all(member.mtime == 946684800 for member in archive.getmembers())
        assert archive.extractfile('b12x/example.py').read() == b'value = 2\n'
    assert git('rev-parse', 'HEAD') == original_head
    assert git('write-tree') == original_tree
