import os
import pathlib
import shutil
import time
import types

import h5py

from bmlab.file import is_session_file
from bmlab.session import Session

from bmicro.gui import batch_cache
from bmicro.gui.main import BMicro

DATA = pathlib.Path(__file__).parent.parent / 'data'


def _repetitions(progress='extraction'):
    return {'0': {'label': 'Repetition 0', 'valid': True, 'nr_spectra': 5,
                  'calib_keys': ['1', '2'], 'checked': True,
                  'progress': progress, 'warning': None}}


def _paths(tmp_path):
    """ A raw file and a (minimal) session file, as bmlab writes it. """
    raw = tmp_path / 'RawData' / 'file.h5'
    session = tmp_path / 'EvalData' / 'file.h5'
    raw.parent.mkdir()
    session.parent.mkdir()
    raw.write_bytes(b'raw')
    _write_session(session)
    return raw, session


def _write_session(path):
    """ What Session.save() does: a new file with the session in it. """
    with h5py.File(path, 'w') as f:
        f.create_group('session')
        f.attrs['version'] = 'bmlab_0.0.1'


def test_entry_inside_the_session_file_is_read_back(tmp_path):
    raw, session = _paths(tmp_path)
    base = (('S0', 295.15, None, None), (1, True, False))

    assert batch_cache.write(raw, session, _repetitions(), base)
    cached = batch_cache.read(raw, session)

    assert cached['repetitions']['0']['calib_keys'] == ['1', '2']
    assert cached['progress']['0']['progress'] == 'extraction'
    assert cached['saved_base'] == base  # tuples again, as compared with
    assert is_session_file(session)  # still a session file for bmlab


def test_entry_is_dropped_by_saving_and_ignored_for_a_changed_raw_file(
        tmp_path):
    raw, session = _paths(tmp_path)
    batch_cache.write(raw, session, _repetitions(), None)

    raw.write_bytes(b'new raw data')  # e.g. repetitions removed
    assert batch_cache.read(raw, session) is None

    batch_cache.write(raw, session, _repetitions(), None)
    _write_session(session)  # saved again: a new file without the entry
    assert batch_cache.read(raw, session) is None


def test_nothing_is_written_without_a_session_or_into_other_files(tmp_path):
    raw = tmp_path / 'raw.h5'
    raw.write_bytes(b'raw')
    assert not batch_cache.write(raw, tmp_path / 'none.h5', _repetitions(),
                                 None)
    assert not (tmp_path / 'none.h5').exists()  # no file is created
    assert batch_cache.read(raw, tmp_path / 'none.h5') is None

    other = tmp_path / 'other.h5'  # an h5 file that is not a session
    with h5py.File(other, 'w') as f:
        f.create_group('Brillouin')
    assert not batch_cache.write(raw, other, _repetitions(), None)
    with h5py.File(other, 'r') as f:
        assert batch_cache.ATTRIBUTE not in f.attrs


def test_garbage_entry_is_ignored(tmp_path):
    raw, session = _paths(tmp_path)
    with h5py.File(session, 'r+') as f:
        f.attrs[batch_cache.ATTRIBUTE] = '{not json'
    assert batch_cache.read(raw, session) is None


def _copy_water_with_session(tmp_path):
    """ Water.h5 with a saved session next to it (Water.session.h5). """
    path = tmp_path / 'Water.h5'
    shutil.copy(DATA / 'Water.h5', path)
    session = Session.get_instance()
    session.clear()
    session.set_file(path)
    session.save()
    session.clear()
    return path, tmp_path / 'Water.session.h5'


def test_second_scan_reads_neither_raw_file_nor_session(tmp_path, mocker):
    path, session_path = _copy_water_with_session(tmp_path)
    first = BMicro._scan_batch_repetitions(path)  # reads, stores the entry
    assert first and batch_cache.read(path, session_path) is not None

    reader = mocker.patch('bmicro.gui.main.BrillouinFile',
                          side_effect=AssertionError('raw file read'))
    session_reader = mocker.patch('bmicro.gui.main.Serializer.deserialize',
                                  side_effect=AssertionError('session read'))
    second = BMicro._scan_batch_repetitions(path)

    assert second == first
    reader.assert_not_called()
    session_reader.assert_not_called()
    mocker.stopall()

    # bmlab still loads the session that carries the entry
    session = Session.get_instance()
    session.clear()
    session.set_file(path)
    assert session.file is not None
    session.clear()

    # a changed raw file is scanned again
    os.utime(path, ns=(time.time_ns(), time.time_ns() + 10 ** 9))
    assert batch_cache.read(path, session_path) is None
    assert BMicro._scan_batch_repetitions(path) == first


def test_progress_after_saving_comes_from_the_open_session(tmp_path, mocker):
    path = tmp_path / 'Water.h5'
    shutil.copy(DATA / 'Water.h5', path)
    session = Session.get_instance()
    session.clear()
    session.set_file(path)
    repetitions = BMicro._scan_batch_repetitions(path)
    session.save()  # writes a new file: without an entry
    session_path = tmp_path / 'Water.session.h5'
    assert batch_cache.read(path, session_path) is None
    session_reader = mocker.patch('bmicro.gui.main.Serializer.deserialize',
                                  side_effect=AssertionError('session read'))
    me = types.SimpleNamespace(
        _refresh_batch_progress=BMicro._refresh_batch_progress)

    BMicro._refresh_file_progress(
        me, {'path': path, 'repetitions': repetitions})

    session_reader.assert_not_called()
    assert batch_cache.read(path, session_path)['progress'] is not None
    session.clear()


def _state(**config):
    state = types.SimpleNamespace(
        batch_reanalyse=False, batch_forced={},
        batch_config={'export': {'export': False},
                      'evaluation': {'evaluate': True}},
        PROGRESS_STEPS=BMicro.PROGRESS_STEPS,
        STEP_PROGRESS=BMicro.STEP_PROGRESS,
        _selected_batch_repetitions=BMicro._selected_batch_repetitions)
    for name, value in config.items():
        setattr(state, name, value)
    return state


def _file(*progress, checked=True):
    return {'path': pathlib.Path('x.h5'), 'repetitions': {
        str(i): {'valid': True, 'checked': checked, 'progress': p}
        for i, p in enumerate(progress)}}


def test_files_whose_step_is_done_are_not_opened():
    needs = BMicro._batch_file_needs_step
    state = _state()
    assert not needs(state, _file('calibration', 'evaluation'), 'extraction')
    assert needs(state, _file('calibration', 'none'), 'extraction')
    assert needs(state, _file('extraction'), 'calibration')
    assert not needs(state, _file('none', checked=False), 'extraction')

    # uncertain or redone: look at the file
    assert needs(state, _file('unknown'), 'extraction')
    assert needs(state, _file(None), 'extraction')
    assert needs(state, {'path': 'x.h5', 'repetitions': {}}, 'extraction')
    state.batch_forced = {('x.h5', 'calibration'): {'0'}}
    assert needs(state, {'path': 'x.h5', 'repetitions': {
        '0': {'valid': True, 'checked': True, 'progress': 'peak selection'}}},
        'calibration')
    assert needs(_state(batch_reanalyse=True), _file('evaluation'),
                 'extraction')


def test_files_without_anything_left_to_evaluate_are_not_opened():
    needs = BMicro._batch_file_needs_evaluation
    assert not needs(_state(), _file('evaluation', 'evaluation'))
    assert needs(_state(), _file('evaluation', 'peak selection'))
    assert not needs(_state(), _file('peak selection', checked=False))

    export = _state()
    export.batch_config['export']['export'] = True  # needs the results
    assert needs(export, _file('evaluation'))
    assert needs(_state(batch_reanalyse=True), _file('evaluation'))

    nothing = _state()
    nothing.batch_config['evaluation']['evaluate'] = False
    assert not needs(nothing, _file('none'))


def test_every_repetition_with_data_is_ticked_by_default(tmp_path):
    path = tmp_path / 'Water_two_repetitions.h5'
    shutil.copy(DATA / 'Water_two_repetitions.h5', path)

    repetitions = BMicro._scan_batch_repetitions(path)

    assert len(repetitions) >= 2
    for rep in repetitions.values():
        assert rep['checked'] == rep['valid']
    assert sum(rep['valid'] for rep in repetitions.values()) >= 2
