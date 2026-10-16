"""
What the batch table needs to know about a source file - its repetitions
(read from the large raw file) and how far each is analysed (which
otherwise means deserialising the whole saved session, seconds per file)
- is kept inside the session file itself, as one small attribute of the
file ('bmicro_batch', a JSON text), so a lookup takes milliseconds.

bmlab ignores the attribute (it only reads the 'session' group and the
'version' attribute), and Session.save() writes a new file every time,
which drops it: after a save, the entry is added again by the batch
(see BMicro._refresh_file_progress()); a session saved elsewhere simply
has none until the next look, which reads it the slow way once and adds
it. The entry also records the raw file's size and modification time and
is only used while they are unchanged, so a changed raw file (or a
session edited together with it) can never be mistaken for current.
"""
import json
import logging
import os

import h5py

logger = logging.getLogger(__name__)

# Bump when what the entries mean changes (e.g. how progress is derived)
VERSION = 1
ATTRIBUTE = 'bmicro_batch'
REPETITION_KEYS = ('label', 'valid', 'nr_spectra', 'calib_keys')
PROGRESS_KEYS = ('progress', 'warning')


def file_stamp(path):
    """ Size and modification time of a file, None if it is missing. """
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return [stat.st_size, stat.st_mtime_ns]


def _tuples(value):
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    return value


def read(raw_path, session_path):
    """
    The entry stored in the session file, or None if there is no usable
    one (no session, no entry, other version, raw file changed since):
    {'repetitions': {key: {...}}, 'progress': {key: {...}},
    'saved_base': (setup, orientation) signatures or None}.
    """
    try:
        with h5py.File(session_path, 'r') as f:
            text = f.attrs.get(ATTRIBUTE)
        if text is None:
            return None
        data = json.loads(text)
        if data.get('version') != VERSION \
                or data['raw'] != file_stamp(raw_path):
            return None
        return {
            'repetitions': {
                key: {name: info[name] for name in REPETITION_KEYS}
                for key, info in data['repetitions'].items()},
            'progress': {
                key: {name: info[name] for name in PROGRESS_KEYS}
                for key, info in data['progress'].items()},
            'saved_base': _tuples(data['saved_base']),
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def write(raw_path, session_path, repetitions, saved_base):
    """
    Stores the batch table's repetitions of a file (`repetitions` as in
    the table) with their progress in its session file. Never creates a
    file: without a session there is nothing to store it in. Returns
    whether it was written.
    """
    data = {
        'version': VERSION,
        'raw': file_stamp(raw_path),
        'repetitions': {
            key: {name: rep[name] for name in REPETITION_KEYS}
            for key, rep in repetitions.items()},
        'progress': {
            key: {name: rep.get(name) for name in PROGRESS_KEYS}
            for key, rep in repetitions.items()},
        'saved_base': saved_base,
    }
    try:
        text = json.dumps(data)
        with h5py.File(session_path, 'r+') as f:
            if 'session' not in f:  # not a session file
                return False
            f.attrs[ATTRIBUTE] = text
    except (OSError, RuntimeError, TypeError, ValueError):
        # a missing session, a file in use, an entry too large for an
        # attribute...: the next look just reads it the slow way
        logger.debug('Could not store %s in %s', ATTRIBUTE, session_path,
                     exc_info=True)
        return False
    return True
