import logging
import pathlib

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle as MPLCircle

from bmlab.controllers import CalibrationController
from bmlab.session import get_session_file_path

logger = logging.getLogger(__name__)


def plots_folder(source_path):
    """
    The folder the analysis plots of a source file are saved in: next to
    its session file (i.e. in the EvalData folder, if there is one),
    named like the session file plus '_plots'.
    """
    session_path = pathlib.Path(get_session_file_path(source_path))
    name = session_path.name
    for suffix in ('.session.h5', '.h5'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return session_path.parent / f'{name}_plots'


def evaluated_repetitions(session):
    """ Keys of the repetitions of the open file with evaluation results. """
    keys = []
    for key, evm in session.evaluation_models.items():
        try:
            if np.any(np.isfinite(evm.results['brillouin_peak_position_f'])):
                keys.append(key)
        except Exception:
            pass
    return keys


def _save(fig, path):
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=150, bbox_inches='tight')


def plot_extraction_image(ax, session, calib_key, frame):
    """ The calibration image with its crop, points and arc. """
    em = session.extraction_model()
    img = session.get_calibration_image(calib_key, frame)
    ax.imshow(img.T, origin='lower', vmin=100, vmax=300)

    points = em.get_points(calib_key)
    if len(points) >= 3:
        em.set_image_shape(img.shape)
        for arc in em.get_arc_by_calib_key(calib_key):
            dr = arc[-1] - arc[0]
            ax.add_patch(matplotlib.patches.FancyArrow(
                *arc[0], dr[0], dr[1], head_width=0, head_length=0,
                color='Yellow', alpha=0.5))
    for point in points:
        ax.add_patch(MPLCircle(point, radius=3, color='red', alpha=0.5))
    ax.set_title(f'Calibration {calib_key} - Frame {frame + 1}')


def plot_calibration_spectrum(ax, session, calib_key, frame):
    """
    The calibration spectrum with the fit regions, fitted peaks and the
    expected calibration frequencies. Returns False if there is no
    spectrum to show (no arc yet).
    """
    cc = CalibrationController()
    spectra, _, _ = cc.extract_spectra(calib_key, frame_num=frame)
    cm = session.calibration_model()
    if spectra is None or not cm or len(spectra) == 0:
        return False
    spectrum = spectra[0]

    frequencies = cm.get_frequencies_by_calib_key(calib_key)
    frequency = None
    if frequencies:
        frequency = 1e-9 * frequencies[frame]
        ax.plot(frequency, spectrum)
        ax.set_xlabel('$f$ [GHz]')
        ax.set_xlim(1e-9 * np.min(frequencies), 1e-9 * np.max(frequencies))
    else:
        ax.plot(spectrum)
        ax.set_xlabel('$f$ [pix]')
        ax.set_xlim(0, len(spectrum))
    ax.set_ylim(bottom=0)
    ax.set_title(f'Calibration {calib_key} - Frame {frame + 1}')
    top = np.nanmax(spectrum)

    for regions, color in ((cm.get_brillouin_regions(calib_key), 'r'),
                           (cm.get_rayleigh_regions(calib_key), 'm')):
        for region in regions:
            mask = np.arange(int(region[0]), int(region[1]))
            x = frequency[mask] if frequency is not None else mask
            ax.plot(x, spectrum[mask], color)

    for region_key in range(len(cm.get_brillouin_regions(calib_key))):
        fit = cm.brillouin_fits.get_fit(calib_key, region_key, frame)
        if fit is None:
            continue
        w0s = fit.w0s
        w0s_f = cm.get_frequency_by_calib_key(w0s, calib_key)
        positions = 1e-9 * w0s_f[:2] if w0s_f is not None else w0s[:2]
        ax.vlines(positions, 0, top, colors=['black'])

    for region_key in range(len(cm.get_rayleigh_regions(calib_key))):
        fit = cm.rayleigh_fits.get_fit(calib_key, region_key, frame)
        if fit is None:
            continue
        w0_f = cm.get_frequency_by_calib_key(fit.w0, calib_key)
        ax.vlines(1e-9 * w0_f if w0_f is not None else fit.w0,
                  0, top, colors=['black'])

    expected = cc.expected_frequencies(calib_key, frame)
    if expected is not None:
        ax.vlines(1e-9 * expected, 0, top, colors=['green'])
    return True


def export_calibration_plots(session, repetitions=None):
    """
    Saves PNGs of everything shown in the extraction and calibration
    tabs for the open file: per repetition, calibration and frame, the
    2D image with crop, points and arc, and the 1D spectrum with fit
    regions, fitted peaks and calibration lines. The 1D plot is only
    written once the calibration has an arc.

    Returns the folder the plots were written to, or None if the file
    has no calibrations.
    """
    if session.file is None:
        return None
    folder = plots_folder(session.file.path)
    previous = session.current_repetition_key()
    written = False
    try:
        for rep_key in session.file.repetition_keys():
            if repetitions is not None and rep_key not in repetitions:
                continue
            session.set_current_repetition(rep_key)
            for calib_key in session.get_calib_keys(sort_by_time=True) or []:
                frame_count = session.get_calibration_image_count(calib_key)
                for frame in range(frame_count):
                    stem = (f'rep{rep_key}_calibration{calib_key}'
                            f'_frame{frame + 1}')
                    folder.mkdir(parents=True, exist_ok=True)
                    written = True
                    for kind, draw in (('2D', plot_extraction_image),
                                       ('1D', plot_calibration_spectrum)):
                        fig = Figure(figsize=(8, 5))
                        ax = fig.add_subplot(111)
                        try:
                            if draw(ax, session, calib_key, frame) is False:
                                continue
                            _save(fig, folder / f'{stem}_{kind}.png')
                        except Exception:
                            logger.warning(
                                'Could not plot %s %s', stem, kind,
                                exc_info=True)
    finally:
        session.set_current_repetition(previous)
    return folder if written else None
