"""
Setup and orientation are the base of every analysis step. Changing them
makes what was built on them outdated: a new orientation changes the
geometry of the extraction points (so every step from extraction on is
outdated), a new setup changes the calibration (every step from
calibration on). bmlab itself does not clear anything when they change,
so this is done here, for the batch evaluation and for the Data tab.
"""
import math

import numpy as np

from bmlab.models import ExtractionModel, CalibrationModel, \
    PeakSelectionModel, EvaluationModel


def setup_signature(setup):
    """ What makes two setups different for the analysis. """
    if setup is None:
        return None
    calibration = getattr(setup, 'calibration', None)
    return (getattr(setup, 'key', None),
            getattr(setup, 'temperature', None),
            getattr(calibration, 'shift_methanol', None),
            getattr(calibration, 'shift_water', None))


def orientation_signature(rotation, reflection):
    return (int(rotation) % 4, bool(reflection['vertically']),
            bool(reflection['horizontally']))


def saved_base_signature(models):
    """
    (setup signature, orientation signature) of the models read from a
    session file; None if the file does not have them.
    """
    try:
        return (setup_signature(models.setup),
                orientation_signature(models.orientation.rotation,
                                      models.orientation.reflection))
    except Exception:
        return None


def has_analysis(session, key):
    """
    Whether repetition `key` of the open file has anything analysed:
    extraction points, calibration fits, peak regions or evaluation
    results.
    """
    extraction = session.extraction_models.get(key)
    calibration = session.calibration_models.get(key)
    peak_selection = session.peak_selection_models.get(key)
    evaluation = session.evaluation_models.get(key)
    if extraction is not None and any(extraction.points.values()):
        return True
    if calibration is not None and calibration.frequencies:
        return True
    if peak_selection is not None and (
            getattr(peak_selection, 'brillouin_regions_f', None)
            or getattr(peak_selection, 'rayleigh_regions_f', None)):
        return True
    try:
        return evaluation is not None and bool(np.any(np.isfinite(
            evaluation.results['brillouin_peak_position_f'])))
    except Exception:
        return False


def reset_analysis(session, keys, from_extraction):
    """
    Replaces the analysis of the repetitions `keys` of the open file by
    empty models: calibration, peak selection and evaluation, and with
    `from_extraction` the extraction points too. The extraction models
    get the image shape and arc width a freshly loaded file would have.
    """
    for key in keys:
        if from_extraction:
            session.extraction_models[key] = ExtractionModel()
        session.calibration_models[key] = CalibrationModel()
        session.peak_selection_models[key] = PeakSelectionModel()
        session.evaluation_models[key] = EvaluationModel()
    if from_extraction:
        session.set_image_shape()  # for every repetition, depends only
        # on the current orientation and crop
        for key in keys:
            payload = session.file.get_repetition(key).payload
            image_keys = payload.image_keys()
            if image_keys:
                model = session.extraction_models[key]
                model.set_arc_width(math.ceil(
                    model.arc_width / payload.get_binning_factor(
                        image_keys[0])))
