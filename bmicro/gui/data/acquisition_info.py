"""
What BrillouinAcquisition recorded about how a repetition was measured:
how it ended, the grid, the overviews and what the drift correction did.

Shown in the Data tab. The text is made here, from what bmlab reads, so
that it can be tested without a window.
"""
import logging

logger = logging.getLogger(__name__)

# a repetition that did not end normally
NOT_FINISHED = {
    'preparing': 'never started to measure',
    'surface-scan': 'stopped during the surface scan',
    'surface-scan-stopped': 'stopped during the surface scan',
    'measuring': 'interrupted while measuring (crash or power loss)',
    'stopped': 'stopped (cancelled)',
}


def _grid_lines(payload):
    grid = payload.get_grid_definition()
    if grid is None:
        return []
    lines = []
    for axis in ('x', 'y', 'z'):
        text = f"{grid['steps'][axis]} points"
        if grid['range_um'][axis] is not None:
            low, high = grid['range_um'][axis]
            text += f", {low:.1f} to {high:.1f} um"
        if grid['step_um'][axis] is not None:
            text += f", every {grid['step_um'][axis]:.2f} um"
        lines.append(f"  {axis}: {text}")
    if grid['scan_loop_level'] is not None:
        order = sorted(grid['scan_loop_level'].items(), key=lambda i: i[1])
        lines.append('  scan order (innermost loop first): ' + ', '.join(
            axis for axis, _ in order))
    if grid['coordinates_absolute'] is not None:
        lines.append('  coordinates: ' + (
            'absolute (stage)' if grid['coordinates_absolute']
            else 'relative to the start position'))
    if grid['roi_polygon_um'] is not None:
        lines.append(
            f"  ROI polygon drawn at acquisition: "
            f"{grid['roi_polygon_um'][0].size} corners")
    return ['Grid:'] + lines


def _overview_lines(payload):
    events = payload.get_overview_events()
    objectives = payload.get_overview_objectives()
    if not events and not objectives:
        return []
    lines = ['Overviews:']
    if events:
        kinds = {}
        for event in events:
            kinds[event['kind_name']] = kinds.get(event['kind_name'], 0) + 1
        lines.append('  ' + str(len(events)) + ' events (' + ', '.join(
            f'{count} {name}' for name, count in kinds.items()) + ')')
    for slot, objective in objectives.items():
        text = f"  objective in slot {slot}"
        if objective['magnification']:
            text += f", {objective['magnification']:g}x"
        if objective['is_reference_objective']:
            text += ', reference objective'
        elif objective['has_fov_offset'] and objective['fov_offset']:
            text += ", FOV offset ({:.1f}, {:.1f}) um".format(
                *objective['fov_offset'])
        lines.append(text)
    return lines


def _drift_lines(payload):
    drift = payload.get_runtime_drift()
    if drift is None:
        return []
    lines = ['Drift correction while measuring: on']
    detail = []
    if drift['objective_slot'] is not None:
        detail.append(f"objective slot {drift['objective_slot']:g}")
    if drift['min_ncc'] is not None:
        detail.append(f"minimum correlation {drift['min_ncc']:g}")
    if detail:
        lines.append('  ' + ', '.join(detail))
    events = drift['events']
    decisions = {}
    for event in events:
        name = event['decision_name'] or 'unknown'
        decisions[name] = decisions.get(name, 0) + 1
    if decisions:
        lines.append('  steps: ' + ', '.join(
            f'{count} {name}' for name, count in decisions.items()))
    if any(event.get('reference_replaced') for event in events):
        lines.append(
            '  the reference of the previous run could not be used: the '
            'first overview of this run became the new reference')
    if drift['applied']:
        lines.append(
            f"  the grid followed the sample, by up to "
            f"{drift['max_offset_um']:.1f} um")
        lines.append('  positions x, y of the export are positions on the '
                     'sample')
    elif events:
        lines.append('  NOTHING WAS CORRECTED: no overview could be matched '
                     'to the reference.')
        lines.append('  The drift correction after the measurement is still '
                     'needed for this repetition.')
        stale = events[0]['seconds_since_reference']
        if events[0]['decision_name'] == 'no match' and stale:
            lines.append(
                f"  (the reference was already {stale / 60:.0f} min old at "
                f"the first overview: taken over from an earlier run)")
    return lines


def acquisition_summary(repetition):
    """
    Plain text about how `repetition` (a bmlab Repetition) was measured;
    '' if the file carries nothing of it.
    """
    lines = []
    try:
        state = repetition.acquisition_state
        if state is not None:
            if state == 'finished':
                lines.append('Finished normally.')
            else:
                lines.append('NOT FINISHED: ' + NOT_FINISHED.get(
                    state, state) + '.')
        payload = repetition.payload
        resumed = payload.get_resume_events()
        if resumed:
            lines.append(f'Continued {len(resumed)} time(s) after a stop.')
        for part in (_grid_lines, _overview_lines, _drift_lines):
            block = part(payload)
            if block:
                lines.extend([''] + block if lines else block)
        rescans = payload.get_surface_rescans()
        if rescans:
            failed = sum(1 for r in rescans if not r['success'])
            lines.extend(['', f'Surface scanned again {len(rescans)} time(s)'
                          + (f', {failed} failed' if failed else '') + '.'])
    except Exception:
        logger.warning('Could not summarise the acquisition', exc_info=True)
    return '\n'.join(lines)
