from bmicro.gui.evaluation.evaluation_view import EvaluationView


def test_get_decimation_stride():
    """
    Regression test for the rotate-drag LOD used by the 3D
    evaluation plot (_draw_3d_surfaces / on_3d_button_press):
    small grids should render at full detail, dense ones should be
    decimated.

    Note: a GUI-level test that actually renders the 3D matplotlib
    plot (_draw_3d_surfaces itself) was attempted here but had to be
    dropped - it reproducibly crashes pytest-qt's test process in
    this environment (exit code 127) even though the *identical* code
    runs correctly as a standalone script (verified manually, see PR
    discussion). This looks like a pytest-qt / matplotlib mplot3d
    interaction issue in this environment, not a bug in the LOD code
    itself, but landing a test that crashes the suite isn't
    acceptable - only the dependency-free stride calculation is
    covered here.
    """
    # 3*5*7 = 105 points, well under the default 600-point threshold
    assert EvaluationView._get_decimation_stride((3, 5, 7)) == 1
    # 11*30*16 = 5280 points, well over it
    assert EvaluationView._get_decimation_stride((11, 30, 16)) > 1
