def _install_crash_log():
    """
    Writes the Python stack of every thread (and Qt's own warnings and
    fatal messages) to ~/.bmicro/crash.log. A hard crash inside Qt
    (e.g. a QThread destroyed while running) otherwise leaves no trace
    at all but a Windows error event.
    """
    import faulthandler
    import os
    import time
    from PyQt6 import QtCore

    folder = os.path.join(os.path.expanduser('~'), '.bmicro')
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, 'crash.log')
    # Kept open for the whole run: faulthandler writes to the raw handle
    log = open(path, 'a', buffering=1)
    log.write(f'--- BMicro started {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    faulthandler.enable(file=log, all_threads=True)

    def qt_message(mode, context, message):
        if mode != QtCore.QtMsgType.QtDebugMsg:
            log.write(f'{time.strftime("%H:%M:%S")} Qt {mode.name}: '
                      f'{message}\n')
            if mode == QtCore.QtMsgType.QtFatalMsg:
                faulthandler.dump_traceback(file=log, all_threads=True)

    QtCore.qInstallMessageHandler(qt_message)
    return path


def main():
    import os
    from importlib import resources
    import sys
    import logging

    from PyQt6 import QtGui, QtWidgets

    from bmicro.gui.main import BMicro
    from bmicro._version import version as __version__
    """
    Starts the BMicro application and handles its life cycle.
    """
    crash_log = _install_crash_log()
    app = QtWidgets.QApplication(sys.argv)
    # set window icon
    ref = resources.files('bmicro') / 'img'
    with resources.as_file(ref) as imdir:
        icon_path = os.path.join(imdir, "icon.png")
        app.setWindowIcon(QtGui.QIcon(icon_path))

    window = BMicro()
    window.show()

    for arg in sys.argv:
        if arg == '--version':
            print(__version__)
            QtWidgets.QApplication.processEvents()
            sys.exit(0)
        elif arg.startswith('--log='):
            log_level = arg[6:]
            logging.basicConfig(level=log_level)

    if sys.argv[-1].endswith('.h5'):
        window.open_file(sys.argv[-1])

    sys.exit(app.exec())


if __name__ == '__main__':
    # Necessary to make multiprocessing work with pyinstaller
    from multiprocessing import freeze_support, set_start_method
    freeze_support()
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()
