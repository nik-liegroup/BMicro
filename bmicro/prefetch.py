import logging
from concurrent.futures import ThreadPoolExecutor

from bmlab.session import Session

logger = logging.getLogger(__name__)


class FilePrefetcher(object):
    """
    Loads the next files of a batch in a background thread while the
    current one is being worked on, so that opening it takes only the
    part that has to happen on the main thread.

    What is loaded is Session.preload(): the opened file with its
    repetitions (positions, metadata) and its saved session - never the
    image data, which stays on disk. At most `max_files` files are held
    at a time; each is handed out once by take() and then belongs to the
    caller.
    """

    def __init__(self, max_files=5):
        self.max_files = max_files
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._futures = {}

    def request(self, paths):
        """
        Makes the first `max_files` of `paths` (the files due next, in
        order) the ones that are loaded or being loaded. Anything held
        that isn't among them any more is released.
        """
        wanted = [str(path) for path in paths][:self.max_files]
        for path in list(self._futures):
            if path not in wanted:
                self._release(self._futures.pop(path))
        for path in wanted:
            if path not in self._futures:
                self._futures[path] = self._executor.submit(
                    Session.preload, path)

    def take(self, path):
        """
        Returns the preloaded data for `path` for Session.set_file(), or
        None if there is none (not requested, or loading failed - the
        caller then just loads it itself). Waits if it is still loading.
        """
        future = self._futures.pop(str(path), None)
        if future is None:
            return None
        try:
            return future.result()
        except Exception:
            logger.warning('Could not preload %s', path, exc_info=True)
            return None

    @staticmethod
    def _release(future):
        def close(done):
            try:
                Session.close_preloaded(done.result())
            except Exception:
                pass
        future.add_done_callback(close)

    def shutdown(self):
        for future in self._futures.values():
            self._release(future)
        self._futures = {}
        self._executor.shutdown(wait=False)
