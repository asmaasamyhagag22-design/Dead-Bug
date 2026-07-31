"""The Dead Bug coach as a web app: camera, YouTube link, or uploaded file.

Three ways in, one engine. :mod:`deadbug.live.engine` does the work for all of
them, so a rep counted in the browser is counted by the same code that produced
the offline numbers.
"""

from .analysis import analyse_video, assess

__all__ = ["analyse_video", "assess"]
