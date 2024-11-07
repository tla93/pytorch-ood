"""

"""
import os
from os.path import join
from typing import Callable, Optional

from .base import ImageDatasetBase


class FoolingImages(ImageDatasetBase):
    """
    From the paper *Deep neural networks are easily fooled: High confidence predictions for unrecognizable images*.

    .. image::  https://i.stack.imgur.com/pBm48.png
        :width: 800px
        :alt: Fooling Images
        :align: center


    :see Website: `Link <https://anhnguyen.me/project/fooling/>`__
    :see Paper: `ArXiv <https://arxiv.org/pdf/1412.1897>`__
    """

    dirs = [f"run_{i}" for i in range(10)]

    base_folder = "10-runs-x-1000-cppns"
    url = "https://s.anhnguyen.me/10_runs_x_1000_cppns.tar.gz"
    filename = "10_runs_x_1000_cppns.tar.gz"
    md5hash = "0910d63973b1512770f37bebdbb53e37"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super(FoolingImages, self).__init__(root, transform, target_transform, download)

    def _load_files(self):
        self.basedir = os.path.join(self.root, self.base_folder)
        files = []
        for d in self.dirs:
            p = join(self.basedir, d, "map_gen_5000")
            files += [join(p, f) for f in os.listdir(p) if f.endswith(".png")]

        return files
