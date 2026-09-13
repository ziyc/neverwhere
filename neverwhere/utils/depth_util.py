from asyncio import sleep

import numpy as np
from contextlib import contextmanager

def z2r(z, fov, *, h, w):
    """
    Convert metric depth to range map

    :param z: metric depth, in Size(H, W)
    :param fov: verticle field of view
    :param h: height of the image
    :param w: width of the image
    """
    f = h / 2 / np.tan(fov / 2)
    x = (np.arange(w) - w / 2) / f
    y = (np.arange(h) - h / 2) / f
    xs, ys = np.meshgrid(x, y, indexing="xy")

    d = z * np.sqrt(1 + xs ** 2 + ys ** 2)

    return d, xs, ys

@contextmanager
def invisibility(physics, geom_names: list):
    try:
        # Check if the geom exists
        original_rgbas = []
        for geom_name in geom_names:
            original_rgba = physics.named.model.geom_rgba[geom_name].copy()
            # Set RGBA to fully transparent to hide the geom
            physics.named.model.geom_rgba[geom_name] = [0, 0, 0, 0]
            original_rgbas.append(original_rgba)

        geom_exists = True
    except KeyError:
        # print(f"Geom '{geom_name}' does not exist in the model.")
        geom_exists = False

    try:
        yield
    finally:
        if geom_exists:
            # Restore original RGBA color if the geom existed
            for geom_name, original_rgba in zip(geom_names, original_rgbas):
                physics.named.model.geom_rgba[geom_name] = original_rgba
