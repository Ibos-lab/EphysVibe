import os

AREAS = list(os.environ.get("PIPELINES_AREAS", default=["lip", "v4", "pfc", "eyes"]))
N_CHANNELS = list(os.environ.get("PIPELINES_N_CHANNELS", default=[32, 64, 64, 3]))
