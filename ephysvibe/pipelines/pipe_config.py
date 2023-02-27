import os

area_start_nch = {
    "lip": [0, 32],
    "v4": [32, 64],
    "pfc": [96, 64],
    "eyes": [160, 3],
}  # area:[start ch, n ch]
AREAS = os.environ.get("PIPELINES_AREAS", default=area_start_nch)
TOTAL_CH = int(os.environ.get("PIPELINES_TOTAL_CH", default=163))
