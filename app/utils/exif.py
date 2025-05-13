from PIL import Image, ExifTags
from typing import Optional, Tuple

def get_image_gps(path: str) -> Tuple[Optional[float], Optional[float]]:
    img = Image.open(path)
    info = img._getexif() or {}

    # find the GPSInfo tag code
    gps_tag = next(
        (tag for tag, name in ExifTags.TAGS.items() if name == "GPSInfo"),
        None
    )
    if gps_tag not in info:
        return None, None

    gps = info[gps_tag]

    def _to_deg(vals):
        # vals may be [(num,den), ...] or already floats/IFDRationals
        d, m, s = vals
        def to_float(x):
            try:
                # tuple/list of (num, den)
                return x[0] / x[1]
            except Exception:
                # IFDRational or plain number
                return float(x)
        return to_float(d) + to_float(m) / 60.0 + to_float(s) / 3600.0

    try:
        lat = _to_deg(gps[2])
        if gps.get(1) == 'S':
            lat = -lat
        lon = _to_deg(gps[4])
        if gps.get(3) == 'W':
            lon = -lon
        return lat, lon
    except Exception:
        # any parsing error, just return None
        return None, None
