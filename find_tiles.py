import sys
import json
import urllib.parse
import urllib.request

lon, lat = float(sys.argv[1]), float(sys.argv[2])
d = 0.005
params = {
    "bbox": f"{lon - d},{lat - d},{lon + d},{lat + d}",
    "datasets": "Lidar Point Cloud (LPC)",
    "prodFormats": "LAZ",
    "max": 20,
}
url = "https://tnmaccess.nationalmap.gov/api/v1/products?" + urllib.parse.urlencode(params)
print(url)
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research project)"})
with urllib.request.urlopen(req, timeout=60) as r:
    data = json.load(r)
print("Total products:", data.get("total"))
for i, it in enumerate(data.get("items", [])):
    mb = (it.get("sizeInBytes") or 0) / 1e6
    print(i, round(mb, 1), "MB", it.get("publicationDate"), it.get("title"))
    print("    ", it.get("downloadURL"))
