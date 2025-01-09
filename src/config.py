from pathlib import Path
from datetime import datetime

root_path = Path(__file__).parent.parent

RESOLUTION_DEFAULT = 250
MAX_DISTANCE_STOP_DEFAULT = 500
N_MAX_DESTINATION = 25

MAX_N_ITEMS_GOOGLE_MAPS_API_SINGLE_RUN = 100
MONTHLY_MAX_FREE_GOOGLE_API_CALLS = 40_000
DATETIME_TRANSIT_GOOGLE_API = datetime(2025, 1, 9, 8, 0, 0)
SCORE_COLUMN = "score"
TAG_PROVINCIA = "denominazione_provincia"

overpass_url = 'http://overpass-api.de/api/interptreter'

SHAPE_CRS = "EPSG:32632"
OVERPASS_CRS = "EPSG:4326"
GOOGLE_MAPS_CRS = "EPSG:4326"

province_shape_path = root_path / Path("data/input/province_shapes.parquet")
public_transport_log_path = root_path / Path(r"data/log/public_transport.log")
public_transport_routes_path = root_path / Path("data/input/routes.csv")
public_stop_path = root_path / Path("data/input/stops.parquet")
score_data_path = root_path / Path("data/output/public_transport_score_grid.csv")
