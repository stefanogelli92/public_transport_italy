import os
import time
import logging
from logging.handlers import RotatingFileHandler
import csv
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from sklearn.cluster import KMeans
import overpy
import googlemaps

import src.config as cfg
from src.secret import GOOGLE_API


class PublicTransportAnalysis:

    def __init__(self, initial_n_api_call = 0):
        self.logger = self.setup_logger()
        self.n_api_call = initial_n_api_call

    @staticmethod
    def setup_logger(
        log_file: Path = cfg.public_transport_log_path,
        max_file_size: int = 5 * 1024 * 1024,
        backup_count: int = 3
    ) -> logging.Logger:
        """
        Set up a logger with a rotating file handler and console handler.

        Parameters:
            log_file (Path): Path to the log file.
            max_file_size (int): Maximum size of the log file in bytes before rotation (default 5 MB).
            backup_count (int): Number of backup files to keep (default 3).

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger('_public_transport')
        logger.setLevel(logging.DEBUG)

        # Avoid adding duplicate handlers
        if not logger.handlers:
            # Create log directory if it does not exist
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # File handler
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_file_size, backupCount=backup_count
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)

            # Adding handlers to logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger

    def generate_grid_with_resolutions(
        self,
        polygon: Union[Polygon, MultiPolygon],
        resolutions: Union[float, List[float]]
    ) -> gpd.GeoDataFrame:
        """
        Generate a regular grid of points within a given polygon with multiple resolutions.

        Parameters:
            polygon (Union[Polygon, MultiPolygon]): Polygon or MultiPolygon to generate the grid within.
            resolutions (Union[float, List[float]]): Grid resolutions in meters.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing grid points and their associated maximum resolutions.
        """
        # Ensure resolutions is a list
        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        # Logging start
        self.logger.debug(f"Start generating grid for polygon with resolution {min(resolutions)}m.")
        start_time = datetime.now()

        # Validate polygon input
        if isinstance(polygon, Polygon):
            polygon_list = [polygon]
        elif isinstance(polygon, MultiPolygon):
            polygon_list = list(polygon.geoms)
        else:
            raise ValueError("The input must be a Polygon or MultiPolygon.")

        # Initialize results
        grid_points = []
        resolution_labels = []

        for single_polygon in polygon_list:
            minx, miny, maxx, maxy = single_polygon.bounds

            # Create grid coordinates
            x_coords = np.arange(minx, maxx, min(resolutions))
            y_coords = np.arange(miny, maxy, min(resolutions))
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)

            # Combine coordinates into shapely Points
            potential_points = [Point(x, y) for x, y in zip(x_grid.ravel(), y_grid.ravel())]

            # Filter points within the polygon
            filtered_points = list(filter(single_polygon.contains, potential_points))

            # Assign the maximum resolution for each point
            for point in filtered_points:
                max_resolution = max(
                    (res for res in resolutions if (point.x - minx) % res == 0 and (point.y - miny) % res == 0),
                    default=None
                )
                grid_points.append(point)
                resolution_labels.append(max_resolution)

        # Construct GeoDataFrame
        grid_gdf = gpd.GeoDataFrame(
            {"resolution": resolution_labels},
            geometry=grid_points,
            crs=cfg.SHAPE_CRS
        )

        # Logging end
        end_time = datetime.now()
        self.logger.debug(f"Generated {len(grid_gdf)} points in grid in {end_time - start_time}.")

        return grid_gdf

    @staticmethod
    def download_public_transport_stops(output_path: Path = cfg.public_stop_path) -> gpd.GeoDataFrame:
        """
        Download public transport stops using the Overpass API and save them as a GeoDataFrame.

        Parameters:
            output_path (Path): File path to save the resulting GeoDataFrame in Parquet format.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing public transport stops with geometry and city information.
        """
        # Initialize Overpass API client
        api = overpy.Overpass()

        # Query Overpass API for public transport stops in Italy
        query = """
            area["ISO3166-1"="IT"][admin_level=2];
            (
              node["public_transport"](area);
            );
            out body;
        """
        result = api.query(query)

        # Extract stops data
        stops_data = []
        for node in result.nodes:
            stop_name = node.tags.get("name", "N/A")
            latitude = node.lat
            longitude = node.lon
            stops_data.append((stop_name, latitude, longitude))

        # Convert to DataFrame
        stops_df = pd.DataFrame(stops_data, columns=["name", "latitude", "longitude"])

        # Create GeoDataFrame with point geometry
        stops_gdf = gpd.GeoDataFrame(
            stops_df.drop(columns=["latitude", "longitude"]),
            geometry=gpd.points_from_xy(stops_df.longitude, stops_df.latitude),
            crs=cfg.OVERPASS_CRS,
        )

        # Reproject to target CRS
        stops_gdf = stops_gdf.to_crs(cfg.SHAPE_CRS)

        # Add city information
        df_province = gpd.read_parquet(cfg.province_shape_path)
        stops_gdf = gpd.sjoin(stops_gdf, df_province, predicate="within", how="left")
        stops_gdf = stops_gdf[["name", "geoemtry", cfg.TAG_PROVINCIA]]

        # Save to Parquet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stops_gdf.to_parquet(output_path)

        return stops_gdf

    @staticmethod
    def get_public_stops(path = cfg.public_stop_path):
        stops = gpd.read_parquet(path)
        if stops.crs is None:
            stops.set_crs(cfg.SHAPE_CRS)
        return stops

    def filter_grid_points(
        self,
        grid_points: gpd.GeoDataFrame,
        transport_stops: gpd.GeoDataFrame,
        max_distance: float = cfg.MAX_DISTANCE_STOP_DEFAULT,
    ) -> gpd.GeoDataFrame:
        """
        Filter grid points that are within a specified maximum distance from public transport stops.

        Parameters:
            grid_points (gpd.GeoDataFrame): GeoDataFrame containing the grid points.
            transport_stops (gpd.GeoDataFrame): GeoDataFrame containing transport stop locations.
            max_distance (float): Maximum allowable distance (in meters) for grid points from any stop.

        Returns:
            gpd.GeoDataFrame: Filtered GeoDataFrame containing grid points within the specified distance.
        """
        self.logger.info(f"Start filtering grid points with max distance: {max_distance}m.")
        start_time = datetime.now()

        initial_count = len(grid_points)

        # Ensure transport_stops geometry is buffered by the specified distance
        stop_buffer = transport_stops.geometry.buffer(max_distance)

        # Combine all buffers into a single MultiPolygon for efficient spatial filtering
        combined_buffer = stop_buffer.unary_union

        # Filter grid points within the combined buffer
        filtered_points = grid_points[grid_points.geometry.within(combined_buffer)]

        elapsed_time = datetime.now() - start_time
        self.logger.info(f"Filtered grid points from {initial_count} to {len(filtered_points)} in {elapsed_time}.")
        return filtered_points

    def calculate_clusters(
        self,
        gdf: gpd.GeoDataFrame,
        num_clusters: int = cfg.N_MAX_DESTINATION,
    ) -> gpd.GeoDataFrame:
        """
        Group points into clusters using KMeans clustering.

        Parameters:
            gdf (GeoDataFrame): GeoDataFrame containing points with a geometry column.
            num_clusters (int): Number of clusters to form.

        Returns:
            GeoDataFrame: GeoDataFrame with an additional column 'cluster' indicating cluster assignments.
        """
        if 'geometry' not in gdf:
            raise ValueError("Input GeoDataFrame must contain a 'geometry' column.")

        if len(gdf) < num_clusters:
            raise ValueError(
                f"Number of clusters ({num_clusters}) exceeds the number of points in the dataset ({len(gdf)}).")

        # Extract coordinates for clustering
        coordinates = np.array([[point.x, point.y] for point in gdf.geometry])

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        gdf['cluster'] = kmeans.fit_predict(coordinates)

        self.logger.info(f"Assigned {num_clusters} clusters to {len(gdf)} points.")
        return gdf

    def aggregate_clusters(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Aggregate points in each cluster to find the cluster center and count.

        Parameters:
            gdf (GeoDataFrame): GeoDataFrame with points and a 'cluster' column.

        Returns:
            GeoDataFrame: Aggregated GeoDataFrame with one row per cluster.
        """
        if 'cluster' not in gdf:
            raise ValueError("GeoDataFrame must contain a 'cluster' column for aggregation.")

        aggregated_data = []

        # Group by cluster and calculate size and central point
        for cluster, group in gdf.groupby('cluster'):
            cluster_size = len(group)

            # Find the most central point in the cluster
            center_index = group.geometry.apply(
                lambda p: group.geometry.distance(p).sum()
            ).idxmin()
            center_point = group.geometry.loc[center_index]

            aggregated_data.append({'cluster': cluster, 'size': cluster_size, 'geometry': center_point})

        result_gdf = gpd.GeoDataFrame(aggregated_data, geometry='geometry', crs=gdf.crs)
        self.logger.info(f"Aggregated {len(gdf)} points into {len(result_gdf)} clusters.")
        return result_gdf

    def load_cached_results(
        self, output_file: str, provincia_name: str
    ) -> Optional[pd.DataFrame]:
        """
        Load cached results from a file, filtering by the specified province name.

        Parameters:
            output_file (str): Path to the file containing cached results.
            provincia_name (str): Name of the province to filter the results.

        Returns:
            Optional[pd.DataFrame]: DataFrame with cached results filtered by province, or None if the file does not exist.
        """
        if not os.path.exists(output_file):
            self.logger.warning(f"Cache file {output_file} does not exist.")
            return None

        try:
            # Load the cached data
            df = pd.read_csv(output_file)

            # Validate required columns
            required_columns = {"provincia_name", "destination_x", "destination_y", "origin_x", "origin_y"}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns in the cache file: {missing_columns}")

            # Filter data for the specified province
            df = df[df["provincia_name"] == provincia_name]

            # Create origin and destination keys
            df["destination_key"] = self._create_key(df, "destination_x", "destination_y")
            df["origin_key"] = self._create_key(df, "origin_x", "origin_y")

            self.logger.info(f"Loaded {len(df)} cached results for province: {provincia_name}.")
            return df

        except Exception as e:
            self.logger.error(f"Error loading cached results from {output_file}: {e}")
            return None

    @staticmethod
    def _create_key(
        df: gpd.GeoDataFrame,
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        geometry_column: Optional[str] = None
    ) -> pd.Series:
        """
        Generate a unique key for each row in a DataFrame based on coordinates.

        Parameters:
            df (gpd.GeoDataFrame): DataFrame or GeoDataFrame containing the data.
            x_column (Optional[str]): Name of the column for the x-coordinate.
            y_column (Optional[str]): Name of the column for the y-coordinate.
            geometry_column (Optional[str]): Name of the geometry column, if using GeoDataFrame.

        Returns:
            pd.Series: Series containing the unique keys for each row.
        """
        if x_column and y_column:
            # Generate key from x and y columns
            return (
                    df[x_column].round(12).astype(str) + "-" +
                    df[y_column].round(12).astype(str)
            )
        elif geometry_column:
            # Ensure the DataFrame has the required CRS and a valid geometry column
            if not isinstance(df, gpd.GeoDataFrame):
                raise TypeError("A GeoDataFrame is required when using 'geometry_column'.")
            if geometry_column not in df:
                raise ValueError(f"Geometry column '{geometry_column}' not found in DataFrame.")
            if df.crs is None:
                raise ValueError("GeoDataFrame must have a defined CRS.")

            # Temporarily convert CRS for key generation
            original_crs = df.crs
            df = df.to_crs(cfg.GOOGLE_MAPS_CRS)
            keys = (
                    df[geometry_column].x.round(12).astype(str) + "-" +
                    df[geometry_column].y.round(12).astype(str)
            )
            df = df.to_crs(original_crs)  # Restore original CRS
            return keys
        else:
            raise ValueError("Either 'x_column' and 'y_column', or 'geometry_column' must be provided.")

    def save_route_duration(
        self,
        output_file: str,
        point1: tuple,
        point2: tuple,
        provincia_name: str,
        duration: float,
        distance: float
    ) -> None:
        """
        Save a result to a CSV file.

        Parameters:
            output_file (str): Path to save results.
            point1 (tuple): Starting point as (longitude, latitude).
            point2 (tuple): Destination as (longitude, latitude).
            provincia_name (str): Name of the provincia.
            duration (float): Total travel time in seconds.
            distance (float): Air distance between the origin and the destination.
        """
        if not os.path.exists(output_file):
            # Write header if the file does not exist
            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "origin_longitude", "origin_latitude",
                    "destination_longitude", "destination_latitude",
                    "provincia_name", "duration_seconds", "distance_meters"
                ])

        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                point1[1], point1[0],
                point2[1], point2[0],
                provincia_name, duration, distance
            ])

        self.logger.debug(
            f"Saved result: origin={point1}, destination={point2}, "
            f"provincia={provincia_name}, duration={duration}s, distance={distance}m."
        )

    def calculate_durations_distances(
        self,
        origins: gpd.GeoDataFrame,
        destinations: gpd.GeoDataFrame,
        api_key: str,
        output_file: str,
        provincia_name: str,
        departure_time: datetime,
    ):
        """
        Use Google Maps Distance Matrix API to calculate travel times and distances.

        Parameters:
            origins (GeoDataFrame): Dataset with all the point of the final grid.
            destinations (GeoDataFrame): Dataset with all the point of the evaluation grid.
            api_key (str): Google Maps API key.
            output_file (str): Path to the CSV file for saving results.
            provincia_name (str): Name of the provincia.
            departure_time (datetime): Datetime for the google maps api calculation.
        """
        # Load cached results
        cached_results = self.load_cached_results(output_file, provincia_name)

        # Initialize the output file if not already present
        if not os.path.exists(output_file):
            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["origin_x", "origin_y", "destination_x", "destination_y", "provincia_name", "duration", "distance"])

        # Generate unique keys for origins and destinations
        destinations["key"] = self._create_key(destinations, geometry_column="geometry")
        origins["key"] = self._create_key(origins, geometry_column="geometry")

        if (cached_results is not None) and (len(cached_results) > 0):
            # Filter destinations already processed for all origins
            destinations = self._filter_processed_destinations(destinations, origins, cached_results)
            # Filter origins already processed for all destinations
            origins = self._filter_processed_origins(origins, destinations, cached_results)

        while not origins.empty:
            # Calculate the number of origins per batch based on remaining API calls
            n_max_api_per_batch = min(
                cfg.MAX_N_ITEMS_GOOGLE_MAPS_API_SINGLE_RUN,
                cfg.MONTHLY_MAX_FREE_GOOGLE_API_CALLS - self.n_api_call
            )
            if n_max_api_per_batch < len(destinations):
                self.logger.warning("Monthly API call limit reached. Stopping further processing.")
                break

            n_origins_per_batch = int(n_max_api_per_batch // cfg.N_MAX_DESTINATION)
            batch_origins = origins.iloc[:n_origins_per_batch]

            _destinations = destinations.copy()
            while not _destinations.empty:
                batch_destinations = _destinations.iloc[:cfg.N_MAX_DESTINATION]
                # Check pair in batch
                if (cached_results is None) or (len(cached_results) == 0):
                    self.run_google_distance_matrix(batch_origins, batch_destinations, api_key, output_file,
                                                    departure_time=departure_time, provincia_name=provincia_name)
                else:
                    check = (
                            cached_results["destination_key"].isin(batch_destinations.key)
                            & (cached_results["origin_key"].isin(batch_origins.key))
                            )
                    if check.sum() == 0:
                        self.run_google_distance_matrix(batch_origins, batch_destinations, api_key, output_file,
                                                   departure_time=departure_time, provincia_name=provincia_name)
                    else:
                        for d in batch_destinations:
                            for o in batch_origins:
                                check = (cached_results["destination_key"]==d.key) & (cached_results["origin_key"]==o.key)
                                if check.sum() == 0:
                                    self.run_google_distance_matrix([o], [d], api_key, output_file,
                                                               departure_time=departure_time, provincia_name=provincia_name)

                _destinations = _destinations.iloc[cfg.N_MAX_DESTINATION:]
            origins = origins.iloc[n_origins_per_batch:]

    def _filter_processed_destinations(self, destinations, origins, cached_results):
        """Remove destinations already processed for all origins."""
        drops = []
        for i, dest in destinations.iterrows():
            processed = cached_results[cached_results["destination_key"] == dest.key]
            all_origins_processed = origins["key"].isin(processed["origin_key"]).all()
            if all_origins_processed:
                drops.append(i)
        if drops:
            self.logger.info(f"Removed {len(drops)} destinations already processed.")
            destinations.drop(index=drops, inplace=True)
        return destinations

    def _filter_processed_origins(self, origins, destinations, cached_results):
        """Remove origins already processed for all destinations."""
        drops = []
        for i, origin in origins.iterrows():
            processed = cached_results[cached_results["origin_key"] == origin.key]
            all_destinations_processed = set(destinations["key"]) == set(processed["destination_key"])
            if all_destinations_processed:
                drops.append(i)
        if drops:
            self.logger.info(f"Removed {len(drops)} origins already processed.")
            origins.drop(index=drops, inplace=True)
        return origins

    def run_google_distance_matrix(
        self,
        origins: gpd.GeoDataFrame,
        destinations: gpd.GeoDataFrame,
        api_key: str,
        output_file: str,
        departure_time: datetime,
        provincia_name: str,
    ) -> None:
        """
        Queries the Google Distance Matrix API and saves results for given origins and destinations.
        """
        # Compute air-line distances in EPSG:32632
        air_line_distances = []
        for origin_geom in origins.geometry:
            distances = origin_geom.distance(destinations.geometry)
            air_line_distances.append(distances.values)

        gmaps = googlemaps.Client(key=api_key)
        # Transform to WGS84 for Google Maps
        origins = origins.to_crs(cfg.GOOGLE_MAPS_CRS)
        destinations = destinations.to_crs(cfg.GOOGLE_MAPS_CRS)

        origins = list(origins.geometry.apply(lambda geom: (geom.y, geom.x)))
        destinations = list(destinations.geometry.apply(lambda geom: (geom.y, geom.x)))

        self.n_api_call += (len(origins) * len(destinations))
        matrix = gmaps.distance_matrix(origins, destinations, mode="transit", departure_time=departure_time)
        # Loop through the origins and destinations directly
        for origin_idx, row in enumerate(matrix["rows"]):
            origin_coord = origins[origin_idx]
            for dest_idx, element in enumerate(row["elements"]):
                destination_coord = destinations[dest_idx]

                # Extract duration
                duration = element.get("duration", {}).get("value", None)
                air_line_distance = air_line_distances[origin_idx][dest_idx]

                if duration is None:
                    self.logger.warning(f"No directions found from point {origin_coord} to point {destination_coord}")

                # Save the result
                self.save_route_duration(output_file, origin_coord, destination_coord, provincia_name,
                                 duration, air_line_distance)

            # Pause to avoid exceeding rate limits
            time.sleep(1)

    def _get_provincia_series(
        self,
        provincia: Union[pd.Series, str],
    ) -> pd.Series:
        if isinstance(provincia, str):
            provincia_name = provincia
            provincia = gpd.read_parquet(cfg.province_shape_path)
            provincia = provincia[provincia[cfg.TAG_PROVINCIA] == provincia_name].iloc[0]
        return provincia

    def process_province(
        self,
        provincia: Union[pd.Series, str],
        output_resolution: float,
        evaluation_resolution: float,
        api_key: str,
        max_distance: float = cfg.MAX_DISTANCE_STOP_DEFAULT,
        evaluation_n_points: int = 50,
        output_file: str = cfg.public_transport_routes_path,
        departure_time: datetime = cfg.DATETIME_TRANSIT_GOOGLE_API,
    ):
        provincia = self._get_provincia_series(provincia)

        self.logger.info(f"Processing province {provincia[cfg.TAG_PROVINCIA]}")

        grid = self.generate_grid_with_resolutions(provincia['geometry'], resolutions=[output_resolution, evaluation_resolution])

        stop = self.get_public_stops()
        stop = stop[stop[cfg.TAG_PROVINCIA] == provincia[cfg.TAG_PROVINCIA]]
        grid = self.filter_grid_points(grid, stop, max_distance=max_distance)
        destinations = self.calculate_clusters(grid[grid["resolution"] >= evaluation_resolution], num_clusters=evaluation_n_points)
        destinations = self.aggregate_clusters(destinations)

        grid = grid[grid["resolution"] >= output_resolution]
        self.calculate_durations_distances(origins=grid, destinations = destinations, api_key = api_key,
                                           output_file= output_file, provincia_name = provincia[cfg.TAG_PROVINCIA],
                                           departure_time= departure_time)

    def load_routes(
        self,
        output_file: str = cfg.public_transport_routes_path,
        provincia_name: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        df = pd.read_csv(output_file)
        if provincia_name is not None:
            df = df[df["provincia_name"]==provincia_name]
        self.logger.info(f"Loaded {len(df)} routes.")
        df["origin_x"] = df["origin_x"].round(12)
        df["origin_y"] = df["origin_y"].round(12)
        df["destination_x"] = df["destination_x"].round(12)
        df["destination_y"] = df["destination_y"].round(12)
        return df

    def calculate_score(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        df = df[(df["origin_x"] != df["destination_x"]) | (df["origin_y"] != df["destination_y"])]
        df["check"] = df["duration"].notnull() & (df["duration"]>0) & (df["distance"]>0)
        df[cfg.SCORE_COLUMN] = np.where(
            df["check"],
            #1 / df["duration"],  # Media tempi di percorrenza
            df["distance"] / df["duration"],  # Media delle velocità
            1.2, # Media velocità a piedi (m/s)
        )
        df[cfg.SCORE_COLUMN] = df[cfg.SCORE_COLUMN] * df["size"]
        mean_distances = df["distance"].mean()
        df["distance"].fillna(mean_distances, inplace=True)
        df["distance"] = df["distance"] * df["size"]
        agg_func = {
            cfg.SCORE_COLUMN: (cfg.SCORE_COLUMN, "sum"),
            "distance": ("distance", "sum"),
            "size": ("size", "sum"),
            "count": ("check", "sum"),
        }
        df = df.groupby(["origin_x", "origin_y"]).agg(**agg_func).reset_index()
        df[cfg.SCORE_COLUMN] = df[cfg.SCORE_COLUMN] / df["size"]
        df["distance"] = df["distance"] / df["size"]
        #df = df[df["count"] >= 45]
        df[cfg.SCORE_COLUMN] = df[cfg.SCORE_COLUMN] / df["distance"]
        df[cfg.SCORE_COLUMN] = df[cfg.SCORE_COLUMN].clip(1.2 / mean_distances) #
        df = gpd.GeoDataFrame(df.drop(columns = ["origin_x", "origin_y"]),
                              geometry=gpd.points_from_xy(df["origin_x"], df["origin_y"]), crs=cfg.GOOGLE_MAPS_CRS)
        df.to_crs(cfg.SHAPE_CRS, inplace=True)
        df[cfg.SCORE_COLUMN] = (df[cfg.SCORE_COLUMN] - df[cfg.SCORE_COLUMN].min()) / (df[cfg.SCORE_COLUMN].max() - df[cfg.SCORE_COLUMN].min())
        return df

    def create_output_provincia(
        self,
        provincia: Union[pd.Series, str],
        output_resolution: float,
        evaluation_resolution: float,
        max_distance: float = cfg.MAX_DISTANCE_STOP_DEFAULT,
        evaluation_n_points: int = 50,
        routes_path: str = cfg.public_transport_routes_path,
        plot_path: Optional[str] = "default",
    ):
        provincia = self._get_provincia_series(provincia)

        self.logger.info(f"Processing provincia {provincia[cfg.TAG_PROVINCIA]}")

        grid = self.generate_grid_with_resolutions(provincia['geometry'], resolutions=[output_resolution, evaluation_resolution])

        stop = self.get_public_stops()
        stop = stop[stop[cfg.TAG_PROVINCIA] == provincia[cfg.TAG_PROVINCIA]]
        destinations = self.filter_grid_points(grid, stop, max_distance=max_distance)
        destinations = self.calculate_clusters(destinations[destinations["resolution"] >= evaluation_resolution], num_clusters=evaluation_n_points)
        destinations = self.aggregate_clusters(destinations)

        grid = grid[grid["resolution"] >= output_resolution]

        score_df = self.load_routes(output_file=routes_path)

        grid.to_crs(cfg.GOOGLE_MAPS_CRS, inplace=True)
        grid["origin_x"] = grid.geometry.x.round(12)
        grid["origin_y"] = grid.geometry.y.round(12)
        grid = pd.DataFrame(grid[["origin_x", "origin_y"]])
        destinations.to_crs(cfg.GOOGLE_MAPS_CRS, inplace=True)
        destinations["destination_x"] = destinations.geometry.x.round(12)
        destinations["destination_y"] = destinations.geometry.y.round(12)
        destinations = pd.DataFrame(destinations[["destination_x", "destination_y", "size"]])
        df = grid.assign(key=0).merge(destinations.assign(key=0), on = 'key').drop(columns = ['key'])
        df = df.merge(score_df, on=["origin_x", "origin_y", "destination_x", "destination_y"], how="left")
        del grid, destinations, score_df

        df = self.calculate_score(df)
        self.plot_provincia_result(provincia, df, path = plot_path)
        self.create_grid_points_file(df, provincia_name=provincia[cfg.TAG_PROVINCIA])
        return df

    def create_grid_points_file(
        self,
        df,
        provincia_name: str,
        path: Path = cfg.score_data_path,
    ):
        df.to_crs(cfg.GOOGLE_MAPS_CRS, inplace=True)
        df["longitude"] = df.geometry.x
        df["latitude"] = df.geometry.y
        df = df[["longitude", "latitude", cfg.SCORE_COLUMN]]
        df["provincia"] = provincia_name
        if path.is_file():
            df_old = pd.read_csv(path)
            df_old = df_old[df_old["provincia"]!=provincia_name]
            df = pd.concat([df, df_old])
        df.to_csv(path, index=False)

    def plot_grid_creation(
        self,
        provincia: Union[pd.Series, str],
        path: Optional[str] = "default",
        resolutions: list[float] = [2000, 1000, 500, 250],
    ):
        provincia = self._get_provincia_series(provincia)
        if path == "default":
            path = f"{provincia[cfg.TAG_PROVINCIA].lower()}_plot_grid_selection.pdf"
        colors = ["yellow", "orange", "blue", "green", "purple", "brown"]
        grid_points = self.generate_grid_with_resolutions(provincia["geometry"], resolutions)
        stops = self.get_public_stops()
        stops = stops[stops[cfg.TAG_PROVINCIA] == provincia[cfg.TAG_PROVINCIA]]
        grid_filtered = self.filter_grid_points(grid_points, stops)
        destinations = self.calculate_clusters(grid_filtered[grid_filtered["resolution"] == 1000], num_clusters=50)
        destinations = self.aggregate_clusters(destinations)

        fig, ax = plt.subplots(figsize=(15, 15))
        gpd.GeoDataFrame(provincia.to_frame().T, geometry="geometry").plot(ax=ax)
        for i, r in enumerate(resolutions[::-1]):
            df_plot = grid_points[grid_points["resolution"] == r]
            n_points = df_plot.shape[0]
            df_plot.plot(ax=ax, label=f"Resolution {r} ({n_points} points)", alpha=0.1, markersize=10,
                         c=colors[i % len(colors)])
        for i, r in enumerate(resolutions[::-1]):
            df_plot = grid_filtered[grid_filtered["resolution"] == r]
            n_points = df_plot.shape[0]
            df_plot.plot(ax=ax, label=f"Filtered Resolution {r} ({n_points} points)", alpha=0.5, markersize=10,
                         c=colors[i % len(colors)])
        stops.plot(ax=ax, label=f"Stops", marker="p", alpha=0.4, markersize=3, color="black")
        destinations.plot(ax=ax, label=f"Destinations", marker="*", alpha=1, markersize=20, color="red")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if path:
            plt.savefig(path, dpi=500, bbox_inches="tight")
        plt.show()

    def plot_provincia_result(
        self,
        provincia: Union[pd.Series, str],
        df: pd.DataFrame,
        path: Optional[str] = "default",
    ):
        provincia = self._get_provincia_series(provincia)
        if path == "default":
            path = f"{provincia[cfg.TAG_PROVINCIA].lower()}_plot_point_result.pdf"

        df_plot = df[df["score"] > df["score"].min()].copy()
        fig, ax = plt.subplots(figsize=(15, 15))
        gpd.GeoDataFrame(provincia.to_frame().T, geometry="geometry").boundary.plot(ax=ax, color="grey")
        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list(
    "transparent_to_blue_gradient",
    [(0, (0, 0, 1, 0)), (0.5, (0.5, 0.7, 1, 0.5)), (1, (0, 0, 1, 1))]
        )
        im = ax.scatter(x=df_plot.geometry.x, y=df_plot.geometry.y, c = df_plot.score, cmap="Blues")  #, cmap=cmap, edgecolors="none"

        ax.set_axis_off()
        plt.colorbar(im)
        if path:
            plt.savefig(path, dpi=500, bbox_inches="tight")
        plt.show()


if __name__== "__main__":
    # Download public stops
    #PublicTransportAnalysis().download_public_transport_stops()

    # Plot Grids selection
    #PublicTransportAnalysis().plot_grid_creation("Milano")

    # Calculate Distances
    #public_transport = PublicTransportAnalysis(initial_n_api_call=0)
    #public_transport.process_province(
    #    provincia = "Milano",
    #    output_resolution = 500,
    #    evaluation_resolution = 2000,
    #    api_key = GOOGLE_API,
    #)

    # PLot Results
    PublicTransportAnalysis().create_output_provincia(provincia = "Milano", output_resolution=500, evaluation_resolution = 2000)
