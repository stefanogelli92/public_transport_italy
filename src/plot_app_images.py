from pathlib import Path
from typing import Union, Optional, List
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

import src.config as cfg
from src.analysis import PublicTransportAnalysis


def plot_idea_image(
    provincia: Union[pd.Series, str] = "Milano",
    center_point: tuple = (514860.8615675086, 5034571.840081673),
    path: str = cfg.root_path / Path("data/images/idea.png"),
):
    analysis = PublicTransportAnalysis()
    provincia = analysis._get_provincia_series(provincia)

    grid_points = analysis.generate_grid_with_resolutions(provincia["geometry"], 1000)
    destinations = analysis.calculate_clusters(grid_points, num_clusters=50)
    destinations = analysis.aggregate_clusters(destinations)
    x_value = min(grid_points.geometry.x, key=lambda x: abs(x - center_point[0]))
    y_value = min(grid_points.geometry.y, key=lambda x: abs(x - center_point[1]))
    grid_points = grid_points[(grid_points.geometry.x == x_value) & (grid_points.geometry.y == y_value)]
    grid_points["origin_x"] = grid_points.geometry.x
    grid_points["origin_y"] = grid_points.geometry.y
    grid_points = pd.DataFrame(grid_points[["origin_x", "origin_y"]])
    destinations["destination_x"] = destinations.geometry.x
    destinations["destination_y"] = destinations.geometry.y
    destinations = pd.DataFrame(destinations[["destination_x", "destination_y"]])
    df = grid_points.assign(key=0).merge(destinations.assign(key=0), on='key').drop(columns=['key'])

    fig, ax = plt.subplots(figsize=(15, 15))
    gpd.GeoDataFrame(provincia.to_frame().T, geometry="geometry").boundary.plot(ax=ax, color="grey")
    for _, row in df.iterrows():
        ax.plot([row["origin_x"], row["destination_x"]],
                [row["origin_y"], row["destination_y"]], 'ro-')
    ax.set_axis_off()
    if path:
        plt.savefig(path, dpi=500, bbox_inches="tight")

def plot_grid_image(
    provincia: Union[pd.Series, str] = "Milano",
    resolutions: Union[list, int] = [2000, 1000, 500],
    path: str = cfg.root_path / Path("data/images/grid.png"),
):
    analysis = PublicTransportAnalysis()
    provincia = analysis._get_provincia_series(provincia)

    grid_points = analysis.generate_grid_with_resolutions(provincia["geometry"], resolutions)
    grid_points.geometry = grid_points.geometry.buffer(
        distance=int(min(resolutions)/3),
        cap_style=3,
    )
    resolutions = sorted(resolutions)

    fig, ax = plt.subplots(figsize=(15, 15))
    gpd.GeoDataFrame(provincia.to_frame().T, geometry="geometry").boundary.plot(ax=ax, color="grey")
    colors = ["Orange", "Red", "Blue"]
    for i, resolution in enumerate(resolutions):
        df_plot = grid_points[grid_points["resolution"]==resolution]

        df_plot.plot(ax=ax, color=colors[i], label=f"{resolution}m")
    ax.set_axis_off()
    from matplotlib.lines import Line2D
    lines = [
        Line2D([0], [0], linestyle="none", marker="s", markersize=10, markerfacecolor=t.get_facecolor())
        for t in ax.collections[1:]
    ]
    labels = [t.get_label() for t in ax.collections[1:]]
    ax.legend(lines, labels, title="Resolutions", prop={'size': 20}, fontsize=20, title_fontsize=20, loc="lower left")
    if path:
        plt.savefig(path, dpi=500, bbox_inches="tight")

def plot_two_grid_combination_image(
    n_points: int = 5,
    path: str = cfg.root_path / Path("data/images/two_grid.png"),
):
    point_color = "blue"
    arrow_color = "black"
    grid_x, grid_y = np.meshgrid(range(n_points), range(n_points))
    orange_grid_x, orange_grid_y = np.meshgrid(range(0, n_points, 4), range(0, n_points, 4))

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].set_title(f"One-Grid ({n_points**2}x{n_points**2-1} = {(n_points**2) * (n_points**2-1)} routes)", fontsize=20)
    ax[1].set_title(f"Two-Grid ({n_points**2-1}x{math.ceil(n_points/4)**2} = {(n_points**2-1) * (math.ceil(n_points/4)**2)} routes)", fontsize=20)
    ax[0].scatter(grid_x, grid_y, color=point_color, s=500, marker='o', label='500m')
    ax[1].scatter(grid_x, grid_y, color=point_color, s=500, marker='o', label='500m')
    ax[1].scatter(orange_grid_x, orange_grid_y, color='orange', s=500, marker='o', label='2000m')
    for i in range(n_points):
        for j in range(n_points):
            for k in range(n_points):
                for l in range(n_points):
                    if (i, j) != (k, l):
                        ax[0].annotate('', xy=(grid_x[i, j], grid_y[i, j]),
                                         xytext=(grid_x[k, l], grid_y[k, l]), fontsize=7,
                                         arrowprops=dict(edgecolor=arrow_color, arrowstyle='<-'))
                        if (k%4 == 0) and (l%4 == 0):
                            ax[1].annotate('', xy=(grid_x[i, j], grid_y[i, j]),
                                         xytext=(grid_x[k, l], grid_y[k, l]), fontsize=7,
                                         arrowprops=dict(edgecolor=arrow_color, arrowstyle='<-'))
    ax[0].axis('off')
    ax[1].axis('off')
    ax[1].legend(title="Resolution", bbox_to_anchor=(1, 0.5), loc="center left", fontsize=20)
    if path:
        plt.savefig(path, dpi=500, bbox_inches="tight")

def plot_filter_points_image(
    provincia: Union[pd.Series, str] = "Milano",
    path: str = cfg.root_path / Path("data/images/filter_points.png"),
):
    analysis = PublicTransportAnalysis()
    provincia = analysis._get_provincia_series(provincia)

    grid_points = analysis.generate_grid_with_resolutions(provincia["geometry"], [2000, 500])
    stops = analysis.get_public_stops()
    stops = stops[stops[cfg.TAG_PROVINCIA] == provincia[cfg.TAG_PROVINCIA]]
    grid_filtered = analysis.filter_grid_points(grid_points, stops)
    destinations = analysis.calculate_clusters(grid_filtered[grid_filtered["resolution"] == 2000], num_clusters=50)
    destinations = analysis.aggregate_clusters(destinations)

    fig, ax = plt.subplots(figsize=(15, 15))
    gpd.GeoDataFrame(provincia.to_frame().T, geometry="geometry").boundary.plot(ax=ax)
    df_plot = grid_points[grid_points["resolution"] >= 500]
    df_plot.plot(ax=ax, label=f"Ignored", alpha=0.3, markersize=20,
                     c="red")
    df_plot = grid_filtered[grid_filtered["resolution"] >= 500]
    df_plot.plot(ax=ax, label=f"Selected", alpha=1, markersize=20,
                 c="green")
    stops.plot(ax=ax, label=f"Stops", marker="p", alpha=1, markersize=6, color="black")
    destinations.plot(ax=ax, label=f"Destinations", marker="*", alpha=1, markersize=100, color="purple")
    ax.axis('off')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    if path:
        plt.savefig(path, dpi=500, bbox_inches="tight")

def plot_score_image(
    provincia: Union[pd.Series, str] = "Milano",
    center_point: tuple = (514860.8615675086, 5034571.840081673),
    path: str = cfg.root_path / Path("data/images/score.png"),
):
    analysis = PublicTransportAnalysis()
    provincia = analysis._get_provincia_series(provincia)

    grid_points = analysis.generate_grid_with_resolutions(provincia['geometry'],
                                               resolutions=[500, 2000])

    stop = analysis.get_public_stops()
    stop = stop[stop[cfg.TAG_PROVINCIA] == provincia[cfg.TAG_PROVINCIA]]
    grid_points = analysis.filter_grid_points(grid_points, stop, max_distance=500)
    destinations = analysis.calculate_clusters(grid_points[grid_points["resolution"] == 2000],
                                           num_clusters=50)
    destinations = analysis.aggregate_clusters(destinations)

    x_value = min(grid_points.geometry.x, key=lambda x: abs(x - center_point[0]))
    y_value = min(grid_points.geometry.y, key=lambda x: abs(x - center_point[1]))
    grid_points = grid_points[(grid_points.geometry.x == x_value) & (grid_points.geometry.y == y_value)]
    grid_points["origin_x1"] = grid_points.geometry.x
    grid_points["origin_y1"] = grid_points.geometry.y
    grid_points.to_crs(cfg.GOOGLE_MAPS_CRS, inplace=True)
    grid_points["origin_x"] = grid_points.geometry.x.round(12)
    grid_points["origin_y"] = grid_points.geometry.y.round(12)
    grid_points = pd.DataFrame(grid_points[["origin_x", "origin_y", "origin_x1", "origin_y1"]])
    destinations["destination_x1"] = destinations.geometry.x
    destinations["destination_y1"] = destinations.geometry.y
    destinations.to_crs(cfg.GOOGLE_MAPS_CRS, inplace=True)
    destinations["destination_x"] = destinations.geometry.x.round(12)
    destinations["destination_y"] = destinations.geometry.y.round(12)
    destinations = pd.DataFrame(destinations[["destination_x", "destination_y", "destination_x1", "destination_y1"]])
    df = grid_points.assign(key=0).merge(destinations.assign(key=0), on='key').drop(columns=['key'])
    score_df = analysis.load_routes()
    df = df.merge(score_df, on=["origin_x", "origin_y", "destination_x", "destination_y"], how="left")

    fig, ax = plt.subplots(figsize=(15, 15))
    gpd.GeoDataFrame(provincia.to_frame().T, geometry="geometry").boundary.plot(ax=ax, color="grey")
    for _, row in df.iterrows():
        ax.plot([row["origin_x1"], row["destination_x1"]],
                [row["origin_y1"], row["destination_y1"]], 'ro-')
        ax.text(x=(row["destination_x1"]),  #row["origin_x1"] + /2
                y=(row["destination_y1"]),  #row["origin_y1"] + /2
                s=f"{time.strftime('%H:%M:%S', time.gmtime(row['duration']))}",
                va="center", ha="center", backgroundcolor="white",
                )
    ax.set_axis_off()
    if path:
        plt.savefig(path, dpi=500, bbox_inches="tight")

if __name__ == "__main__":
    plot_idea_image()
    plot_grid_image()
    plot_two_grid_combination_image()
    plot_filter_points_image()
    plot_score_image()