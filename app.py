import numpy as np
import streamlit as st
import pandas as pd
import pydeck as pdk
import os

import src.config as cfg

# Function to load grid data from backend
DATA_FILE_PATH = cfg.score_data_path

@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        return data
    else:
        st.error(f"Data file not found. Please ensure the backend data file exists. {file_path}")
        return None

# Website title
st.set_page_config(
    page_title="Public Transport Accessibility Analysis",
    page_icon=None,
    layout="wide", initial_sidebar_state="auto", menu_items=None)
# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)
margins_css = """
    <style>
        .main > div {
            padding-left: 0rem;
            padding-right: 0rem;
        }
    </style>
"""
st.markdown(margins_css, unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .stButton button { display: none; }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Italian Public Transport Accessibility Analysis", anchor=False)
map_tab, info_tab, download_tab = st.tabs([":world_map: Map", ":mag: Methodology", ":open_file_folder: Download Data"])
col1, col2 = map_tab.columns([1,3])

# Load data from backend
data = load_data(DATA_FILE_PATH)

if data is not None:

    # Normalize scores for visualization (min-max normalization)
    data['score'] = np.exp(data['score'])
    data['normalized_score'] =  (data['score'] - data['score'].min()) / (data['score'].max() - data['score'].min())

    # Remove minimum score rows to avoid rendering empty cells
    data = data[data['normalized_score'] > 0]

    # Adjust the polygon size to ensure proper coverage of the grid
    polygon_size_x = 0.0064 / 2  # Adjusted value for ~500 meters
    polygon_size_y = 0.0045 / 2  # Adjusted value for ~500 meters

    # Calculate bounding box to center the map and set map limits
    min_lat, max_lat = data['latitude'].min(), data['latitude'].max()
    min_lon, max_lon = data['longitude'].min(), data['longitude'].max()
    center_lat, center_lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2

    # Set zoom level to 10
    zoom = 9

    # Prepare pydeck layers
    layer = pdk.Layer(
        "PolygonLayer",
        data,
        get_polygon=f"[[[longitude - {polygon_size_x}, latitude - {polygon_size_y}], "
                    f"[longitude - {polygon_size_x}, latitude + {polygon_size_y}], "
                    f"[longitude + {polygon_size_x}, latitude + {polygon_size_y}], "
                    f"[longitude + {polygon_size_x}, latitude - {polygon_size_y}], "
                    f"[longitude - {polygon_size_x}, latitude - {polygon_size_y}]]]",
        get_fill_color="[0, 0, 255, normalized_score * 255]",  # Transparency based on normalized score
        pickable=False,
        auto_highlight=False,
        stroked=False,  # Disable borders for polygons
    )

    # Set the initial view state and map constraints
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
    )

    # Render deck.gl map with restored page settings
    col2.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state
    ), use_container_width=True, height=800)

else:
    col2.warning("No data available. Please ensure the backend file is properly set up.")

# Contact Section
#col1.header("Explore Public Transport Accessibility Across Italy", anchor=False)
col1.header("", anchor=False)
col1.markdown("""
This interactive map lets you explore public transport accessibility in Italy at a highly detailed level. 

Areas with more vibrant colors indicate locations that are better served, while those with lighter shades represent areas with less service. 

This analysis allows for a variety of applications, helping to compare and evaluate accessibility across different points. 

The scores are calculated based on public transport travel times between a given point and other areas within the same province.


:warning: This project is still ongoing, and currently, the accessibility data is available only for the provincia of Milano.
""")

info_tab.header("Methodology", anchor=False)
info_tab.markdown("In this section, we guide you through the methodology behind the analysis, explaining the key steps and considerations that were taken to evaluate public transport accessibility. The goal of this section is to provide a detailed understanding of how the analysis was carried out, the solution developed, the challenges faced, and the open questions that remain for future work.")
info_tab.markdown("""
1. [Objective](#objective)
2. [Solution](#solution)
3. [Next Steps](#next_steps)
4. [Code](#code)
""")
info_tab.divider()
info_tab.header("Objective", anchor="objective")
info_tab.markdown("The main objective of this analysis is to precisely identify areas well-served by public transport across Italy. The goal is to create a quantitative score that is as objective as possible, allowing for a fair and accurate comparison between any two zones in the country. By focusing on a detailed and data-driven approach, this methodology aims to provide a clear and standardized measure of public transport accessibility, helping to highlight areas of strength and areas in need of improvement.")

info_tab.divider()
info_tab.header("Solution", anchor="solution")
image_info1, text_info1 = info_tab.columns([1,3], vertical_alignment="center")
text_info1.subheader("The idea", anchor=False)
text_info1.markdown("The idea behind this analysis is to calculate the public transport travel times from the point where the score is being calculated to 'every' other part of the province. By averaging these values, weighted by the distance traveled, we obtain an accessibility score for the starting point. This approach ensures that the score reflects the level of connectivity of each point within its surrounding area, providing a more nuanced and precise measurement of public transport accessibility.")
image_info1.image("data/images/idea.png")

text_info2, image_info2 = info_tab.columns([3,1], vertical_alignment="center")
text_info2.subheader("The Grid", anchor=False)
text_info2.markdown("How are the points for calculating the distances chosen? The answer is to create a grid over the province. The resolution of this grid must be high enough to ensure an adequate level of detail for the analysis, but not so fine as to make the calculations too computationally expensive. The resolution currently used is 500m x 500m, striking a balance between detail and computational efficiency.")
image_info2.image("data/images/grid.png")

image_info3, text_info3 = info_tab.columns([1,3], vertical_alignment="center")
text_info3.subheader("Two-Grid Method", anchor=False)
text_info3.markdown("To calculate the accessibility score with a high resolution, a two-stage grid system is used. First, a fine grid is created over the province, with high-resolution points where the accessibility score for each location will be calculated. This fine grid provides the detailed output of the analysis, capturing the granularity of the public transport service.")
text_info3.markdown("Next, a second, less dense grid is overlaid on the province. The points in this coarser grid serve as the locations to which each point of the fine grid will be compared to calculate travel times using public transport. This second grid is less detailed because its primary function is to identify the key destinations for travel time calculation, not to directly impact the resolution of the final accessibility score.")
text_info3.markdown("The use of a coarse grid reduces the computational cost of the analysis without compromising the level of detail in the final output, as the fine grid's resolution determines the granularity of the accessibility scores.")
image_info3.image("data/images/two_grid.png")

text_info4, image_info4  = info_tab.columns([3,1], vertical_alignment="center")
text_info4.subheader("Route Reduction and Clustering", anchor=False)
text_info4.markdown("To optimize the calculation of accessibility scores and reduce computational costs, several methods are used to streamline the number of routes involved in the analysis. First, points are filtered to include only those within 500 meters of a public transport stop. This ensures that only areas with direct access to public transport are considered, excluding points that are too far from any transit services.")
text_info4.markdown("Next, clustering is applied to the less dense grid described earlier. The points are grouped into clusters, with a fixed number of 50 points per cluster in each province. Clustering reduces the number of routes considered for calculation while still maintaining an adequate level of detail for the analysis.")
image_info4.image("data/images/filter_points.png")


image_info5, text_info5 = info_tab.columns([1,3], vertical_alignment="center")
text_info5.subheader("Travel Time Calculation Using Public Transport", anchor=False)
text_info5.markdown("The next step in the analysis was to calculate travel times using public transport. For this, the Google Maps API was chosen as the data source, providing access to the most up-to-date and accurate data available. Using the API, the travel time for each pair of points in the grid was calculated, which, together with the air distance, will be used to weigh the travel time for the score calculation.")
image_info5.image("data/images/maps.PNG")

text_info6, image_info6 = info_tab.columns([3,1], vertical_alignment="center")
text_info6.subheader("The Score", anchor=False)
text_info6.markdown("This step is crucial as it involves how to aggregate the scores to properly weigh the different travel times calculated for each starting point. There are several ways to perform this aggregation, and the challenge lies in the absence of a clear reference to determine the most appropriate method for calculating a public transport accessibility score.")
text_info6.markdown("Currently, the calculation involves determining the average speed of each trip, which is computed as the ratio between the air distance (to avoid favoring fast routes with significant deviations) and the travel time. This score is then averaged and divided by the mean distance to the destination points within the same province.")
text_info6.markdown("While further analyses are underway to refine this process, this method provides a useful starting point for measuring accessibility to public transport in a meaningful way.")
image_info6.image("data/images/score.png")

info_tab.divider()
info_tab.header("Next Steps", anchor="next_steps")
info_tab.markdown("While the current analysis provides valuable insights into public transport accessibility, several key tasks remain to further improve and refine the results.")
info_tab.markdown("""
1. Complete the analysis for other provinces:
The analysis is currently available for the province of Milan, but expanding the coverage to additional provinces is essential to create a comprehensive map of public transport accessibility across Italy.

2. Optimize route filtering:
An important next step is finding an additional filter to reduce the number of routes used to calculate travel times. This will help decrease the computational cost associated with using the Google Maps API, ensuring the analysis remains scalable and efficient.

3. Develop a validation method:
Another crucial task is to establish a method for validating the results. This will help ensure the accuracy of the accessibility scores and enable the identification of the most effective method for calculating the scores."""
                  )
info_tab.markdown("These next steps are vital for enhancing the quality of the analysis, improving scalability, and ensuring that the chosen methods are both effective and reliable.")

info_tab.divider()
info_tab.header("Code", anchor="code")
info_tab.markdown("This analysis was developed using Python, and the full code is available in the GitHub repository ([link](https://github.com/stefanogelli92/public_transport_italy)).")
info_tab.markdown("Please note that in order to run the code, a valid Google Maps API key is required to access the public transport data.")
info_tab.divider()

download_tab.header("Download Data", anchor=False)
download_tab.markdown("You can download the results of the analysis in CSV format by clicking the link below.")
download_tab.markdown(" - [public_transport_score_grid.csv](https://drive.google.com/file/d/1skm8LKcafcWy2aT0C5YQwntc7uAOVcQ5/view?usp=drive_link) (File size: 310 KB)")

download_tab.divider()
download_tab.header("Dataset Structure", anchor=False)
download_tab.markdown("""
| Column   |      Description      |
|----------|-------------|
| Latitude |  The latitude of the point in the grid (in decimal degrees). |
| Longitude |    The longitude of the point in the grid (in decimal degrees).   |
| Score | The accessibility score for that point, calculated based on the public transport analysis. |
""")
download_tab.divider()
download_tab.markdown("The coordinates in the dataset are in the Google Maps reference system (EPSG:4326), which uses latitude and longitude in decimal degrees.")
download_tab.divider()
