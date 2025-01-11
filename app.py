import json

import numpy as np
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
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

# Function to load the language file
def load_language_file(language_code):
    file_name = f"data/texts/{language_code}.json"
    with open(file_name, "r", encoding='utf-8') as file:
        return json.load(file)

# Website title
st.set_page_config(
    page_title="Italian Public Transport Analysis",
    page_icon=None,
    layout="wide", initial_sidebar_state="collapsed", menu_items=None)

# CSS for custom positioning and fixed width of the radio selector
st.html(
    """
    <style>
    .language-selector {
        margin-left: auto; 
        margin-right: 0;    
        width: 200px;
    }
    </style>
    """
)

if "selected_language" not in st.session_state:
    st.session_state["selected_language"] = "eng"

texts = load_language_file(st.session_state["selected_language"])

title = st.title(texts["title"], anchor=False)

with stylable_container(
    key="lang_sel_cont",
    css_styles="""
    {
        margin-left: auto; 
        margin-right: 0;    
        width: 200px;
    }
    """
):
    selected_language = st.radio(
        "",
        ["eng", "ita"],
        format_func=lambda x: "English" if x == "eng" else "Italiano",
        horizontal=True,
        key="language-selector",
        label_visibility="collapsed",
    )
if selected_language != st.session_state["selected_language"]:
    st.session_state["selected_language"] = selected_language
    texts = load_language_file(selected_language)
    st.rerun()

map_tab, info_tab, download_tab = st.tabs(
    [f":world_map: {texts['map_tab']}", f":mag: {texts['method_tab']}", f":open_file_folder: {texts['download_tab']}"])
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
    ), use_container_width=True)  #, height=800
    col2.caption(texts["map_caption"])

else:
    col2.warning("No data available. Please ensure the backend file is properly set up.")

col1.header("", anchor=False)
col1.markdown(texts["short_description_1"])
col1.markdown(texts["short_description_2"])
col1.markdown(texts["short_description_3"])
col1.markdown(texts["short_description_4"])
col1.markdown(texts["short_description_5"])

info_tab.header(texts["method_tab"], anchor=False)
info_tab.markdown(texts["method_intro"])
info_tab.markdown(f"""
1. [{texts["cap1_header"]}](#objective)
2. [{texts["cap2_header"]}](#solution)
3. [{texts["cap3_header"]}](#next_steps)
4. [{texts["cap4_header"]}](#code)
""")
info_tab.divider()
info_tab.header(texts["cap1_header"], anchor="objective")
info_tab.markdown(texts["cap1_text_1"])
info_tab.markdown(texts["cap1_text_2"])

info_tab.divider()
info_tab.header(texts["cap2_header"], anchor="solution")
image_info1, text_info1 = info_tab.columns([1,3], vertical_alignment="center")
text_info1.subheader(texts["cap2_1_header"], anchor=False)
text_info1.markdown(texts["cap2_1_text"])
image_info1.image("data/images/idea.png")

text_info2, image_info2 = info_tab.columns([3,1], vertical_alignment="center")
text_info2.subheader(texts["cap2_2_header"], anchor=False)
text_info2.markdown(texts["cap2_2_text"])
image_info2.image("data/images/grid.png")

image_info3, text_info3 = info_tab.columns([1,3], vertical_alignment="center")
text_info3.subheader(texts["cap2_3_header"], anchor=False)
text_info3.markdown(texts["cap2_3_text_1"])
text_info3.markdown(texts["cap2_3_text_2"])
text_info3.markdown(texts["cap2_3_text_3"])
image_info3.image("data/images/two_grid.png")

text_info4, image_info4  = info_tab.columns([3,1], vertical_alignment="center")
text_info4.subheader(texts["cap2_4_header"], anchor=False)
text_info4.markdown(texts["cap2_4_text_1"])
text_info4.markdown(texts["cap2_4_text_2"])
image_info4.image("data/images/filter_points.png")


image_info5, text_info5 = info_tab.columns([1,3], vertical_alignment="center")
text_info5.subheader(texts["cap2_5_header"], anchor=False)
text_info5.markdown(texts["cap2_5_text"])
image_info5.image("data/images/maps.PNG")

text_info6, image_info6 = info_tab.columns([3,1], vertical_alignment="center")
text_info6.subheader(texts["cap2_6_header"], anchor=False)
text_info6.markdown(texts["cap2_6_text_1"])
text_info6.markdown(texts["cap2_6_text_2"])
text_info6.markdown(texts["cap2_6_text_3"])
image_info6.image("data/images/score.png")

info_tab.divider()
info_tab.header(texts["cap3_header"], anchor="next_steps")
info_tab.markdown(texts["cap3_text_1"])
info_tab.markdown(texts["cap3_text_2"])
info_tab.markdown(texts["cap3_text_3"])
info_tab.markdown(texts["cap3_text_4"])
info_tab.markdown(texts["cap3_text_5"])

info_tab.divider()
info_tab.header(texts["cap4_header"], anchor="code")
info_tab.markdown(texts["cap4_text_1"])
info_tab.markdown(texts["cap4_text_2"])
info_tab.divider()

download_tab.header(texts["download_tab"], anchor=False)
download_tab.markdown(texts["download_text_1"])
download_tab.markdown(texts["download_text_2"])

download_tab.divider()
download_tab.header(texts["download_header_1"], anchor=False)
download_tab.markdown(texts["download_text_3"])
download_tab.markdown(texts["download_text_4"])
download_tab.divider()

st.html("<style> header.stAppHeader {display: none;}</style>")
st.html(""" <style>.block-container {
    padding-top: 0rem;
    padding-bottom: 1rem;
    padding-left: 1rem;
    padding-right: 1rem;
}</style>""")