import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import duckdb as ddb
from deep_translator import GoogleTranslator
import os
import plotly.express as px
from IPython.display import display, HTML
import importlib
import pydeck as pdk
import random

import load_footprints
import type_proportions
importlib.reload(load_footprints)
importlib.reload(type_proportions)
from load_footprints import random_selection, point_selection
from type_proportions import building_amount, building_amount_zipf

layer_list = []

tags = building_amount_zipf(threshold = 1000, correction = 5)

for i in range(len(tags)):

    tag = tags[i][0]
    amount = tags[i][1]
    
    random_selection('buildings_russia.osm.pbf', 'russia_rnd_selection_2.parquet', 'building', tag, count = amount, is_init = i == 0)
    # point_selection('buildings_russia.osm.pbf', 'output.parquet', 37.4, 55.75, 'building', tag, count = amount, is_init = i == 0)
    
    buildings_with_type = gpd.read_parquet('russia_rnd_selection_2.parquet')
    buildings_with_type = buildings_with_type[buildings_with_type["type"] == tag]
    
    geojson_layer = pdk.Layer(
        "GeoJsonLayer",
        buildings_with_type,
        opacity=0.8,
        stroked=True,
        filled=True,
        extruded=True,
        wireframe=True,
        get_elevation="levels * 2",
        get_fill_color=f"[{random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)}]",
        get_line_color="[255, 255, 255]",
        pickable=True
    )

    layer_list.append(geojson_layer)
    
view_state = pdk.ViewState(
    latitude=55.75,
    longitude=37.4,
    zoom=10,
    pitch=45
)
    
r = pdk.Deck(layers=layer_list, initial_view_state=view_state)
r.to_html("3d_map_2.html")

    