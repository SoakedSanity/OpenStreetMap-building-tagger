import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import duckdb as ddb


def random_selection(pbf_path: str, output_path: str, target_key: str = 'building', target_value: str = 'yes', count: int = 100, is_init: bool = True):
    """Returns a GeoParquet file with randomly selected objects with specified OSM key-tag combination.

    Args:
        pbf_path (str): Path to PBF file with input OSM data.
        output_path (str): Path to GeoParquet file for the output.
        target_key (str): OSM target key name ('bulding' by default).
        target_value (str): OSM target tag name. 
        count (str): Amount of objects to query (100 by default).
        is_init(bool): Signifies if this is a first instance of the function.

    """
    if is_init:
        union_instruction = ""
    else:
        union_instruction = f"SELECT * FROM '{output_path}' UNION ALL"

    ddb.sql(f"""
    
    INSTALL osmium FROM community;
    INSTALL spatial;
    LOAD osmium;
    LOAD spatial;
    
    COPY (
        {union_instruction}
        SELECT * FROM (
            SELECT 
                id, 
                tags, 
                tags['building'] as type,
                tags['building:levels'] as levels,
                geometry,
            FROM '{pbf_path}'
            WHERE tags['{target_key}'] = '{target_value}'
            ORDER BY random() LIMIT {count})
    ) TO '{output_path}';
    """)

def point_selection(pbf_path: str, output_path: str, point_x: float, point_y: float, target_key: str = 'building', target_value: str = 'yes', count: int = 100, is_init: bool = True):

    """Returns a GeoParquet file with objects with specified OSM key-tag combination closest to a specified point.

    Args:
        pbf_path (str): Path to PBF file with input OSM data.
        output_path (str): Path to GeoParquet file for the output.
        point_x (str): Latitude of central point.
        point_y (str): Longitude of central point.
        target_key (str): OSM target key name ('bulding' by default).
        target_value (str): OSM target tag name.
        count (str): Amount of objects to query (100 by default).
        s_init(bool): Signifies if this is a first instance of the function.

    """
    
    if is_init:
        union_instruction = ""
    else:
        union_instruction = f"SELECT * FROM '{output_path}' UNION ALL"
        
    ddb.sql(f"""
    
    INSTALL osmium FROM community;
    INSTALL spatial;
    LOAD osmium;
    LOAD spatial;
    
    COPY (
        {union_instruction}
        SELECT 
            id, 
            tags,
            tags['building'] as type,
            tags['building:levels'] AS levels,
            geometry
        FROM (
        SELECT 
            id, 
            tags,
            tags['building'] as type,
            tags['building:levels'] AS levels,
            geometry,
            ST_Distance(geometry, ST_Point({point_x}, {point_y})) AS distance
        FROM '{pbf_path}'
        WHERE tags['{target_key}'] = '{target_value}'
        ORDER BY distance ASC
        LIMIT {count})
    ) TO '{output_path}';
    """)