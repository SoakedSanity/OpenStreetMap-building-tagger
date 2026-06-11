import duckdb as ddb
import geopandas as gpd


def compute_contest_layyers(ddb_path: str, output_roads_path: str = 'roads.parquet',  output_retail_path: str = 'shop.parquet', output_buildings_path: str = 'builndings.parquet'):

    ddb.sql("""
        INSTALL osmium FROM community;
        INSTALL spatial;
        LOAD osmium;
        LOAD spatial;
        
        COPY (
            SELECT 
                id, 
                tags,
                geometry
            FROM '{ddb_path}'
            WHERE kind = 'line' 
                AND tags['highway'] = 'primary'
            ) TO '{output_roads_path}'
    """)

    ddb.sql("""
        INSTALL osmium FROM community;
        INSTALL spatial;
        LOAD osmium;
        LOAD spatial;
        
        COPY (
            SELECT 
                id, 
                tags,
                geometry
            FROM '{ddb_path}'
            WHERE tags['shop'] IS NOT NULL
        ) TO '{output_shop_path}'
    """)

    ddb.sql("""
        INSTALL osmium FROM community;
        INSTALL spatial;
        LOAD osmium;
        LOAD spatial;
        
        COPY (
            SELECT 
                id, 
                tags,
                geometry
            FROM '{ddb_path}'
            WHERE tags['buildings'] IS NOT NULL
        ) TO '{output_buildings_path}'
    """)

    
