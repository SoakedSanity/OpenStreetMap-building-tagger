import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.geometry.polygon import orient
import seaborn as sns
import matplotlib.colors as mcolors
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

def compute_features(buildings_train_path, output_path):
    buildings_train = gpd.read_parquet(buildings_train_path)
    
    buildings_train = buildings_train.to_crs(32637)
    
    def has_few_coords(geom):
        if geom.is_empty:
            return True
        if geom.geom_type == 'Polygon':
            if len(geom.exterior.coords) <= 3:
                return False
            for interior in geom.interiors:
                if len(interior.coords) <= 3:
                    return False
            return True
        elif geom.geom_type == 'MultiPolygon':
            return all(has_few_coords(part) for part in geom.geoms)
        elif geom.geom_type in ['LineString', 'LinearRing']:
            coords = list(geom.coords)
            return len(coords) >= 4
        else:
            return False

    buildings_train = buildings_train[buildings_train.geometry.apply(has_few_coords)]
    buildings_train['geometry'] = buildings_train['geometry'].apply(lambda geom: Polygon(geom.coords) if geom.geom_type == 'LineString' else geom)
    buildings_train = buildings_train.explode(index_parts=False)
    
    buildings_train.geometry = buildings_train.geometry.apply(lambda geom: orient(geom, sign=1.0))

    buildings_train['level_count'] = pd.to_numeric(buildings_train['levels'], errors='coerce').fillna(1).astype(float).round().astype(int)
    
    buildings_train['oriented_mbr'] = buildings_train.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
    buildings_train['area'] = buildings_train.geometry.area
    
    buildings_train = buildings_train.set_geometry("oriented_mbr")
    buildings_train['area_MRR'] = buildings_train.geometry.area
    
    def get_rect_sides(geom):
        if geom.geom_type != 'Polygon':
            return None, None
        coords = list(geom.exterior.coords)
        side_a = ((coords[0][0] - coords[1][0])**2 + (coords[0][1] - coords[1][1])**2)**0.5
        side_b = ((coords[1][0] - coords[2][0])**2 + (coords[1][1] - coords[2][1])**2)**0.5
        return sorted([side_a, side_b], reverse=True)
    
    buildings_train[['length', 'width']] = buildings_train['oriented_mbr'].apply(get_rect_sides).tolist()
    buildings_train = buildings_train.set_geometry("geometry")
    
    buildings_train['perimeter'] = buildings_train.geometry.length
    
    buildings_train['compactness'] = buildings_train['area'] / buildings_train['perimeter']
    buildings_train['compactness_index'] = 4 * np.pi * buildings_train['area'] / buildings_train['perimeter'] ** 2
    
    buildings_train['rectangularity'] = buildings_train['area'] / buildings_train['area_MRR']
    
    buildings_train['elongatedness'] = buildings_train['length'] / buildings_train['width']
    
    buildings_train['convex_hull'] = buildings_train.geometry.apply(lambda geom: geom.convex_hull)
    buildings_train = buildings_train.set_geometry("convex_hull")
    
    buildings_train['convex_hull_area'] = buildings_train.geometry.area
    buildings_train['convex_hull_perimeter'] = buildings_train.geometry.length
    buildings_train['convexity'] = buildings_train['area'] / buildings_train['convex_hull_area']
    buildings_train['sinuosity'] = buildings_train['convex_hull_perimeter'] / buildings_train['perimeter']
    buildings_train = buildings_train.set_geometry("geometry")
    
    buildings_train['vertex_count'] = buildings_train.geometry.count_coordinates() - 1
    
    def count_convex_corners(geometry):
        if geometry.geom_type != 'Polygon':
            return 0
    
        coords = list(geometry.exterior.coords)[:-1]
        n = len(coords)
        convex_corners = 0
    
        for i in range(n):
    
            p0 = np.array(coords[i - 1])
            p1 = np.array(coords[i])
            p2 = np.array(coords[(i + 1) % n])
            v1 = p0 - p1
            v2 = p2 - p1
            cross_prod = np.cross(v1, v2)
    
            if cross_prod > 0:
                convex_corners += 1
    
        return convex_corners
    
    buildings_train['convex_vertex_count'] = buildings_train.geometry.apply(count_convex_corners)
    buildings_train['convex_vertex_ratio'] = buildings_train['convex_vertex_count']/buildings_train['vertex_count']
    
    buildings_train = buildings_train.set_geometry("geometry")
    buildings_train['inscribed_circle'] = buildings_train.geometry.maximum_inscribed_circle()
    buildings_train = buildings_train.set_geometry("inscribed_circle")
    buildings_train['inscribed_radius'] = buildings_train.geometry.length
    buildings_train['inscribed_area_ratio'] = (np.pi * buildings_train['inscribed_radius'] ** 2) / buildings_train['area']
    
    buildings_train = buildings_train.set_geometry("geometry")
    buildings_train['enclosed_radius'] = buildings_train.geometry.minimum_bounding_radius()
    buildings_train['enclosed_area_ratio'] = buildings_train['area'] / (np.pi * buildings_train['enclosed_radius'] ** 2)
    
    buildings_train['circular_variance'] = buildings_train['inscribed_radius'] / buildings_train['enclosed_radius']
    
    buildings_train['floor_area'] = buildings_train['level_count'] * buildings_train['area']
    buildings_train['height_ratio'] = 2.7 * buildings_train['level_count'] / buildings_train['area'] ** (1/2)
    
    buildings_train['hole_count'] = buildings_train.geometry.count_interior_rings()
    
    buildings_train['centroid'] = buildings_train.geometry.centroid
    buildings_train['representative_pt'] = buildings_train.geometry.representative_point()
    buildings_train['normalized_centroid_offset'] = buildings_train['centroid'].distance(buildings_train['representative_pt']) / buildings_train['area']

    buildings_train.to_parquet(output_path)

    return buildings_train

def display_correlation_matrix(buildings_train):
    buildings_train_clean = buildings_train.drop(["area_MRR", "length", "width", "compactness_index", "convex_hull_area", "convex_hull_perimeter", "convex_vertex_count", "circular_variance", "inscribed_area_ratio"], axis = 1)
    corr_matrix = buildings_train_clean.corr(numeric_only=True)
    
    colors = ["red", "yellow", "yellow", "green", "yellow", "yellow", "red"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("symmetric_cmap", colors)
    
    plt.figure(figsize=(12,12))
    sns.heatmap(corr_matrix, annot=True, cmap=custom_cmap, fmt='.2f', vmin=-1, vmax=1)
    plt.show()

if __name__ == "__main__":
    buildings_train = compute_features('russia_rnd_selection_2.parquet', 'russia_rnd_selection_features_2.parquet')