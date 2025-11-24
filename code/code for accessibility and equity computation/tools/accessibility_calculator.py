# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 21:44:09 2025

@author: user
"""

import numpy as np
from scipy.spatial.distance import cdist
from pyproj import Transformer
import gc

def calculate_gini(population, accessibility):
    """Calculate the Gini coefficient."""
    if len(accessibility) == 0 or len(population) == 0:
        return 0
    
    # Filter out entries with zero population
    valid_mask = population > 0
    population = population[valid_mask]
    accessibility = accessibility[valid_mask]

    if len(accessibility) == 0 or len(population) == 0:
        return 0

    # If accessibility is concentrated in only one grid and all other grids have zero accessibility
    if np.count_nonzero(accessibility) == 1:
        return 1.0

    # Sort accessibility in ascending order
    sorted_indices = np.argsort(accessibility)
    sorted_population = population[sorted_indices]
    sorted_accessibility = accessibility[sorted_indices]

    # Compute cumulative population and cumulative accessibility
    cum_population = np.cumsum(sorted_population)
    cum_accessibility = np.cumsum(sorted_accessibility)

    # Normalize cumulative population and cumulative accessibility
    cum_population = cum_population / cum_population[-1]
    cum_accessibility = cum_accessibility / cum_accessibility[-1]

    # Calculate the Gini coefficient
    gini = 1 - np.sum((cum_accessibility[1:] + cum_accessibility[:-1]) * \
                  (cum_population[1:] - cum_population[:-1]))

    # Ensure the Gini coefficient is within [0, 1]
    return max(0, min(gini, 1))


class AccessibilityCalculator:
    def __init__(self, grids_gdf, station_gdf, d0, supply_col='evse_count'):
        
        """
        :param grids_gdf: Demand-side grid GeoDataFrame, must include the 'population' field
        :param station_gdf: Actual charging station GeoDataFrame, must include 'geometry' and 'supply' fields
        :param d0: Distance threshold (in kilometers)
        :param supply_col: Name of the charging station supply column
        """

        self.grids_gdf = grids_gdf
        self.station_gdf = station_gdf
        self.d0 = d0
        self.supply_col = supply_col
        self.accessibility_col = 'Accessibility'
        self.pop_accessibility_col = 'Populated Accessibility'
        self.grids_gdf[self.accessibility_col] = 0.0
        self.grids_gdf[self.pop_accessibility_col] = 0.0

    @staticmethod
    def gaussian_decay(distance, d0):
        norm_factor = (1 - np.exp(-0.5)).astype(np.float32)
        decay_values = np.where(distance <= d0,
                                ((np.exp(-0.5 * (distance / d0) ** 2) - np.exp(-0.5)) / norm_factor).astype(np.float32),
                                0).astype(np.float32)
        return decay_values.astype(np.float32)

    @staticmethod
    def project_coordinates(lats, lons, target_crs='EPSG:3857'):
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        xs, ys = transformer.transform(lons, lats)
        return np.vstack((xs, ys)).T.astype(np.float32)

    def calculate_accessibility_m2sfca(self):
        if self.grids_gdf.empty:
            print("Zero data in the grid")
            return None
        if self.station_gdf.empty:
            print("No charging station")
            self.grids_gdf[self.accessibility_col] = 0.0
            self.grids_gdf[self.pop_accessibility_col] = 0.0
            return True
    
        station_coords = np.array([(geom.y, geom.x) for geom in self.station_gdf.geometry], dtype=np.float32)
        supplies = self.station_gdf[self.supply_col].values.astype(np.float32)  # shape: (M,)
        grid_centroids = [(geom.centroid.y, geom.centroid.x) for geom in self.grids_gdf.geometry]
        population = self.grids_gdf['population'].values.astype(np.float32)    # shape: (N,)

        station_proj = self.project_coordinates(*zip(*station_coords))  # shape: (M, 2)
        grid_proj = self.project_coordinates(*zip(*grid_centroids))     # shape: (N, 2)
    
        distances = cdist(grid_proj, station_proj, metric='euclidean').astype(np.float32) / 1000  # shape: (N, M)
    
        W = self.gaussian_decay(distances, d0=self.d0)
        W[distances > self.d0] = 0  # 超过阈值设为0

        weighted_demand = W.T @ population  # shape: (M,)
    
        weighted_demand = np.where(weighted_demand > 0, weighted_demand, 1.0).astype(np.float32)
    
        #  D_ij = (S_j * W_ij) / (weighted_demand_j)
        #   reshape supplies: (M,) → (1, M)
        #   reshape weighted_demand: (M,) → (1, M)
        D = (W * supplies.reshape(1, -1)) / weighted_demand.reshape(1, -1)  # shape: (N, M)
    
        #  A_i = sum_j D_ij * W_ij
        accessibility_scores = np.sum(D * W, axis=1).astype(np.float32)  # shape: (N,)
    
        self.grids_gdf[self.accessibility_col] = np.nan_to_num(accessibility_scores)
        self.grids_gdf[self.pop_accessibility_col] = self.grids_gdf[self.accessibility_col] * population
    
        del distances, W, D, weighted_demand, station_coords, station_proj, grid_proj, supplies
        gc.collect()
        return True
    
    def calculate_accessibility_2sfca(self):
        if self.grids_gdf.empty or self.station_gdf.empty:
            print("Zero data in the grid")
            return None

        station_coords = np.array([(geom.y, geom.x) for geom in self.station_gdf.geometry], dtype=np.float32)
        supplies = self.station_gdf[self.supply_col].values.astype(np.float32)

        station_proj = self.project_coordinates(*zip(*station_coords))

        grid_centroids = [(geom.centroid.y, geom.centroid.x) for geom in self.grids_gdf.geometry]
        grid_proj = self.project_coordinates(*zip(*grid_centroids))

        distances = cdist(grid_proj, station_proj, metric='euclidean').astype(np.float32) / 1000

        decay_matrix = self.gaussian_decay(distances, d0=self.d0)
        decay_matrix[distances > self.d0] = 0  # 超过阈值设为 0

        population = self.grids_gdf['population'].values.astype(np.float32)
        total_demands = decay_matrix.T @ population
        total_demands = np.where(total_demands > 0, total_demands, 1).astype(np.float32)

        R_j = (supplies / total_demands).astype(np.float32)
        accessibility_scores = (decay_matrix @ R_j).astype(np.float32)

        self.grids_gdf[self.accessibility_col] = np.nan_to_num(accessibility_scores).astype(np.float32)
        self.grids_gdf[self.pop_accessibility_col] = self.grids_gdf[self.accessibility_col] * population

        del distances, decay_matrix, total_demands, R_j, grid_proj, station_proj, station_coords, supplies
        gc.collect()
        return True
    
    def summarize_gini_coefficient(self):

        city_gini_summary = self.grids_gdf.groupby(['name', 'gb']).apply(
            lambda group: calculate_gini(group['population'].values, group[self.pop_accessibility_col].values)
        ).reset_index(name='Gini_Coefficient')
        
        return city_gini_summary

    def summarize_accessibility(self):

        city_accessibility_summary = self.grids_gdf.groupby(['name','gb']).apply(
            lambda group: (group[self.accessibility_col] * group['population']).sum() * 100000 / group['population'].sum()
        ).reset_index(name='Average_Accessibility')
        return city_accessibility_summary
    
    def summarize_spatial__accessibility(self):

        city_spatial_accessibility_summary = self.grids_gdf.groupby(['name','gb']).apply(
            lambda group: (group[self.pop_accessibility_col].sum() / group[self.pop_accessibility_col].count())
        ).reset_index(name='Average_Accessibility')
        return city_spatial_accessibility_summary
    