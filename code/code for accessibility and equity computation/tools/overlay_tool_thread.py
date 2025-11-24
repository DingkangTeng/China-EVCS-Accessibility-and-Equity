# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:45:13 2025

@author:  Ruichen MA
"""
import geopandas as gpd
import rasterio
from shapely.geometry import box
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from tools import accessibility_calculator
import matplotlib.pyplot as plt


def read_population_data(population_tif):
    with rasterio.open(population_tif) as src:
        population_data = src.read(1)
        population_transform = src.transform
        population_crs = src.crs
    return population_data, population_transform, population_crs

def create_population_gdf(population_data, population_transform, population_crs):
    print('creating population geodataframe ...')
    rows, cols = np.where(population_data >= 1)
    population_geometries = []
    population_values = []

    for row, col in tqdm(zip(rows, cols), total=len(rows), desc='Processing pixels'):
        x, y = population_transform * (col, row)
        geom = box(x, y, x + population_transform[0], y + population_transform[4])
        population_geometries.append(geom)
        population_values.append(population_data[row, col])

    population_gdf = gpd.GeoDataFrame({'geometry': population_geometries, 'population': population_values}, crs=population_crs)
    return population_gdf

def read_stations_data(stations_file, target_crs):
    print('reading stations data ...')
    stations_gdf = gpd.read_file(stations_file)
    stations_gdf = stations_gdf.to_crs(target_crs)
    stations_gdf = stations_gdf[['OBJECTID','pname','cityname','pcode','citycode','geometry']]
    return stations_gdf


def plot_accessibility_with_stations(grids_gdf, station_gdf, 
                                     column='Accessibility', 
                                     cmap='GnBu', 
                                     title='Charging Accessibility'):

    fig, ax = plt.subplots(figsize=(10, 8))
    
    grids_gdf.plot(
        column=column, ax=ax, legend=True, cmap=cmap,
        edgecolor='lightgrey', linewidth=0.2
    )
    

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    ax.legend()
    plt.tight_layout()
    plt.show()

    
def plot_population_with_stations(grids_gdf, station_gdf, 
                                   column='population', 
                                   cmap='Oranges', 
                                   title='Population Distribution and Charging Stations'):

    fig, ax = plt.subplots(figsize=(10, 8))

    grids_gdf.plot(
        column=column, ax=ax, legend=True, cmap=cmap,
        edgecolor='lightgrey', linewidth=0.2
    )
    
    station_gdf.plot(ax=ax, color='red', markersize=0.5, label='Charging Station')

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    ax.legend()
    plt.tight_layout()
    plt.show()


def process_evse_type(city, joined_pop_gdf, joined_sta_gdf, d0, plot=False):
    print(f"\nProcessing EVSE accessibility in City {city}...")

    joined_sta_gdf = joined_sta_gdf.copy()
    joined_sta_gdf['supply'] = 1

    calculator = accessibility_calculator.AccessibilityCalculator(
        joined_pop_gdf.reset_index(drop=True),
        joined_sta_gdf,
        d0=d0,
        supply_col='supply'
    )
    calculator.calculate_accessibility_m2sfca()

    if plot:
        plot_accessibility_with_stations(calculator.grids_gdf, calculator.station_gdf)
        plot_population_with_stations(calculator.grids_gdf, calculator.station_gdf)

    summary_accessibility = calculator.summarize_accessibility()
    summary_spatial = calculator.summarize_spatial__accessibility()
    gini = calculator.summarize_gini_coefficient()

    print(f"City level Acc: {summary_accessibility}")
    print(f"Gini: {gini}")

    return summary_accessibility, summary_spatial, gini


def overlay_tiff(output_tif_directory: str, tiff_file_after_clipped: str, stations_file: str, country_file: str, d0: float, year: int): 
    print("Step2: Start Overlaying ...")

    output_dir = os.path.join(os.getcwd(), "output", str(year))
    os.makedirs(output_dir, exist_ok=True)

    population_data, population_transform, population_crs = read_population_data(
        os.path.join(output_tif_directory, tiff_file_after_clipped))
    print('Reading population TIFF dataset finished.')

    population_gdf = create_population_gdf(population_data, population_transform, population_crs)
    print('Creating population GeoDataFrame finished.')

    stations_gdf = read_stations_data(stations_file, population_crs)
    print('Reading EVCS data finished.')

    country_gdf = gpd.read_file(country_file, engine='pyogrio')[['name', 'gb', 'geometry']]

    exclude_list = ['澳门特别行政区', '香港特别行政区', '台湾省']
    country_gdf = country_gdf[~country_gdf['name'].isin(exclude_list)]
    print('Reading city boundary data finished.')

    all_accessibility = pd.DataFrame()
    all_spatial = pd.DataFrame()
    all_gini = pd.DataFrame()

    for city in tqdm(list(country_gdf['name'].unique()), desc="Processing cities"):
        city_boundary_gdf = country_gdf[country_gdf['name'] == city]
        joined_pop_gdf = gpd.sjoin(population_gdf, city_boundary_gdf, how='inner')
        joined_sta_gdf = gpd.sjoin(stations_gdf, city_boundary_gdf, how='inner')

        print(f"City: {city}, Station Count: {len(joined_sta_gdf)}, Population: {joined_pop_gdf['population'].sum()}")

        summary, spatial_summary, gini_summary = process_evse_type(
            city, joined_pop_gdf, joined_sta_gdf, d0=d0, plot=False)
        
        all_accessibility = pd.concat([all_accessibility, summary], ignore_index=True)
        all_spatial = pd.concat([all_spatial, spatial_summary], ignore_index=True)
        all_gini = pd.concat([all_gini, gini_summary], ignore_index=True)

    all_accessibility.to_csv(os.path.join(output_dir, f'{year}_city_accessibility_summaries.csv'), index=False)
    all_spatial.to_csv(os.path.join(output_dir, f'{year}_city_spatial_accessibility_summaries.csv'), index=False)
    all_gini.to_csv(os.path.join(output_dir, f'{year}_city_gini_summaries.csv'), index=False)

    print("-" * 50)