# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:56:07 2025

@author:  Ruichen MA

The code of running China 2SFCA and M2SFCA Accessibility and Equity from 2015 to 2025
"""

from tools import clip_tool, overlay_tool_thread
import os

if __name__ == "__main__":
    
    """
    load data
    """
    
    for year in range(2015, 2026):
        # usa pop dataset
        input_tif_directory = r'C:\Users\User\data\_SampleData\CN-WorldPOP\raw'
        
        # clipped usa pop dataset
        output_tif_directory = r'C:\Users\User\data\_SampleData\CN-WorldPOP\clipped'
        
        # china boundary
        geojson_path = r'C:\Users\User\data\_SampleData\CN-map\china_map_clean.geojson'
        
        # fix tiff year if beyond 2020
        tiff_year = min(year, 2025)
    
        # specific pop tiff 2014-2020 (use 2020 if year > 2020)
        tiff_file = f'chn_agesex_structures_{tiff_year}_CN_1km.tif'
        
        # clipped pop tiff
        tiff_file_after_clipped = f'chn_ppp_{tiff_year}_1km_Aggregated_UNadj_clipped_usa.tif'
    
        # EVCS dataset (always use current year)
        stations_file = rf'C:\Users\User\data\_SampleData\CN-EV charging station\T{year}.geojson'
    
        """
        accessibility calculation procedure
        """
        # Step1: clip tiff file
        tiff_clipped_path = os.path.join(output_tif_directory, tiff_file)
        
        if not os.path.exists(tiff_clipped_path):
            print(f"Clipped file not found for year {year}, performing clip...")
            clip_tool.clip_tiff(input_tif_directory, output_tif_directory, geojson_path, tiff_file)
        else:
            print(f"Clipped file already exists for year {year}, skipping clip.")
    
        # Step2: overlay pop and EVCS 
        results = overlay_tool_thread.overlay_tiff( output_tif_directory,
                                                    tiff_file_after_clipped,
                                                    stations_file,
                                                    geojson_path,
                                                    d0=1,
                                                    year=year 
                                                    )

