import json
from .app import CSES as app
import pandas as pd
import geopandas as gpd


#code for combining json files
def combine_jsons(file_list, BUCKET_NAME, s3):
    all_data_df = gpd.GeoDataFrame()
    for json_file in file_list:
        obj = s3.Object(BUCKET_NAME, json_file)
        stations_geojson = json.load(obj.get()['Body']) 
        gdf = gpd.read_file(obj.get()['Body'])
        all_data_df = pd.concat([all_data_df, gdf]).set_crs(crs= 'EPSG:4326')

    return all_data_df

#code for reach json files
def reach_json(reach_ids,BUCKET, BUCKET_NAME, S3):
        csv_key = 'Streamstats/Streamstats.csv'
        obj = BUCKET.Object(csv_key)
        body = obj.get()['Body']
        Streamstats = pd.read_csv(body)
        Streamstats.pop('Unnamed: 0')
        Streamstats.drop_duplicates(subset = 'NWIS_site_id', inplace = True)
        Streamstats.reset_index(inplace = True, drop = True)

        #Convert to geodataframe
        StreamStats = gpd.GeoDataFrame(Streamstats, geometry=gpd.points_from_xy(Streamstats.dec_long_va, Streamstats.dec_lat_va))

        #the csv loses the 0 in front of USGS ids, fix
        NWIS = list(Streamstats['NWIS_site_id'].astype(str))
        Streamstats['NWIS_site_id'] = ["0"+str(i) if len(i) <8 else i for i in NWIS]  

        #Get streamstats information for each USGS location
        sites = pd.DataFrame()

        for site in reach_ids:
            s = Streamstats[Streamstats['NWIS_site_id'] ==  str(site)]
            sites = pd.concat([sites, s])

        stateids = list(set(list(sites['state_id'])))

        stationpaths = []
        for state in stateids:
            stations_path = f"GeoJSON/StreamStats_{state}_4326.geojson" #will need to change the filename to have state before 4326
            stationpaths.append(stations_path)

        #combine stations
        combined = combine_jsons(stationpaths, BUCKET_NAME, S3)
        
        #get site ids out of DF to make new geojson
        finaldf = gpd.GeoDataFrame()
        for site in reach_ids:
            df = combined[combined['USGS_id'] == site]
            finaldf = pd.concat([finaldf, df])

        #reset index and drop any duplicates
        finaldf.reset_index(inplace = True, drop = True)
        finaldf.drop_duplicates('USGS_id', inplace = True)        

        return finaldf
