import json
from pathlib import Path
import pandas as pd
import geopandas as gpd

from tethys_sdk.layouts import MapLayout
from tethys_sdk.routing import controller
from .app import CSES as app

#functions to load AWS data
import boto3
import os
from botocore import UNSIGNED 
from botocore.client import Config
import os
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

#Model evaluation metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_percentage_error
import hydroeval as he


#Date picker
from tethys_sdk.gizmos import DatePicker
from django.shortcuts import render, reverse, redirect
from tethys_sdk.gizmos import DatePicker, SelectInput, TextInput
import datetime
from django.http import JsonResponse
from django.urls import reverse_lazy
from datetime import datetime
from datetime import date, timedelta

#Connect web pages
from django.http import HttpResponse 

#utils
from .utils import combine_jsons, reach_json

#Set Global Variables

BUCKET_NAME = 'streamflow-app-data'
S3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
BUCKET = S3.Bucket(BUCKET_NAME) 

#Controller base configurations
BASEMAPS = [
        {'ESRI': {'layer':'NatGeo_World_Map'}},
        {'ESRI': {'layer':'World_Street_Map'}},
        {'ESRI': {'layer':'World_Imagery'}},
        {'ESRI': {'layer':'World_Shaded_Relief'}},
        {'ESRI': {'layer':'World_Topo_Map'}},
        'OpenStreetMap',      
    ]
MAX_ZOOM = 16
MIN_ZOOM = 1
BACK_URL = reverse_lazy('community_streamflow_evaluation_system:home')

#Controller for the state class 
@controller(
    name="state_eval",
    url="state_eval/",
    app_workspace=True,
)   
class State_Eval(MapLayout): 
    # Define base map options
    app = app
    back_url = BACK_URL
    base_template = 'community_streamflow_evaluation_system/base.html'
    map_title = 'State Evaluation Class'
    map_subtitle = 'Evaluate hydrological model performance for a State of interest.'
    basemaps = BASEMAPS
    max_zoom = MAX_ZOOM
    min_zoom = MIN_ZOOM
    show_properties_popup = True  
    plot_slide_sheet = True
    template_name = 'community_streamflow_evaluation_system/state_eval.html' 
   
     
    def get_context(self, request, *args, **kwargs):
        """
        Create context for the Map Layout view, with an override for the map extents based on stream and weather gauges.

        Args:
            request (HttpRequest): The request.
            context (dict): The context dictionary.

        Returns:
            dict: modified context dictionary.
        """

        start_date_picker = DatePicker( 
            name='start-date',
            display_text='Start Date',
            autoclose=False,
            format='mm-dd-yyyy',
            start_date='01-01-1980',
            end_date= '12-30-2020',
            start_view='year',
            today_button=False, 
            initial='01-01-2019'
        ) 
        end_date_picker = DatePicker( 
            name='end-date',
            display_text='End Date',
            start_date='01-01-1980',
            end_date= '12-30-2020',
            autoclose=False,
            format='mm-dd-yyyy',
            start_view='year',
            today_button=False, 
            initial='06-11-2019'
        )
        
        state_id = SelectInput(display_text='Select State',
                                    name='state_id',
                                    multiple=False,
                                    options=[("Alaska", "AK"),
                                            ("Alabama", "AL"),
                                            ("Arizona", "AZ"),
                                            ("Arkansas", "AR"),
                                            ("California", "CA"),
                                            ("Colorado", "CO"),
                                            ("Connecticut", "CT"),
                                            ("Delaware", "DE"),
                                            ("Florida", "FL"),
                                            ("Georgia", "GA"),
                                            ("Hawaii", "HI"),
                                            ("Idaho", "ID"),
                                            ("Illinois", "IL"),
                                            ("Indiana", "IN"),
                                            ("Iowa", "IA"),
                                            ("Kansas", "KS"),
                                            ("Kentucky", "KY"),
                                            ("Louisiana", "LA"),
                                            ("Maine", "ME"),
                                            ("Maryland", "MD"),
                                            ("Massachusetts", "MA"),  
                                            ("Michigan", "MI"),
                                            ("Minnesota", "MN"),
                                            ("Mississippi", "MS"),
                                            ("Missouri", "MO"),
                                            ("Montana", "MT"),
                                            ("Nebraska", "NE"),
                                            ("Nevada", "NV"),
                                            ("New Hampshire", "NH"),
                                            ("New Jersey", "NJ"),
                                            ("New Mexico", "NM"),
                                            ("New York", "NY"),
                                            ("North Carolina", "NC"),
                                            ("North Dakota", "ND"),
                                            ("Ohio", "OH"),
                                            ("Oklahoma", "OK"),
                                            ("Oregon", "OR"),
                                            ("Pennsylvania", "PA"),
                                            ("Rhode Island", "RI"),
                                            ("South Carolina", "SC"),
                                            ("South Dakota", "SD"),
                                            ("Tennessee", "TN"),
                                            ("Texas", "TX"),
                                            ("Utah", "UT"),
                                            ("Vermont", "VT"),
                                            ("Virginia", "VA"),
                                            ("Washington", "WA"),
                                            ("West Virginia", "WV"),   
                                            ("Wisconsin", "WI"),  
                                            ("Wyoming", "WY")
                                        ],
                                    initial=['Alabama'], #it would be cool to change this depending on the current state input.
                                    select2_options={'placeholder': 'Select a State',
                                                    'allowClear': True})
        
        model_id = SelectInput(display_text='Select Model',
                                    name='model_id',
                                    multiple=False,
                                    options=[
                                            ("National Water Model v2.1", "NWM_v2.1"),
                                            ("National Water Model v3.0", "NWM_v3.0"),
                                            ("NWM MLP extension", "MLP"),
                                            ("NWM XGBoost extension", "XGBoost"),
                                            ("NWM CNN extension", "CNN"),
                                            ("NWM LSTM extension", "LSTM"),
                                        
                                            ],
                                    initial=['National Water Model v2.1'],
                                    select2_options={'placeholder': 'Select a model',
                                                    'allowClear': True})

        # Call Super   
        context = super().get_context( 
            request,  
            *args, 
            **kwargs
        )
        context['start_date_picker'] = start_date_picker  
        context['end_date_picker'] = end_date_picker 
        context['state_id'] = state_id
        context['model_id'] = model_id
        return context

    def compose_layers(self, request, map_view, app_workspace, *args, **kwargs): 
        """
        Add layers to the MapLayout and create associated layer group objects.
        """
        try: 
            #http request for user inputs
            state_id = request.GET.get('state_id')
            startdate = request.GET.get('start-date')
            startdate = startdate.strip('][').split(', ')
            enddate = request.GET.get('end-date')
            enddate = enddate.strip('][').split(', ')
            model_id = request.GET.get('model_id')
            model_id = model_id.strip('][').split(', ')
      
            # USGS stations - from AWS s3
            stations_path = f"GeoJSON/StreamStats_{state_id}_4326.geojson" 
            obj = S3.Object(BUCKET_NAME, stations_path)

            # set the map extend based on the stations
            gdf = gpd.read_file(obj.get()['Body'], driver='GeoJSON')
            map_view['view']['extent'] = list(gdf.geometry.total_bounds)

            #update json with start/end date, modelid to support click, adjustment in the get_plot_for_layer_feature()
            gdf['startdate'] = datetime.strptime(startdate[0], '%m-%d-%Y').strftime('%Y-%m-%d')
            gdf['enddate'] = datetime.strptime(enddate[0], '%m-%d-%Y').strftime('%Y-%m-%d')
            gdf['model_id'] = model_id[0]

            stations_geojson = json.loads(gdf.to_json()) 
            stations_geojson.update({"crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" }}})          
        

            stations_layer = self.build_geojson_layer(
                geojson=stations_geojson,
                layer_name='USGS Stations',
                layer_title='USGS Station',
                layer_variable='stations',
                visible=True,
                selectable=True,
                plottable=True,
            ) 

            # Create layer groups
            layer_groups = [
                self.build_layer_group(
                    id='nextgen-features',
                    display_name='NextGen Features',
                    layer_control='checkbox',  # 'checkbox' or 'radio'
                    layers=[
                        stations_layer,
                    ],
                    visible= True
                )
            ]

        except: 
            #Default state id to initiat mapping
            print('No useable inputs, default mapping')
            state_id = 'AL'
    
            # USGS stations - from AWS s3
            stations_path = f"GeoJSON/StreamStats_{state_id}_4326.geojson" #will need to change the filename to have state before 4326
            obj = S3.Object(BUCKET_NAME, stations_path)
            stations_geojson = json.load(obj.get()['Body']) 

            # set the map extend based on the stations
            gdf = gpd.read_file(obj.get()['Body'], driver='GeoJSON')
            map_view['view']['extent'] = list(gdf.geometry.total_bounds)
        

            stations_layer = self.build_geojson_layer(
                geojson=stations_geojson,
                layer_name='USGS Stations',
                layer_title='USGS Station',
                layer_variable='stations',
                visible=True,
                selectable=True,
                plottable=True,
            )

            # Create layer groups
            layer_groups = [
                self.build_layer_group(
                    id='nextgen-features',
                    display_name='NextGen Features',
                    layer_control='checkbox',  # 'checkbox' or 'radio'
                    layers=[
                        stations_layer,
                    ],
                    visible= True
                )
            ]

        return layer_groups


    @classmethod
    def get_vector_style_map(cls):
        return {
            'Point': {'ol.style.Style': {
                'image': {'ol.style.Circle': {
                    'radius': 5,
                    'fill': {'ol.style.Fill': {
                        'color': 'white',
                    }},
                    'stroke': {'ol.style.Stroke': {
                        'color': 'red',
                        'width': 3
                    }}
                }}
            }},
            'MultiPolygon': {'ol.style.Style': {
                'stroke': {'ol.style.Stroke': {
                    'color': 'navy',
                    'width': 3
                }},
                'fill': {'ol.style.Fill': {
                    'color': 'rgba(0, 25, 128, 0.1)'
                }}
            }},
             'MultiLineString': {'ol.style.Style': {
                'stroke': {'ol.style.Stroke': {
                    'color': 'navy',
                    'width': 2
                }},
                'fill': {'ol.style.Fill': {
                    'color': 'rgba(0, 25, 128, 0.1)'
                }}
            }},
        }

    def get_plot_for_layer_feature(self, request, layer_name, feature_id, layer_data, feature_props, app_workspace,
                                *args, **kwargs):
        """
        Retrieves plot data for given feature on given layer.
        Args:
            layer_name (str): Name/id of layer.
            feature_id (str): ID of feature.
            layer_data (dict): The MVLayer.data dictionary.
            feature_props (dict): The properties of the selected feature.

        Returns:
            str, list<dict>, dict: plot title, data series, and layout options, respectively.
      """     

        # Get the feature ids, add start/end date, and model as features in geojson above to have here.
        id = feature_props.get('id') #we could connect the hydrofabric in here for NWM v3.0
        NHD_id = feature_props.get('NHD_id') 
        state = feature_props.get('state')
        startdate= feature_props.get('startdate')
        enddate = feature_props.get('enddate')
        model_id = feature_props.get('model_id')
  
        # USGS observed flow
        if layer_name == 'USGS Stations':
            layout = {
                'yaxis': {
                    'title': 'Streamflow (cfs)'
                },
                'xaxis': {
                    'title': 'Date'
                }
            }  

            #USGS observed flow
            USGS_directory = f"NWIS/NWIS_sites_{state}.h5/NWIS_{id}.csv"
            obj = BUCKET.Object(USGS_directory)
            body = obj.get()['Body']
            USGS_df = pd.read_csv(body)
            USGS_df.pop('Unnamed: 0')  
            

            #modeled flow, starting with NWM
            try:
                #try to use model/date inputs for plotting
                model_directory = f"{model_id}/NHD_segments_{state}.h5/{model_id}_{NHD_id}.csv"  
                obj = BUCKET.Object(model_directory)
                body = obj.get()['Body']
                model_df = pd.read_csv(body)
                model_df.pop('Unnamed: 0')
                modelcols = model_df.columns.to_list()[-2:]
                model_df = model_df[modelcols]

                 #combine Dfs, remove nans
                USGS_df.drop_duplicates(subset=['Datetime'], inplace=True)
                model_df.drop_duplicates(subset=['Datetime'],  inplace=True)
                USGS_df.set_index('Datetime', inplace = True, drop = True)
                model_df.set_index('Datetime', inplace = True, drop = True)
                DF = pd.concat([USGS_df, model_df], axis = 1, join = 'inner')
                #try to select user input dates
                DF = DF.loc[startdate:enddate]
                DF.reset_index(inplace=True)
                
                time_col = DF.Datetime.to_list()#limited to less than 500 obs/days 
                USGS_streamflow_cfs = DF.USGS_flow.to_list()#limited to less than 500 obs/days 
                Mod_streamflow_cfs = DF[f"{model_id[:3]}_flow"].to_list()#limited to less than 500 obs/days

                #calculate model skill
                r2 = round(r2_score(USGS_streamflow_cfs, Mod_streamflow_cfs),2)
                rmse = round(mean_squared_error(USGS_streamflow_cfs, Mod_streamflow_cfs, squared=False),0)
                maxerror = round(max_error(USGS_streamflow_cfs, Mod_streamflow_cfs),0)
                MAPE = round(mean_absolute_percentage_error(USGS_streamflow_cfs, Mod_streamflow_cfs)*100,0)
                kge, r, alpha, beta = he.evaluator(he.kge,USGS_streamflow_cfs,Mod_streamflow_cfs)
                kge = round(kge[0],2)
 
 
                data = [
                    {
                        'name': 'USGS Observed',
                        'mode': 'lines',
                        'x': time_col,
                        'y': USGS_streamflow_cfs,
                        'line': {
                            'width': 2,
                            'color': 'blue'
                        }
                    },
                    { 
                        'name': f"{model_id} Modeled",
                        'mode': 'lines',
                        'x': time_col,
                        'y': Mod_streamflow_cfs,
                        'line': {
                            'width': 2,
                            'color': 'red'
                        }
                    },
                ]
                

                return f"{model_id} and Observed Streamflow at USGS site: {id} <br> RMSE: {rmse} cfs <br> KGE: {kge} <br> MaxError: {maxerror} cfs", data, layout
            
            except:
                print("No user inputs, default configuration.")
                model = 'NWM_v2.1'
                model_directory = f"{model}/NHD_segments_{state}.h5/{model}_{NHD_id}.csv"  #put state in geojson file
                obj = BUCKET.Object(model_directory)
                body = obj.get()['Body']
                model_df = pd.read_csv(body)
                model_df.pop('Unnamed: 0')

                #combine Dfs, remove nans
                USGS_df.drop_duplicates(subset=['Datetime'], inplace=True)
                model_df.drop_duplicates(subset=['Datetime'],  inplace=True)
                USGS_df.set_index('Datetime', inplace = True)
                model_df.set_index('Datetime', inplace = True)
                DF = pd.concat([USGS_df, model_df], axis = 1, join = 'inner')
                DF.reset_index(inplace=True)
                time_col = DF.Datetime.to_list()[:45] 
                USGS_streamflow_cfs = DF.USGS_flow.to_list()[:45] 
                Mod_streamflow_cfs = DF[f"{model[:3]}_flow"].to_list()[:45]

                #calculate model skill
                r2 = round(r2_score(USGS_streamflow_cfs, Mod_streamflow_cfs),2)
                rmse = round(mean_squared_error(USGS_streamflow_cfs, Mod_streamflow_cfs, squared=False),0)
                maxerror = round(max_error(USGS_streamflow_cfs, Mod_streamflow_cfs),0)
                MAPE = round(mean_absolute_percentage_error(USGS_streamflow_cfs, Mod_streamflow_cfs)*100,0)
                kge, r, alpha, beta = he.evaluator(he.kge,USGS_streamflow_cfs,Mod_streamflow_cfs)
                kge = round(kge[0],2)

                data = [
                    {
                        'name': 'USGS Observed',
                        'mode': 'lines',
                        'x': time_col,
                        'y': USGS_streamflow_cfs,
                        'line': {
                            'width': 2,
                            'color': 'blue'
                        }
                    },
                    {
                        'name': f"Default Configuration: NWM v2.1 Modeled",
                        'mode': 'lines',
                        'x': time_col,
                        'y': Mod_streamflow_cfs,
                        'line': {
                            'width': 2,
                            'color': 'red'
                        }
                    },
                ]


                return f'Default Configuration:{model} Observed Streamflow at USGS site: {id} <br> RMSE: {rmse} cfs <br> KGE: {kge} <br> MaxError: {maxerror} cfs', data, layout
            
            


