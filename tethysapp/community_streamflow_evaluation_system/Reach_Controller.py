import json
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
from botocore.exceptions import ClientError
import os
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

#Model evaluation metrics
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_percentage_error
import hydroeval as he


#Date picker
from tethys_sdk.gizmos import DatePicker
from tethys_sdk.gizmos import DatePicker, SelectInput, TextInput
import datetime
from django.http import JsonResponse
from django.urls import reverse_lazy
from datetime import datetime
from datetime import date, timedelta


#utils
from .utils import combine_jsons, reach_json


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





#Controller for the Reach class
@controller(
    name="reach_eval",
    url="reach_eval/",
    app_workspace=True,
)   
class Reach_Eval(MapLayout): 
    # Define base map options
    app = app
    back_url = BACK_URL
    base_template = 'community_streamflow_evaluation_system/base.html'
    map_title = 'Reach Evaluation Class'
    map_subtitle = 'Evaluate hydrological model performance for reaches of interest.'
    basemaps = BASEMAPS
    max_zoom = MAX_ZOOM
    min_zoom = MIN_ZOOM
    show_properties_popup = True  
    plot_slide_sheet = True
    template_name = 'community_streamflow_evaluation_system/reach_eval.html' 
    
     
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
        
        reach_ids = TextInput(display_text='Enter a USGS site',
                                   name='reach_ids', 
                                   placeholder= 'e.g.: 10224000',
                                   )

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
        context['reach_ids'] = reach_ids
        context['model_id'] = model_id
        return context


    def compose_layers(self, request, map_view, app_workspace, *args, **kwargs): #can we select the geojson files from the input fields (e.g: AL, or a dropdown)
        """
        Add layers to the MapLayout and create associated layer group objects.
        """
       
     
        try: 
             #http request for user inputs
            startdate = request.GET.get('start-date')
            startdate = startdate.strip('][').split(', ')
            enddate = request.GET.get('end-date')
            enddate = enddate.strip('][').split(', ')
            model_id = request.GET.get('model_id')
            model_id = model_id.strip('][').split(', ')
            reach_ids = request.GET.get('reach_ids')
            reach_ids = reach_ids.strip('][').split(', ')

            # USGS stations - from AWS s3
            finaldf = reach_json(reach_ids,BUCKET, BUCKET_NAME, S3)

            #update json with start/end date, modelid to support click, adjustment in the get_plot_for_layer_feature()
            finaldf['startdate'] = datetime.strptime(startdate[0], '%m-%d-%Y').strftime('%Y-%m-%d')
            finaldf['enddate'] = datetime.strptime(enddate[0], '%m-%d-%Y').strftime('%Y-%m-%d')
            finaldf['model_id'] = model_id[0]
            
            '''
            This might be the correct location to determine model performance, this will determine icon color as a part of the geojson file below
            We can also speed up the app by putting all model preds into one csv per state and all obs in one csv per state. - load one file vs multiple.
            '''

            map_view['view']['extent'] = list(finaldf.geometry.total_bounds)
            stations_geojson = json.loads(finaldf.to_json()) 
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
            print('No inputs, going to defaults')
            # #put in some defaults
            # reach_ids = ['10126000', '10068500']
            # startdate = '01-01-2019' 
            # enddate = '01-02-2019'
            # modelid = 'NWM_v2.1'
            # finaldf = reach_json(reach_ids,BUCKET, BUCKET_NAME, S3)
            # map_view['view']['extent'] = list(finaldf.geometry.total_bounds)
            # stations_geojson = json.loads(finaldf.to_json()) 
            # stations_geojson.update({"crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" }}}) 
            stations_geojson = {
                "type": "FeatureCollection",
                "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
                "features": []
            }

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

  
        startdate = request.session.get('start_date', '')
        enddate = request.session.get('end_date', '')
        model_id = request.session.get('model_id', '')

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

            # #USGS observed flow
            # USGS_directory = f"NWIS/NWIS_sites_{state}.h5/NWIS_{id}.csv"
            # obj = BUCKET.Object(USGS_directory)
            # body = obj.get()['Body']
            # USGS_df = pd.read_csv(body)
            # USGS_df.pop('Unnamed: 0')  
            

            try:
                USGS_directory = f"NWIS/NWIS_sites_{state}.h5/NWIS_{id}.csv"
                obj = BUCKET.Object(USGS_directory)
                body = obj.get()['Body']
                USGS_df = pd.read_csv(body)
                USGS_df.pop('Unnamed: 0')
                USGS_df.reset_index(inplace=True)
                USGS_df.drop_duplicates(subset=['Datetime'], inplace=True)
                USGS_df.set_index('Datetime', inplace = True, drop = True)

            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    print(f"File not found: {USGS_directory}")
                    # Handle missing file, perhaps return an empty DataFrame or a helpful message
                    USGS_df = pd.DataFrame()  # or handle accordingly


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
                # USGS_df.drop_duplicates(subset=['Datetime'], inplace=True)
                # USGS_df.set_index('Datetime', inplace = True, drop = True)

                model_df.drop_duplicates(subset=['Datetime'],  inplace=True)
                model_df.set_index('Datetime', inplace = True, drop = True)
                DF = pd.concat([USGS_df, model_df], axis = 1, join = 'inner')
                #try to select user input dates
                DF = DF.loc[startdate:enddate]
                DF.reset_index(inplace=True)
                DF = DF.dropna()
                if DF.empty:
                    data = []
                    return f'No data for Default Configuration:{model} Observed Streamflow at USGS site: {id}', data, layout
                
                time_col = DF.Datetime.to_list()#limited to less than 500 obs/days 
                USGS_streamflow_cfs = DF.USGS_flow.to_list()#limited to less than 500 obs/days 
                Mod_streamflow_cfs = DF[f"{model_id[:3]}_flow"].to_list()#limited to less than 500 obs/days

                #calculate model skill
                # r2 = round(r2_score(USGS_streamflow_cfs, Mod_streamflow_cfs),2)
                rmse = round(root_mean_squared_error(USGS_streamflow_cfs, Mod_streamflow_cfs),0)
                maxerror = round(max_error(USGS_streamflow_cfs, Mod_streamflow_cfs),0)
                # MAPE = round(mean_absolute_percentage_error(USGS_streamflow_cfs, Mod_streamflow_cfs)*100,0)
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
                

                return f'{model_id} and Observed Streamflow at USGS site: {id} <p style="font-size:15px;margin-bottom: 0px;"> RMSE: {rmse}</p> cfs <p style="font-size:15px;margin-bottom: 0px;"> KGE: {kge} </p> <p style="font-size:15px;margin-bottom: 0px;"> MaxError: {maxerror} cfs </p>', data, layout
            except:
                try:
                    print("No user inputs, default configuration.")
                    model = 'NWM_v2.1'
                    model_directory = f"{model}/NHD_segments_{state}.h5/{model}_{NHD_id}.csv"  #put state in geojson file
                    obj = BUCKET.Object(model_directory)
                    body = obj.get()['Body']
                    model_df = pd.read_csv(body)
                    model_df.pop('Unnamed: 0')

                    #combine Dfs, remove nans
                    # USGS_df.drop_duplicates(subset=['Datetime'], inplace=True)
                    # USGS_df.set_index('Datetime', inplace = True)
                    model_df.drop_duplicates(subset=['Datetime'],  inplace=True)
                    model_df.set_index('Datetime', inplace = True)
                    DF = pd.concat([USGS_df, model_df], axis = 1, join = 'inner')
                    DF.reset_index(inplace=True)
                    DF = DF.dropna()

                    if DF.empty:
                        data = []
                        return f'No data for Default Configuration:{model} Observed Streamflow at USGS site: {id}', data, layout

                    time_col = DF.Datetime.to_list()[:45] 
                    USGS_streamflow_cfs = DF.USGS_flow.to_list()[:45] 
                    Mod_streamflow_cfs = DF[f"{model[:3]}_flow"].to_list()[:45]

                    #calculate model skill
                    # r2 = round(r2_score(USGS_streamflow_cfs, Mod_streamflow_cfs),2)
                    rmse = round(root_mean_squared_error(USGS_streamflow_cfs, Mod_streamflow_cfs),0)
                    maxerror = round(max_error(USGS_streamflow_cfs, Mod_streamflow_cfs),0)
                    # MAPE = round(mean_absolute_percentage_error(USGS_streamflow_cfs, Mod_streamflow_cfs)*100,0)
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


                    return f'Default Configuration:{model} Observed Streamflow at USGS site: {id} <p style="font-size:15px;margin-bottom: 0px;"> RMSE: {rmse} cfs </p> <p style="font-size:15px;margin-bottom: 0px;">KGE: {kge}</p> <p style="font-size:15px;margin-bottom: 0px;">MaxError: {maxerror} cfs</p>', data, layout
                except:
                    data = []
                    return f'No data for Default Configuration:{model} Observed Streamflow at USGS site: {id}', data, layout
                
    def update_reach_eval_data(self, request, *args, **kwargs):
        """Respond to AJAX calls from the map page."""
        data = request.POST or request.json()
        request.session['model_id'] = data.get('model_id')
        request.session['start_date'] = data.get('start_date')
        request.session['end_date'] = data.get('end_date')
        request.session['reach_ids'] = data.get('reach_ids')

        reach_ids = request.session['reach_ids']
        reach_ids = reach_ids.strip('][').split(', ')

        try:
            finaldf = reach_json(reach_ids,BUCKET, BUCKET_NAME, S3)
            startdate= request.session['start_date']
            enddate = request.session['end_date']
            model_id = request.session['model_id']

            #update json with start/end date, modelid to support click, adjustment in the get_plot_for_layer_feature()
            finaldf['startdate'] = datetime.strptime(startdate, '%m-%d-%Y').strftime('%Y-%m-%d')
            finaldf['enddate'] = datetime.strptime(enddate, '%m-%d-%Y').strftime('%Y-%m-%d')
            finaldf['model_id'] = model_id[0]

            stations_geojson = json.loads(finaldf.to_json()) 
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
            msg = f"Updated data for the selected reach ID"
        except Exception as e:
            print(f"Error: {e}")
            print('No inputs, going to defaults')
            # #put in some defaults
            # reach_ids = ['10126000', '10068500']
            # startdate = '01-01-2019' 
            # enddate = '01-02-2019'
            # finaldf = reach_json(reach_ids,BUCKET, BUCKET_NAME, S3)
            # stations_geojson = json.loads(finaldf.to_json()) 
            # stations_geojson.update({"crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" }}}) 
            stations_geojson = {
                "type": "FeatureCollection",
                "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
                "features": []
            }

            stations_layer = self.build_geojson_layer(
                geojson=stations_geojson,
                layer_name='USGS Stations',
                layer_title='USGS Station',
                layer_variable='stations',
                visible=True,
                selectable=True,
                plottable=True,
            )
            msg = f"No data available for the selected reach ID"
        return JsonResponse({'success': True, 'message': msg ,'metadata': stations_layer ,'geojson': stations_geojson})