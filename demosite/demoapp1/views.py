from django.shortcuts import render
from django.views.generic import (View, TemplateView)
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.express as px
import math
import numpy
import PIL
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import joblib
from sklearn.cluster import KMeans
import pickle as pkl
import pandas as pd

# Create your views here.

class IndexView(TemplateView):
    template_name = 'demoapp1/index.html'

class AboutView(TemplateView):
    template_name = 'demoapp1/about.html'

def simple_gesture_recognition(request):
    try:
        img_size = (92,70)
        pred_dict = {0:'paper', 1:'rock', 2:'scissors'}
        model = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)),'2022-12-20--18-42_best'))
        # load
        img = Image.open(request.FILES['picture'])
        # resize
        img_resized = img.resize(img_size)
        # cast to array
        img_array = np.array(img_resized)
        # flatten for clustering
        img_flat = img_array.reshape(-1,3)
        # clustering
        kmeans = KMeans(n_clusters=2, random_state=0).fit(img_flat)
        # generate clustered image
        for j in np.unique(kmeans.labels_):
            img_flat[kmeans.labels_==j,:] = kmeans.cluster_centers_[j]
        img_k = img_flat.reshape(img_array.shape)
        img_k = Image.fromarray(img_k)
        # convert to gray scale and fing edges
        img_e = img_k.convert('L')
        img_e = img_e.filter(ImageFilter.FIND_EDGES)
        # crop image to remove edges
        img_e = img_e.crop((1,1,img_size[0]-1,img_size[1]-1))
        # predict
        img_e_p = np.array(img_e)
        img_e_p = img_e_p.reshape(1, img_size[1]-2, img_size[0]-2, 1)
        img_e_p = img_e_p/255
        pred = model.predict(img_e_p)
        label_no = np.argmax(pred)
        label = pred_dict[label_no]

        # generating plots
        images = [img_resized,img_k,img_e]
        plots = []

        for image in images:
            fig = px.imshow(image)
            fig.update_layout(coloraxis_showscale=False, margin=dict(l=0,r=0,b=5,t=5), height=300,
                                paper_bgcolor='#d4eada', hovermode=False, dragmode=False)
            fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            fig_div = plot(fig, output_type='div')
            plots.append(fig_div)

        return render(request,'demoapp1/simple_gesture_recognition.html', context={'plot_div_r':plots[0],
                                                                                    'plot_div_k':plots[1],
                                                                                    'plot_div_e':plots[2],
                                                                                    'label':label})

    except (ValueError, PIL.UnidentifiedImageError):
        return render(request,'demoapp1/simple_gesture_recognition.html', context={'plot_div' : 'Error'})
    except:
        return render(request,'demoapp1/simple_gesture_recognition.html')

def rps_cnn_details(request):
    return render(request,'demoapp1/rps_cnn_details.html')

def apartment_price_estimator(request):
    try:
        lat = float(request.POST['lat'])
        lng = float(request.POST['lng'])
        if (lat < 48.9 or lat > 54.8) or (lng < 14 or lng > 24.2):
            raise ValueError('Please select location in Poland')

        if request.POST['market'] == 'primary':
            market = 'primary_market'
        elif request.POST['market'] == 'aftermarket':
            market = 'aftermarket'
        built = float(request.POST['built'])
        area = float(request.POST['area'])
        rooms = request.POST['rooms']
        floor = request.POST['floor']
        floors = request.POST['floors']
        if int(request.POST['floor']) > int(request.POST['floors']):
            raise ValueError('Selected floor should be in complience with the total number of floors')

        # cluster assignment
        kmeans = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'k_means_model'))
        cluster = kmeans.predict([[lat,lng]])[0]

        # preparint input for estimation
        infile = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'dummy_apartment_frame'),'rb')
        dummy_frame = pkl.load(infile)
        infile.close()

        dummy_frame['area'] = area
        dummy_frame['build_yr'] = built
        if market == 'primary_market':
            dummy_frame.market_primary_market = 1
        if rooms != '1':
            dummy_frame['rooms_' + str(rooms)] = 1
        if floor != '0':
            dummy_frame['floor_' + str(floor)] = 1
        if floors != '0':
            dummy_frame['floors_' + str(floors)] = 1
        if cluster != 0:
            dummy_frame['cluster_' + str(cluster)] = 1

        # ann estimation
        model_ann = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)),'ann_model_2023-11-12--13-21'))
        scaler_ann = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'scaler'))
        pred_ann = model_ann.predict(scaler_ann.transform(dummy_frame))[0][0]
        pred_ann = int(pred_ann)

        # random forest estimation
        model_rf = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'random_forest_model'))
        pred_rf = model_rf.predict(dummy_frame)[0]
        pred_rf = int(pred_rf)

        # prices per sqr m
        pred_ann_m = pred_ann/area
        pred_rf_m = pred_rf/area

        # pd.set_option('max_columns', None)
        # print('lat',lat,type(lat),'\n',
        # 'lng',lng,type(lng),'\n',
        # 'market',market,type(market),'\n',
        # 'built',built,type(built),'\n',
        # 'area',area,type(area),'\n',
        # 'rooms',rooms,type(rooms),'\n',
        # 'floor',floor,type(floor),'\n',
        # 'floors',floors,type(floors),'\n',
        # 'cluster',cluster,type(cluster),'\n\n',
        # 'pred_ann',pred_ann,type(pred_ann),'\n',
        # 'pred_rf',pred_rf,type(pred_rf),'\n\n',
        # dummy_frame)

        return render(request,'demoapp1/apartment_price_estimator.html',
                        context={
                        'lat':round(lat,4),
                        'lng':round(lng,4),
                        'market':request.POST['market'],
                        'built':int(built),
                        'area':int(area),
                        'rooms':rooms,
                        'floor':floor,
                        'floors':floors,
                        'pred_ann':pred_ann,
                        'pred_rf':pred_rf,
                        'pred_ann_m':int(pred_ann_m),
                        'pred_rf_m':int(pred_rf_m)
                        })
    except (ValueError) as e:
        return render(request,'demoapp1/apartment_price_estimator.html', context={'pred_ann':'Error',
                                                                                    'e':e})

    except:
        return render(request,'demoapp1/apartment_price_estimator.html')

def apartment_estimator_details(request):
    return render(request,'demoapp1/apartment_estimator_details.html')

def privacy_policy(request):
    return render(request,'demoapp1/privacy_policy.html')

def under_construction(request):
    return render(request,'demoapp1/under_construction.html')
