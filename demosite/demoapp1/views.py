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

# def calculator(request):
#     try:
#         hue = request.POST['query']
#         ret = eval(hue)
#         dict = {
#             'hue' : hue,
#             'ret' : ret,
#         }
#         return render(request,'demoapp1/calculator.html', context=dict)
#     except (NameError, ZeroDivisionError, ValueError, TypeError):
#         return render(request,'demoapp1/calculator.html', context={'ret' : 'Error'})
#     except:
#         return render(request,'demoapp1/calculator.html')
#
# def graphing_calculator(request):
#     try:
#         eq = request.POST['equation']
#         x_min = float(request.POST['x_min'])
#         x_max = float(request.POST['x_max'])
#         # step = float(request.GET['step'])
#         # x_data = numpy.arange(x_min,x_max,step)
#         x_data = numpy.linspace(x_min,x_max,200,True)
#         y_data = [eval(eq) for x in x_data]
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=x_data,y=y_data,mode='lines'))
#         fig.update_layout(margin=dict(l=10, r=10, b=10, t=10))
#         fig.update_layout(
#                 title={'text': 'f(x) = ' + eq,
#                         'y':0.9,
#                         'x':0.5,
#                         'xanchor': 'center',
#                         'yanchor': 'top'},
#                 xaxis_title="x",
#                 yaxis_title="f(x)",
#                 paper_bgcolor="#caf0f8",
#                 plot_bgcolor='#d8f3dc')
#         plot_div = plot(fig, output_type='div')
#         return render(request, 'demoapp1/graphing_calculator.html', context={'plot_div': plot_div})
#     except (NameError, ZeroDivisionError, ValueError, TypeError, SyntaxError, AttributeError):
#         return render(request,'demoapp1/graphing_calculator.html', context={'plot_div' : 'Error'})
#     except:
#         return render(request,'demoapp1/graphing_calculator.html')

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
        fig_r = px.imshow(img_resized)
        fig_r.update_layout(coloraxis_showscale=False, margin=dict(l=0,r=0,b=5,t=5), height=300,
                            paper_bgcolor='#d4eada', hovermode=False, dragmode=False)
        fig_r.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

        fig_k = px.imshow(img_k)
        fig_k.update_layout(coloraxis_showscale=False, margin=dict(l=0,r=0,b=5,t=5), height=300,
                            paper_bgcolor='#d4eada', hovermode=False, dragmode=False)
        fig_k.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

        fig_e = px.imshow(img_e)
        fig_e.update_layout(coloraxis_showscale=False, margin=dict(l=0,r=0,b=5,t=5), height=300,
                            paper_bgcolor='#d4eada', hovermode=False, dragmode=False)
        fig_e.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

        plot_div_r = plot(fig_r, output_type='div')
        plot_div_k = plot(fig_k, output_type='div')
        plot_div_e = plot(fig_e, output_type='div')
        
        return render(request,'demoapp1/simple_gesture_recognition.html', context={'plot_div_r':plot_div_r,
                                                                                    'plot_div_k':plot_div_k,
                                                                                    'plot_div_e':plot_div_e,
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
        model_ann = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)),'ann_model_2022-06-19--11-31'))
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
