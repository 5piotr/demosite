from django.shortcuts import render
from django.views.generic import (View, TemplateView)
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.express as px
import math
import numpy
import PIL
from PIL import Image, ImageOps
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

def calculator(request):
    try:
        hue = request.POST['query']
        ret = eval(hue)
        dict = {
            'hue' : hue,
            'ret' : ret,
        }
        return render(request,'demoapp1/calculator.html', context=dict)
    except (NameError, ZeroDivisionError, ValueError, TypeError):
        return render(request,'demoapp1/calculator.html', context={'ret' : 'Error'})
    except:
        return render(request,'demoapp1/calculator.html')

def graphing_calculator(request):
    try:
        eq = request.POST['equation']
        x_min = float(request.POST['x_min'])
        x_max = float(request.POST['x_max'])
        # step = float(request.GET['step'])
        # x_data = numpy.arange(x_min,x_max,step)
        x_data = numpy.linspace(x_min,x_max,200,True)
        y_data = [eval(eq) for x in x_data]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_data,y=y_data,mode='lines'))
        fig.update_layout(margin=dict(l=10, r=10, b=10, t=10))
        fig.update_layout(
                title={'text': 'f(x) = ' + eq,
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                xaxis_title="x",
                yaxis_title="f(x)")
        plot_div = plot(fig, output_type='div')
        return render(request, 'demoapp1/graphing_calculator.html', context={'plot_div': plot_div,
                                                                                'eq' : eq})
    except (NameError, ZeroDivisionError, ValueError, TypeError):
        return render(request,'demoapp1/graphing_calculator.html', context={'ret' : 'Error'})
    except:
        return render(request,'demoapp1/graphing_calculator.html')

def simple_gesture_recognition(request):
    try:
        if len(request.FILES['picture']) >= 4194304:
            raise ValueError('file too big')
        if Image.open(request.FILES['picture']).format not in ['JPEG','PNG']:
            raise ValueError('incorrect file type')

        wit = 300
        hei = int(wit*3/4)

        picture = Image.open(request.FILES['picture'])
        picture = picture.resize((wit,hei))
        fig = px.imshow(picture)
        fig.update_layout(coloraxis_showscale=False)
        fig.update_layout(width=wit, height=hei, margin=dict(l=10, r=10, b=10, t=10))
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

        model = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)),'2020-11-13--17-15_best'))
        pred_dict = {0:'paper', 1:'rock', 2:'scissors'}
        my_image = picture.resize((90,60))
        my_image = image.img_to_array(my_image)
        my_image_p = np.expand_dims(my_image, axis=0)
        label_no = np.argmax(model.predict(my_image_p))
        label = pred_dict[label_no]

        plot_div = plot(fig, output_type='div')
        return render(request,'demoapp1/simple_gesture_recognition.html', context={'plot_div': plot_div,
                                                                                    'label':label,
                                                                                    'hue':'hue'})
    except (ValueError, PIL.UnidentifiedImageError):
        return render(request,'demoapp1/simple_gesture_recognition.html', context={'ret' : 'Error'})
    except:
        return render(request,'demoapp1/simple_gesture_recognition.html')

def rps_cnn_details(request):
    return render(request,'demoapp1/rps_cnn_details.html')

def apartment_price_estimator(request):
    # try:
    lat = float(request.POST['lat'])
    lng = float(request.POST['lng'])
    if request.POST['market'] == 'primary':
        market = 'pierwotny'
    elif request.POST['market'] == 'aftermarket':
        market = 'wtorny'
    built = float(request.POST['built'])
    area = float(request.POST['area'])
    if request.POST['rooms'] == '> 8':
        rooms = 'więcej niż 8'
    else:
        rooms = request.POST['rooms']
    if request.POST['floor'] == '0':
        floor = 'parter'
    elif request.POST['floor'] == '30 or more':
        floor = 'powyżej 30'
    else:
        floor = request.POST['floor']
    if request.POST['floors'] == '0':
        floors = '0 (parter)'
    elif request.POST['floors'] == '30 or more':
        floors = 'powyżej 30'
    else:
        floors = request.POST['floors']

    # cluster assignment
    kmeans = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'500_means_cls'))
    cluster = kmeans.predict([[lat,lng]])[0]

    # preparint input for estimation
    infile = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'dummy_apartment_frame'),'rb')
    dummy_frame = pkl.load(infile)
    infile.close()

    dummy_frame.area = area
    dummy_frame.build_yr = built
    if market == 'wtorny':
        dummy_frame.market_wtorny = 1
    if rooms != '1':
        dummy_frame['rooms_' + str(rooms)] = 1
    if floor != '1':
        dummy_frame['floor_' + str(floor)] = 1
    if floors != '0 (parter)':
        dummy_frame['floors_' + str(floors)] = 1
    if cluster != 0:
        dummy_frame['cluster_' + str(cluster)] = 1

    # ann estimation
    model_ann = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)),'500a1_2021-01-13--15-52'))
    scaler_ann = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'scaler_500a1'))
    pred_ann = model_ann.predict(scaler_ann.transform(dummy_frame))[0][0]
    pred_ann = int(pred_ann)

    # random forest estimation
    model_rf = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'random_forest_model_a1'))
    pred_rf = model_rf.predict(dummy_frame)[0]
    pred_rf = int(pred_rf)

    # pd.set_option('max_columns', None)
    print('lat',lat,type(lat),'\n',
    'lng',lng,type(lng),'\n',
    'market',market,type(market),'\n',
    'built',built,type(built),'\n',
    'area',area,type(area),'\n',
    'rooms',rooms,type(rooms),'\n',
    'floor',floor,type(floor),'\n',
    'floors',floors,type(floors),'\n',
    'cluster',cluster,type(cluster),'\n\n',
    'pred_ann',pred_ann,type(pred_ann),'\n',
    'pred_rf',pred_rf,type(pred_rf),'\n\n',
    dummy_frame)

    return render(request,'demoapp1/apartment_price_estimator.html',
                    context={
                    'lat':lat,
                    'lng':lng,
                    'market':market,
                    'built':built,
                    'area':area,
                    'rooms':rooms,
                    'floor':floor,
                    'floors':floors
                    })

    # except:
    #     return render(request,'demoapp1/apartment_price_estimator.html')

def under_construction(request):
    return render(request,'demoapp1/under_construction.html')
