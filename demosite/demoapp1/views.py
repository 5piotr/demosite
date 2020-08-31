from django.shortcuts import render
from django.views.generic import (View, TemplateView)
from plotly.offline import plot
import plotly.graph_objs as go
import math
import numpy

# Create your views here.

class IndexView(TemplateView):
    template_name = 'demoapp1/index.html'

class AboutMeView(TemplateView):
    template_name = 'demoapp1/about_me.html'

def calculator(request):
    try:
        hue = request.GET['query']
        ret = eval(hue)
        dict = {
            'hue' : hue,
            'ret' : ret,
        }
        return render(request,'demoapp1/calculator.html', context=dict)
    except (NameError, ZeroDivisionError):
        return render(request,'demoapp1/calculator.html', context={'ret' : 'Error'})
    except:
        return render(request,'demoapp1/calculator.html')

def graphing_calculator(request):
    try:
        eq = request.GET['equation']
        x_min = float(request.GET['x_min'])
        x_max = float(request.GET['x_max'])
        # step = float(request.GET['step'])
        # x_data = numpy.arange(x_min,x_max,step)
        x_data = numpy.linspace(x_min,x_max,200,True)
        y_data = [eval(eq) for x in x_data]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_data,y=y_data,mode='lines'))
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
    except (NameError, ZeroDivisionError):
        return render(request,'demoapp1/graphing_calculator.html', context={'ret' : 'Error'})
    except:
        return render(request,'demoapp1/graphing_calculator.html')

def under_construction(request):
    return render(request,'demoapp1/under_construction.html')
