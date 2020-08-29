from django.shortcuts import render
from django.views.generic import (View, TemplateView)
from plotly.offline import plot
from plotly.graph_objs import Scatter
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
        return render(request,'demoapp1/calculator.html', context={
                                                            'ret' : 'Error'
                                                            })
    except:
        return render(request,'demoapp1/calculator.html')

def graphing_calculator(request):
    try:
        eq = request.GET['equation']
        x_min = float(request.GET['x_min'])
        x_max = float(request.GET['x_max'])
        step = float(request.GET['step'])
        x_data = numpy.arange(x_min,x_max,step)
        y_data = [eval(eq) for x in x_data]
        plot_div = plot([Scatter(x=x_data, y=y_data,
                            mode='lines', name='test',
                            opacity=0.8, marker_color='green')],
                            output_type='div')
        return render(request, 'demoapp1/graphing_calculator.html', context={'plot_div': plot_div})
    except (NameError, ZeroDivisionError):
        print(1)
        return render(request,'demoapp1/graphing_calculator.html', context={
                                                            'ret' : 'Error'
                                                            })
    except:
        print(2)
        return render(request,'demoapp1/graphing_calculator.html')

    x_min = float(request.GET['x_min'])
    x_max = float(request.GET['x_max'])
    step = float(request.GET['step'])
    x_data = range(x_min,x_max)
    print(type(x_data))
