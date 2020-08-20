from django.shortcuts import render
from django.views.generic import (View, TemplateView)

# Create your views here.

class IndexView(TemplateView):
    template_name = 'demoapp1/index.html'

class AboutMeView(TemplateView):
    template_name = 'demoapp1/about_me.html'

class PlotterView(TemplateView):
    template_name = 'demoapp1/plotter.html'

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
