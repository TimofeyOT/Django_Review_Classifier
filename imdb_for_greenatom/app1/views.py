from django.shortcuts import render
from .forms import *
from app1.models import *


def index_page(request):
    rating = ''
    status = ''
    if request.method == 'POST':
        form = AddReview(request.POST)
        if form.is_valid():
            rating = main_rater(form.cleaned_data['review'])[0]
            if rating > 6:
                status = 'That qualifies as positive'
            else:
                status = 'That qualifies as negative'
    else:
        form = AddReview()

    return render(request, 'index.html', context={'form': form, 'status': status, 'rating': rating})
