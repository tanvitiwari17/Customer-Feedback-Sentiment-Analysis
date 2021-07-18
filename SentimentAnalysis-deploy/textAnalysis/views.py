from django.shortcuts import render, redirect
from django.contrib import messages
import pickle

with open('mnb.pkl', 'rb') as file:
    mnbModel = pickle.load(file)

with open('tfidf.pkl', 'rb') as file:
    tf = pickle.load(file)

def index(request):

    context = dict()
    if(request.method == "POST"):
        comment = request.POST.get('myComment')

        pred = giveMePredictions(comment)
        if pred == ['pos']:
            context["pred"]='1'
            pred = comment + " is a Positive Feedback"
        else:
            context["pred"]='0'
            pred = comment + " is a Negative Feedback"

        messages.success(request, pred)
    return render(request, 'index.html',context)

def giveMePredictions(comment):
    review_vector = tf.transform([comment])
    pred =  mnbModel.predict(review_vector)
    print(pred)

    return pred







