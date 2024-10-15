from django import forms


class AddReview(forms.Form):
    review = forms.CharField(strip=True,
                             widget=forms.Textarea(attrs={'rows': 7, 'class': 'form-input'}), label='')
