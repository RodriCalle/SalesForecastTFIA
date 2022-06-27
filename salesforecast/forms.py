from django import forms
from salesforecast.models import FileUpload


class UploadForm(forms.ModelForm):
    class Meta:
        model = FileUpload
        fields = ['file']