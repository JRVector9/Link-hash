# market/forms.py
from django import forms
from .models import UserProfile

class ProfileForm(forms.ModelForm):
    # ✅ clear 버튼용 (checkbox)
    clear_avatar = forms.BooleanField(required=False, initial=False)

    class Meta:
        model = UserProfile
        fields = ["display_name", "bio", "avatar"]  # ✅ avatar 포함

        widgets = {
            "display_name": forms.TextInput(attrs={
                "class": "form-control",
                "maxlength": "32",
                "placeholder": "Username",
            }),
            "bio": forms.Textarea(attrs={
                "class": "form-control",
                "rows": 4,
                "placeholder": "Write a short intro (optional)",
            }),
        }

    def clean_display_name(self):
        name = (self.cleaned_data.get("display_name") or "").strip()
        if len(name) > 32:
            raise forms.ValidationError("Username must be 32 characters or less.")
        return name
