# Automated Medical Image Analysis for Disease Diagnosis Using Deep Learning

This final-year-project imaging platform includes three model workflows:

- Cardiac detection from chest X-ray images
- Pneumonia detection from chest X-ray images
- Atrium segmentation from cardiac MRI NIfTI volumes

## Run the app

### Recommended

```powershell
.\run_app.ps1
```

Then open:

```text
http://127.0.0.1:5000
```

## Default login

On first run the app seeds one administrator account in `medicalai.db`:

- Username: `admin`
- Password: `MedicalAI2026`

You can override the seeded account before starting the app with:

```powershell
$env:MEDICALAI_ADMIN_USER = "your-user"
$env:MEDICALAI_ADMIN_PASSWORD = "your-password"
$env:MEDICALAI_ADMIN_NAME = "Your Name"
.\run_app.ps1
```

## Input rules

- Cardiac detection: upload chest X-ray images in `.png`, `.jpg`, or `.jpeg`
- Pneumonia detection: upload chest X-ray images in `.png`, `.jpg`, or `.jpeg`
- Atrium segmentation: upload cardiac MRI volumes in `.nii` or `.nii.gz`

## Notes

- Chest X-ray workflows generate a classifier score and activation heatmap.
- Atrium segmentation generates an extracted MRI slice, binary mask, and overlay.
- Analysis history and downloadable text reports are stored locally in `medicalai.db`.
