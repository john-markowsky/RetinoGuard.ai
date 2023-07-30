def get_category(predicted_category):
    if predicted_category == 0: 
        return "Normal"
    elif predicted_category == 1:
        return "Mild"
    elif predicted_category == 2:
        return "Moderate"
    elif predicted_category == 3:
        return "Severe"
    elif predicted_category == 4:
        return "Proliferative"

def get_explanation_text(predicted_category):
    if predicted_category == 0:
        return "No diabetic retinopathy (DR) was detected in this image."
    elif predicted_category == 1:
        return "This means that some small areas of the retina may have damaged blood vessels or swelling."
    elif predicted_category == 2:
        return "Moderate DR was detected in this image. This means that there is a more widespread area of the retina affected by damaged blood vessels or swelling."
    elif predicted_category == 3:
        return "Severe DR was detected in this image. This means that a large portion of the retina is affected by damaged blood vessels or swelling."
    elif predicted_category == 4:
        return "Proliferative diabetic retinopathy (PDR) was detected in this image. This means that there is a significant amount of new blood vessel growth on the retina, which can lead to serious vision problems or even blindness."