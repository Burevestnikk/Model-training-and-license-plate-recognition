import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5

model = load_model("PSI_egor_lezov_31099.h5")

new_image_path = 'C:\\Users\\monst\\Desktop\\modell\\images\\Cars0.png'
# new_image_path = 'C:\\Users\\monst\\Desktop\\modell\\testimage\\test5.png'
img = cv2.imread(new_image_path)
img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_normalized = img_resized / 255.0  # Normalize values
X_new = np.array([img_normalized])

predictions = model.predict(X_new)

class_index = np.argmax(predictions[0])
confidence = predictions[0][class_index]
class_label = "Płyta: Znaleziono" if confidence >= CONFIDENCE_THRESHOLD else "Płyta: Nie Znaleziono"

ny = predictions[0] * 255
ny = ny.astype(int)

if confidence >= CONFIDENCE_THRESHOLD:
    image_with_box = img_resized.copy()
    image_with_box = cv2.rectangle(image_with_box, (ny[0], ny[1]), (ny[2], ny[3]), (255, 0, 0), 1)

    x1, y1, x2, y2 = ny[0], ny[1], ny[2], ny[3]

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB))
    plt.title(f"{class_label} | Prawdopodobieństwo: {confidence*100:.0f}%")

    ##-----------------------------------------------------------------------##
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB))

    plt.xlim(min(x1, x2), max(x1, x2))
    plt.ylim(max(y1, y2), min(y1, y2))

    cropped_region = image_with_box[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

    #EasyOCR
    import easyocr
    reader = easyocr.Reader(['en'])
    results = reader.readtext(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))

    try:
        recognized_text = results[0][1]
        plt.title(f"Płyta: {results[0][1]}")
    except IndexError:
        plt.title(f"Płyta: Błąd")
        print(results)

    # #pytesseract
    # import pytesseract
    # pytesseract.pytesseract.tesseract_cmd = r'F:\\Tesseract-OCR\\tesseract.exe'
    # text = pytesseract.image_to_string(cropped_region, config='--psm 11')
    # plt.title(f"Płyta: {text}")
else:
    plt.title(f"Płyta: Błąd")
    image_with_box = img_resized.copy()
    plt.imshow(cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB))

plt.show(block=False)
plt.axis('off')
plt.show()