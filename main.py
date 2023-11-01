import cv2
import os

def extract_letters_from_image(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение порогового значения для выделения текста
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Поиск контуров на изображении
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Извлечение буквных областей и их координат
    letter_images = []
    letter_coordinates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Фильтрация маленьких областей
        if w > 10 and h > 10:
            letter_image = thresholded[y:y + h, x:x + w]
            letter_images.append(letter_image)
            letter_coordinates.append((x, y, w, h))

    # Создание папки "letters" для сохранения изображений
    if not os.path.exists("letters"):
        os.mkdir("letters")

    # Сохранение изображений букв в папке "letters"
    for i, letter_image in enumerate(letter_images):
        cv2.imwrite(f'letters/letter_{i}.png', letter_image)

if __name__ == '__main__':
    image_path = 'test.jpeg'
    extract_letters_from_image(image_path)
