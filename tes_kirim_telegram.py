import cv2
import requests

img_url = 'https://ajang.avisha.id/wp-content/uploads/2024/05/image-1.png'  # URL gambar yang ingin dikirim
token = "6992326041:AAGLgeu8d8r-3YAD4PeLM6Zfh287l6Ws4nw"  # Token bot Telegram
chat_id = "262249300"  # ID chat di Telegram
caption = "People Detected!!! "

def send_telegram_photo_from_frame(frame, token, chat_id, caption):
    url = f'https://api.telegram.org/bot{token}/sendPhoto'
    # Mengonversi frame OpenCV ke format yang dapat digunakan oleh requests
    _, img_encoded = cv2.imencode('.png', frame)
    files = {'photo': img_encoded.tobytes()}
    params = {'chat_id': chat_id, 'caption': caption}
    resp = requests.post(url, params=params, files=files)
    print(f'Response Code: {resp.status_code}')

# Contoh penggunaan fungsi sendTelegramPhotoFromFrame()
cap = cv2.VideoCapture(1)  # Mengambil video dari kamera
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    send_telegram_photo_from_frame(frame, token, chat_id, caption)
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
