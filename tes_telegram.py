import requests

def send_telegram_photo(img_url, token, chat_id, caption):
    url = f'https://api.telegram.org/bot{token}/sendPhoto'
    files = {'photo': requests.get(img_url).content}
    params = {'chat_id': chat_id, 'caption': caption}
    resp = requests.post(url, params=params, files=files)
    print(f'Response Code: {resp.status_code}')

# Contoh penggunaan fungsi sendTelegram()
img_url = 'https://ajang.avisha.id/wp-content/uploads/2024/05/image-1.png'  # URL gambar yang ingin dikirim
token = "6992326041:AAGLgeu8d8r-3YAD4PeLM6Zfh287l6Ws4nw"  # Token bot Telegram
chat_id = "262249300"  # ID chat di Telegram
caption = "People Detected!!! "
send_telegram_photo(img_url, token, chat_id, caption)
