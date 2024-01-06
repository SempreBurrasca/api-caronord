import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
import hashlib
from eth_account import Account
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets
import string

# Carica il modello .h5
model = load_model('./modelloClassificazioneCappelli-v1.h5')
# Nomi delle classi
class_names = {
    0: 'cap_1',
    1: 'cap_10',
    2: 'cap_11',
    3: 'cap_12',
    4: 'cap_13',
    5: 'cap_14',
    6: 'cap_15',
    7: 'cap_2',
    8: 'cap_3',
    9: 'cap_4',
    10: 'cap_5',
    11: 'cap_6',
    12: 'cap_7',
    13: 'cap_8',
    14: 'cap_9'
}

def get_passphrase(class_name):
    return st.secrets[class_name]

def preprocess_image(image, target_size=(224, 224)):
    """Pre-elabora l'immagine per la classificazione usando la funzione standard di DenseNet."""
    # Converti l'immagine PIL in un array NumPy
    img_array = img_to_array(image)

    # Calcola il centro dell'immagine
    center_x, center_y = img_array.shape[1] // 2, img_array.shape[0] // 2
    half_size = min(center_x, center_y)

    # Ritaglia l'immagine mantenendo il centro
    cropped_img = img_array[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

    # Ridimensiona al target_size
    resized_img = tf.image.resize(cropped_img, target_size)

    # Converti il tensore TensorFlow in un array NumPy
    resized_img_array = resized_img.numpy()

    resized_img_array = np.expand_dims(resized_img_array, axis=0)
    resized_img_array = preprocess_input(resized_img_array)  # Pre-elaborazione specifica per DenseNet
    return resized_img_array

def generate_ethereum_wallet(class_name, passphrase):
    # Calcola il salt basato sulla classe dell'immagine
    salt = hashlib.sha256(class_name.encode()).digest()

    # Configura il KDF per la generazione della chiave privata
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        iterations=100000,  # Configura il numero di iterazioni come preferisci
        salt=salt,
        length=32,  # Specifica la lunghezza della chiave in byte (32 byte per una chiave Ethereum)
        backend=default_backend()
    )

    # Genera la chiave privata
    private_key = kdf.derive(passphrase.encode())

    # Genera un indirizzo di wallet Ethereum dalla chiave privata
    private_key_hex = private_key.hex()
    account = Account.from_key(private_key_hex)
    wallet_address = account.address

    return private_key_hex, wallet_address

def classify_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]  # Ottieni il primo (e unico) risultato

    # Trova la classe con la probabilità più alta che supera il 98%
    max_prob_index = np.argmax(prediction)
    max_prob = prediction[max_prob_index]

    if max_prob > 0.98:  # 98% di probabilità
        class_name = class_names[max_prob_index]
        passphrase = get_passphrase(class_name)
        private_key, wallet_address = generate_ethereum_wallet(class_name, passphrase)
        return f"Questo è il cappello: {class_name}: {max_prob * 100:.2f}%\nIndirizzo ETH: {wallet_address}\nChiave privata: {private_key}\n"
    else:
        return "C'è un errore con il tuo cappello. Contatta il venditore o prova a fare una foto migliore."

def main():
    st.title("Autenticatore P.A.T.T.E.R.N. ")

    uploaded_file = st.file_uploader("Carica un'immagine", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Immagine caricata', use_column_width=True)

        if st.button('Autentica Cappello'):
            prediction = classify_image(image)
            st.write(prediction)

if __name__ == '__main__':
    main()
