import telegram
from telegram.ext import CommandHandler, MessageHandler, filters
import tensorflow as tf
import numpy as np

# Load the CNN model
model = tf.keras.models.load_model('my_model.h5')

# Create a Telegram bot instance
bot = telegram.Bot(token='6175278567:AAHq7iR0JmhhkHXtYnRsnAEzXtFhEQ5IbkE')

# Define a function to handle incoming messages
def handle_message(update, context):
    message = update.message
    if message.photo:
        # Download the photo
        photo_file = message.photo[-1].get_file()
        photo_path = 'photo.jpg'
        photo_file.download(photo_path)

        with open('labels.txt', 'r') as f:
            labels = f.read().split('\n')

        # Make a prediction using the CNN model
        image = tf.keras.preprocessing.image.load_img(photo_path, target_size=(224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255
        predictions = model.predict(image_array)
        predicted_label_index = tf.argmax(predictions, axis=1)[0]
        predicted_label_name = labels[predicted_label_index]
        print("Predicted label:", predicted_label_name)
        # You can use the predictions to create some text response
        response_text = f"Predictions: {predicted_label_name}"

        # Send the response back to the user
        message.reply_text(response_text)

# Start the bot and listen for incoming messages
updater = telegram.ext.Updater(token='6175278567:AAHq7iR0JmhhkHXtYnRsnAEzXtFhEQ5IbkE', use_context=True)
updater.dispatcher.add_handler(telegram.ext.MessageHandler(telegram.ext.Filters.all, handle_message))
updater.start_polling()
