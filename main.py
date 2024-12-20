from tensorflow.keras.models import load_model # type: ignore

model = load_model('Models\\model.keras')
model.summary()