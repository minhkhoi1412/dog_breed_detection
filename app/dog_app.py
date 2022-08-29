import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image


# define ResNet50 model
ResNet50_model = ResNet50(weights="imagenet")
MODEL_FILE = "../saved_models/weights.best.Resnet50.hdf5"
my_model = load_model(MODEL_FILE)

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier(
    '../haarcascades/haarcascade_frontalface_alt.xml')

dog_names = [
    "Affenpinscher",
    "Afghan hound",
    "Airedale terrier",
    "Akita",
    "Alaskan malamute",
    "American eskimo dog",
    "American foxhound",
    "American staffordshire terrier",
    "American water spaniel",
    "Anatolian shepherd dog",
    "Australian cattle dog",
    "Australian shepherd",
    "Australian terrier",
    "Basenji",
    "Basset hound",
    "Beagle",
    "Bearded collie",
    "Beauceron",
    "Bedlington terrier",
    "Belgian malinois",
    "Belgian sheepdog",
    "Belgian tervuren",
    "Bernese mountain dog",
    "Bichon frise",
    "Black and tan coonhound",
    "Black russian terrier",
    "Bloodhound",
    "Bluetick coonhound",
    "Border collie",
    "Border terrier",
    "Borzoi",
    "Boston terrier",
    "Bouvier des flandres",
    "Boxer",
    "Boykin spaniel",
    "Briard",
    "Brittany",
    "Brussels griffon",
    "Bull terrier",
    "Bulldog",
    "Bullmastiff",
    "Cairn terrier",
    "Canaan dog",
    "Cane corso",
    "Cardigan welsh corgi",
    "Cavalier king charles spaniel",
    "Chesapeake bay retriever",
    "Chihuahua",
    "Chinese crested",
    "Chinese shar-pei",
    "Chow chow",
    "Clumber spaniel",
    "Cocker spaniel",
    "Collie",
    "Curly-coated retriever",
    "Dachshund",
    "Dalmatian",
    "Dandie dinmont terrier",
    "Doberman pinscher",
    "Dogue de bordeaux",
    "English cocker spaniel",
    "English setter",
    "English springer spaniel",
    "English toy spaniel",
    "Entlebucher mountain dog",
    "Field spaniel",
    "Finnish spitz",
    "Flat-coated retriever",
    "French bulldog",
    "German pinscher",
    "German shepherd dog",
    "German shorthaired pointer",
    "German wirehaired pointer",
    "Giant schnauzer",
    "Glen of imaal terrier",
    "Golden retriever",
    "Gordon setter",
    "Great dane",
    "Great pyrenees",
    "Greater swiss mountain dog",
    "Greyhound",
    "Havanese",
    "Ibizan hound",
    "Icelandic sheepdog",
    "Irish red and white setter",
    "Irish setter",
    "Irish terrier",
    "Irish water spaniel",
    "Irish wolfhound",
    "Italian greyhound",
    "Japanese chin",
    "Keeshond",
    "Kerry blue terrier",
    "Komondor",
    "Kuvasz",
    "Labrador retriever",
    "Lakeland terrier",
    "Leonberger",
    "Lhasa apso",
    "Lowchen",
    "Maltese",
    "Manchester terrier",
    "Mastiff",
    "Miniature schnauzer",
    "Neapolitan mastiff",
    "Newfoundland",
    "Norfolk terrier",
    "Norwegian buhund",
    "Norwegian elkhound",
    "Norwegian lundehund",
    "Norwich terrier",
    "Nova scotia duck tolling retriever",
    "Old english sheepdog",
    "Otterhound",
    "Papillon",
    "Parson russell terrier",
    "Pekingese",
    "Pembroke welsh corgi",
    "Petit basset griffon vendeen",
    "Pharaoh hound",
    "Plott",
    "Pointer",
    "Pomeranian",
    "Poodle",
    "Portuguese water dog",
    "Saint bernard",
    "Silky terrier",
    "Smooth fox terrier",
    "Tibetan mastiff",
    "Welsh springer spaniel",
    "Wirehaired pointing griffon",
    "Xoloitzcuintli",
    "Yorkshire terrier",
]


def face_detector(img_path):
    """
    Return True if face is detected in image stored at img_path
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def path_to_tensor(img_path):
    """
    Read an image file at img_path and return a numpy array
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


# Dog detector
def ResNet50_predict_labels(img_path):
    """
    Return a prediction vector for image located at img_path
    """
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path):
    """
    Return True if a dog is detected in the image stored at img_path
    """
    prediction = ResNet50_predict_labels(img_path)
    return (prediction <= 268) & (prediction >= 151)


def extract_Resnet50(tensor):
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    return ResNet50(weights='imagenet', include_top=False, 
                    pooling="avg").predict(preprocess_input(tensor))


def Resnet50_predict_breed(img_path):
    """
    Return the predicted dog breed for picture stored at img_path
    """
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = my_model.predict(np.expand_dims(np.expand_dims(bottleneck_feature, axis=0), axis=0))
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def dog_human_detector(img_path):
    """
    Return a string explaining wheteher the image contains a human
    face or dog, and the corresponding breed.
    """
    if face_detector(img_path):
        return "This person resembles a " + Resnet50_predict_breed(img_path)
    elif dog_detector(img_path):
        return "This dog is a " + Resnet50_predict_breed(img_path)
    else:
        return "Neither a dog nor a human face detected in this picture"
