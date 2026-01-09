import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import tempfile
import time
import requests
import os
import urllib3


# Disable warnings from requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Recommendation data
RECOMMENDATIONS = {
    'CNV': {
        "English": {
            "title": "CNV (Choroidal Neovascularization)",
            "summary": "👁️ Oh! Some new blood vessels forming in your retina.",
            "points": ["🩺 Visit a retina specialist promptly",
                       "💉 Anti-VEGF injections, PDT, or laser therapy may help",
                       "🥗 Eat leafy greens, omega-3, AREDS2 supplements",
                       "⏱ Regular OCT monitoring advised"],
            "color": "#FF4C4C"
        },
        "Tamil": {
            "title": "CNV (கோரோயிடல் நியோவாஸ்குலரைசேஷன்)",
            "summary": "👁️ ரெட்டினாவில் புதிய இரத்த நாளங்கள் உருவாகும் போல் தெரிகிறது.",
            "points": ["🩺 ரெட்டினா நிபுணரை உடனடியாக சந்திக்கவும்",
                       "💉 Anti-VEGF ஊசிகள், PDT, லேசர் சிகிச்சை",
                       "🥗 கீரைகள், ஓமேகா-3, AREDS2 சப்பிளிமெண்ட்கள்",
                       "⏱ OCT பரிசோதனைகள் வழக்கமாக செய்யவும்"],
            "color": "#FF4C4C"
        }
    },
    'DME': {
        "English": {
            "title": "DME (Diabetic Macular Edema)",
            "summary": "👁️ Slight swelling in central retina.",
            "points": ["🩺 Joint care with eye doctor & diabetes specialist",
                       "💉 Eye injections or laser may help",
                       "📊 Maintain blood sugar & blood pressure",
                       "⏱ Regular check-ups every 3–6 months"],
            "color": "#FF9F40"
        },
        "Tamil": {
            "title": "DME (நீரிழிவு மாகுலரிங் வீக்கம்)",
            "summary": "👁️ மாகுலா பகுதியில் சிறிது வீக்கம் உள்ளது.",
            "points": ["🩺 கண் நிபுணர் மற்றும் நீரிழிவு நிபுணரை இணைந்து மேலாண்மை",
                       "💉 கண் ஊசிகள் அல்லது லேசர் வீக்கத்தை குறைக்க உதவும்",
                       "📊 சர்க்கரை மற்றும் இரத்த அழுத்தத்தை கட்டுப்படுத்தவும்",
                       "⏱ 3–6 மாதங்களுக்கு ஒருமுறை பரிசோதனை"],
            "color": "#FF9F40"
        }
    },
    'DRUSEN': {
        "English": {
            "title": "Drusen (Early AMD)",
            "summary": "👁️ Tiny deposits visible in retina.",
            "points": ["🥗 Eat antioxidant-rich foods & fish",
                       "🚭 Avoid smoking & wear UV sunglasses",
                       "⏱ OCT every 6–12 months",
                       "💡 Self-check with Amsler grid at home"],
            "color": "#FFD700"
        },
        "Tamil": {
            "title": "Drusen (ஆரம்ப AMD)",
            "summary": "👁️ ரெட்டினாவில் சில சிறிய கலப்படங்கள் இருக்கின்றன.",
            "points": ["🥗 ஆன்டிஆக்ஸிடென்ட்ஸ் நிறைந்த உணவுகள், மீன்",
                       "🚭 புகைபிடிப்பதை தவிர்க்கவும், UV கண்ணாடி அணியவும்",
                       "⏱ 6–12 மாதங்களுக்கு ஒருமுறை OCT",
                       "💡 வீட்டில் Amsler grid மூலம் சுய பரிசோதனை"],
            "color": "#FFD700"
        }
    },
    'NORMAL': {
        "English": {
            "title": "Normal Retina",
            "summary": "👁️ Retina looks healthy!",
            "points": ["🩺 Routine eye check-ups every 1–2 years",
                       "🥗 Maintain balanced diet & active lifestyle",
                       "🕶 Protect eyes with UV sunglasses",
                       "💡 Monitor general health including blood pressure & diabetes"],
            "color": "#4CAF50"
        },
        "Tamil": {
            "title": "சாதாரண ரெட்டினா",
            "summary": "👁️ உங்கள் ரெட்டினா ஆரோக்கியமாக தெரிகிறது.",
            "points": ["🩺 1–2 ஆண்டுகளுக்கு ஒருமுறை கண் பரிசோதனை",
                       "🥗 சமநிலை உணவு மற்றும் உடற்பயிற்சி",
                       "🕶 UV கண்ணாடி அணியவும்",
                       "💡 நீரிழிவு மற்றும் இரத்த அழுத்தத்தை கவனிக்கவும்"],
            "color": "#4CAF50"
        }
    }
}

CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Initialize temporary file path variable
temp_file_path = None

# Streamlit app configuration
st.set_page_config(page_title="Professional OCT Retinal Analysis", page_icon="🧿", layout="wide")

# Sidebar: Language and Page selection
with st.sidebar:
    st.markdown("### 🌐 Language / மொழி தேர்வு")
    lang_selection = st.selectbox("", ["English 🌐", "தமிழ் 🇮🇳"])
    lang = "English" if "English" in lang_selection else "Tamil"

st.sidebar.title("🧿 OCT Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page / பக்கத்தைத் தேர்ந்தெடுக்கவும்",
    ["Home", "About", "Disease Identification"]
)

# Load Model Function
@st.cache_resource
def load_model():
    try:
        # NOTE: The model path "Trained_Model.keras" is assumed to be correct.
        return tf.keras.models.load_model("Trained_Model.keras", compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure 'Trained_Model.keras' is in the current directory.")
        return None

# Prediction function
def model_prediction(test_image_path):
    model = load_model()
    if model is None:
        return 0, 0.0
    # Load and preprocess image
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Predict
    predictions = model.predict(x, verbose=0)
    result_index = np.argmax(predictions)
    confidence = float(np.max(predictions) * 100)
    return result_index, confidence

# Get nearby hospitals
def get_nearby_hospitals(city_name):
    url = f"https://nominatim.openstreetmap.org/search?format=json&q=eye+hospital+in+{city_name}&limit=10"
    # IMPORTANT: Add an appropriate User-Agent header for the Nominatim API
    response = requests.get(url, headers={"User-Agent": "OCT_Retinal_Analysis_App/1.0"})
    hospitals = []
    if response.status_code == 200:
        results = response.json()
        for place in results:
            if 'lat' in place and 'lon' in place:
                name = place.get("display_name", "Unknown Hospital")
                lat = float(place["lat"])
                lon = float(place["lon"])
                maps_url = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=16/{lat}/{lon}"
                hospitals.append({"name": name, "lat": lat, "lon": lon, "maps_url": maps_url})
    return hospitals



# Main page logic
if app_mode == "Home":
    st.header("🧿 OCT Retinal Analysis Platform")
    if lang == "English":
        st.markdown("""
### Welcome to the Professional OCT Retinal Analysis Platform 👁️

Detect, analyze, and understand retinal diseases using **AI-powered OCT classification**.

**Features:**
- Non-invasive, high-precision retina analysis     
- Disease prediction: CNV, DME, Early AMD, Normal     
- Dual-language support: English & Tamil     
- Confidence-based predictions with professional recommendations

---

**OCT detects:**     
- 🟥 CNV (Choroidal Neovascularization)     
- 🟧 DME (Diabetic Macular Edema)     
- 🟨 Drusen (Early AMD)     
- 🟩 Normal Retina

📤 Upload scans ➡️ ⚡ Get instant predictions ➡️ 🔎 Explore insights
""")
    else:
        st.markdown("""
### கண் OCT பகுப்பாய்வு தளம் 👁️

**OCT** கண்களின் ரெட்டினா படங்களை AI மூலம் துல்லியமாக பகுப்பாய்வு செய்ய உதவுகிறது.

**அம்சங்கள்:**
- வலி இல்லாத, துல்லியமான ரெட்டினா பகுப்பாய்வு     
- நோய் கணிப்பு: CNV, DME, ஆரம்ப AMD, இயல்பு     
- இருமொழி ஆதரவு: தமிழ் & ஆங்கிலம்     
- நம்பகமான முடிவுகள் மற்றும் பரிந்துரைகள்

---

OCT மூலம் கண்டறியப்படும் நிலைகள்:     
- 🟥 CNV (புதிய இரத்த நாள வளர்ச்சி)     
- 🟧 DME (நீர் gerekiை கண் வீக்கம்)     
- 🟨 Drusen (ஆரம்ப AMD)     
- 🟩 இயல்பு ரெட்டினா

📤 ஸ்கேன் பதிவேற்றவும் ➡️ ⚡ உடனடி முடிவுகள் ➡️ 🔎 விரிவான தகவல் பெறவும்
""")
elif app_mode == "About":
    if lang == "English":
        st.header("📘 About Dataset & Project")
        st.markdown("""
#### Dataset Overview
Retinal OCT captures retina cross-sections using light waves. Widely used to detect CNV, DME, and AMD.     

**Dataset contains:** 84,495 images, categorized into CNV, DME, DRUSEN, NORMAL.     
All images verified by ophthalmologists for clinical accuracy.

#### Project Goal
AI-driven TensorFlow model to classify retinal diseases automatically, providing fast, reliable support for ophthalmologists.
""")
    else:
        st.header("📘 திட்டம் மற்றும் தரவுத்தொகுப்பு")
        st.markdown("""
#### தரவுத்தொகுப்பு
ரெட்டினா OCT ஒளி அலைகளைப் பயன்படுத்தி படங்களை பதிவு செய்கிறது. CNV, DME, மற்றும் AMD கண்டறிய பயன்படும்.     

**தரவுத்தொகுப்பு:** 84,495 படங்கள் — CNV, DME, DRUSEN, NORMAL.     
ஒவ்வொரு படமும் கண் நிபுணர்களால் சரிபார்க்கப்பட்டது.

#### திட்ட நோக்கம்
AI அடிப்படையிலான TensorFlow மாடல் ரெட்டினா நோய்களை தானாக வகைப்படுத்துகிறது, மருத்துவர்களுக்கு வேகமான மற்றும் நம்பகமான உதவி.
""")
elif app_mode == "Disease Identification":
    st.header("🔍 " + ("Disease Identification" if lang == "English" else "நோய் கண்டறிதல்"))

    upload_text = "📤 Upload your OCT Image:" if lang == "English" else "📤 உங்கள் OCT படத்தை பதிவேற்றவும்:"
    predict_button = "⚡ Predict" if lang == "English" else "⚡ கணிப்பு"
    wait_text = "🔎 Analyzing image... please wait" if lang == "English" else "🔎 படம் பரிசோதிக்கப்படுகிறது... காத்திருங்கள்"
    success_text = "✅ Prediction" if lang == "English" else "✅ முடிவு"

    # Initialize session state variables for persistence
    if "prediction_made" not in st.session_state:
        st.session_state["prediction_made"] = False
    if "hospitals" not in st.session_state:
        st.session_state["hospitals"] = []
    if "last_searched_city" not in st.session_state:
        st.session_state["last_searched_city"] = ""
    if "temp_file_path" not in st.session_state:
        st.session_state["temp_file_path"] = None

    test_image = st.file_uploader(upload_text, type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, caption="Uploaded OCT Scan", width=400)
        with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as tmp_file:
            tmp_file.write(test_image.read())
            st.session_state["temp_file_path"] = tmp_file.name

    temp_file_path = st.session_state.get("temp_file_path")

    if st.button(predict_button) and temp_file_path is not None:
        with st.spinner(wait_text):
            progress_bar = st.progress(0)
            for perc in range(100):
                time.sleep(0.01)
                progress_bar.progress(perc + 1)
            result_index, confidence = model_prediction(temp_file_path)
            predicted_class = CLASS_NAMES[result_index]

        st.success(f"{success_text}: **{predicted_class}** ({confidence:.2f}% confidence)")
        st.session_state["prediction_made"] = True

        # Show recommendations
        col1, col2 = st.columns([1, 2])
        with col2:
            disease_dict = RECOMMENDATIONS[predicted_class]
            st.subheader(f"🩺 {disease_dict[lang]['title']}")
            st.markdown(f"<p style='color:{disease_dict[lang]['color']};font-weight:bold'>{disease_dict[lang]['summary']}</p>", unsafe_allow_html=True)
            for point in disease_dict[lang]['points']:
                st.markdown(f"- {point}")

    # Nearby hospitals
    st.markdown("---")
    st.subheader("🏥 Nearby Eye Hospitals")

    with st.form(key="hospital_search_form"):
        city = st.text_input(
            "Enter your city to find nearby hospitals:" if lang == "English" else "பார்ப்பதற்கான நகரத்தை உள்ளிடவும்:",
            key="city_input",
            value=st.session_state["last_searched_city"]
        )
        submitted = st.form_submit_button("🔎 Search Hospitals")

    if submitted:
        city_to_search = city.strip()
        st.session_state["last_searched_city"] = city_to_search

        if not city_to_search:
            st.warning("Please enter a city name." if lang == "English" else "நகரத்தின் பெயரை உள்ளிடவும்.")
            st.session_state["hospitals"] = []
            st.session_state["hospital_center"] = None
        else:
            with st.spinner("Searching for hospitals..."):
                hospitals = get_nearby_hospitals(city_to_search)
                st.session_state["hospitals"] = hospitals if hospitals else []

                if st.session_state["hospitals"]:
                    first_hospital = st.session_state["hospitals"][0]
                    st.session_state["hospital_center"] = (first_hospital["lat"], first_hospital["lon"])
                    st.success(f"Found {len(hospitals)} hospitals near {city_to_search}." if lang == "English" else f"{city_to_search} அருகில் {len(hospitals)} மருத்துவமனைகள் கண்டுபிடிக்கப்பட்டன.")
                else:
                    st.session_state["hospital_center"] = None
                    st.warning("No eye hospitals found in that area." if lang == "English" else "அந்த பகுதியில் கண் மருத்துவமனை இல்லை.")

    
# Get hospital list safely from session state
hospitals_list = st.session_state.get("hospitals", [])
submitted = st.session_state.get("hospital_submitted", False)

if hospitals_list:
    st.subheader("🏥 List of Hospitals")
    for h in hospitals_list:
        st.markdown(f"- **{h['name']}** | [View Directions]({h['maps_url']})")

elif not submitted and not st.session_state.get("prediction_made"):
    st.info(
        "Enter a city and click search to find nearby eye hospitals."
        if lang == "English"
        else "நகரத்தை உள்ளிடவும் மற்றும் தேடல் பொத்தானை அழுத்தவும்."
    )


# Footer
st.markdown("---")
st.markdown("🔬 Built with ❤️ using **TensorFlow** & **Streamlit** | Professional OCT Retinal Analysis")
