# -*- coding: utf-8 -*-
import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import cv2
import io
import base64
from googletrans import Translator
from gtts import gTTS

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide"
)


# -------- Load your model once --------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plant_disease_model.h5')

model = load_model()

# -------- Class names (must match training order) --------
CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 
    'Soybean___healthy', 
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

# -------- Plant issue to diagnosis & advice mapping --------
plant_issues_dict = {}
for class_name in CLASSES:
    plant = class_name.split('___')[0]
    condition = class_name.split('___')[1].replace('_', ' ')
    
    if 'healthy' in class_name:
        plant_issues_dict[class_name] = ("Healthy", "No action needed. Your plant looks healthy!")
    else:
        plant_issues_dict[class_name] = (
            condition,
            f"Apply appropriate treatment for {condition}. Remove affected leaves, apply recommended fungicide/pesticide, and maintain proper growing conditions."
        )

# -------- Supported languages for translation --------
languages = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "Arabic": "ar"
}

# -------- Language dictionaries for UI texts --------
ui_texts = {
    "English": {
        "menu_title": "🌿Plant Doctor Menu",
        "welcome_title": "Welcome to PlantiFi",
        "welcome_subtitle": "AI-powered plant disease detection with detailed analysis and treatment advice",
        "ai_assistant_title": "Plant Disease Analysis",
        "name_label": "Name",
        "language_label": "Language",
        "weather_label": "Weather Condition",
        "soil_label": "Soil Type",
        "upload_label": "Upload Plant Leaf Image",
        "diagnosis_advice_title": "🧠 Diagnosis & Treatment Advice",
        "user_input_summary": "📋 Analysis Summary",
        "download_csv": "📥 Download Summary as CSV",
        "info_no_upload": "Please upload a plant leaf image to get diagnosis and advice.",
        "developed_by": "👨‍🔬 Developed by Team Green Rebels",
        "crop_calendar_title": "Crop Planting Calendar",
        "crop_stats_title": "Agriculture Insights",
        "prevention_title": "Disease Prevention Tips",
        "weather_options": ["Select", "Sunny", "Rainy", "Cloudy", "Windy", "Humid"],
        "soil_options": ["Select", "Loam", "Clay", "Sandy", "Silt", "Peat", "Chalky"],
        "treatment_title": "💊 Recommended Treatment",
        "prevention_tips": "🛡️ Prevention Strategies",
        "water_advice": "💧 Watering Advice"
    },
    "Hindi": {
        "menu_title": "🌿 प्लांट डॉक्टर मेनू",
        "welcome_title": "🌱प्लांटीफाई में आपका स्वागत है",
        "welcome_subtitle": "एआई-संचालित प्लांट रोग पहचान विस्तृत विश्लेषण और उपचार सलाह के साथ",
        "ai_assistant_title": "🔍 प्लांट रोग विश्लेषण",
        "name_label": "नाम",
        "language_label": "भाषा",
        "weather_label": "मौसम की स्थिति",
        "soil_label": "मिट्टी का प्रकार",
        "upload_label": "पौधे की पत्ती की छवि अपलोड करें",
        "diagnosis_advice_title": "🧠 निदान और उपचार सलाह",
        "user_input_summary": "📋 विश्लेषण सारांश",
        "download_csv": "📥 सारांश सीएसवी के रूप में डाउनलोड करें",
        "info_no_upload": "निदान और सलाह प्राप्त करने के लिए कृपया पौधे की पत्ती की छवि अपलोड करें।",
        "developed_by": "👨‍🔬  टीम ग्रीन रेबेल्स द्वारा विकसित",
        "crop_calendar_title": "📆 फसल रोपण कैलेंडर",
        "crop_stats_title": "📊 कृषि जानकारी",
        "prevention_title": "🌿 रोग निवारण युक्तियाँ",
        "weather_options": ["चुनें", "धूप", "बारिश", "बादल", "हवा", "आर्द्र"],
        "soil_options": ["चुनें", "दोमट मिट्टी", "चिकनी मिट्टी", "बलुई मिट्टी", "गाद मिट्टी", "पीट मिट्टी", "चूना मिट्टी"],
        "treatment_title": "💊 अनुशंसित उपचार",
        "prevention_tips": "🛡️ निवारण रणनीतियाँ",
        "water_advice": "💧 सिंचाई सलाह"
    }
}

# ----------- Background and Styling ------------
background_url = "https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80"

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85)), url({background_url});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
        background-color: rgba(0, 0, 0, 0.5);
    }}
    [data-testid="stSidebar"] {{
        background-color: #1a472a;
        color: white;
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    h1, h2, h3, h4, h5, p, label, div {{
        color: white !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al {{
        background-color: rgba(26, 71, 42, 0.8) !important;
    }}
    a {{
        color: #90ee90;
    }}
    .stButton>button {{
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
    }}
    .stButton>button:hover {{
        background-color: #4caf50;
        color: white;
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
    }}
    .css-1cpxqw2, .css-1hwfws3, .css-14xtw13, .css-1g6gooi {{
        color: black !important;
    }}
    .success-box {{
        background-color: #2e7d32;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }}
    .warning-box {{
        background-color: #ff9800;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }}
    .info-box {{
        background-color: #0288d1;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }}
    .metric-box {{
        background-color: rgba(26, 71, 42, 0.7);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4caf50;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------- Language selection in sidebar ---------
with st.sidebar:
    selected_lang = st.selectbox("🌐 Select Language / भाषा चुनें", options=list(ui_texts.keys()), index=0)

texts = ui_texts.get(selected_lang, ui_texts["English"])  # fallback to English

# --------- Sidebar menu ---------
with st.sidebar:
    selected_menu = option_menu(
        menu_title=texts["menu_title"],
        options=[
            texts["welcome_title"],
            texts["ai_assistant_title"],
            texts["crop_calendar_title"],
            texts["crop_stats_title"],
            texts["prevention_title"]
        ],
        icons=["house", "search", "calendar", "bar-chart", "shield"],
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#1a472a"},
            "icon": {"color": "#90ee90", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px 0",
                "color": "white",
                "font-weight": "bold",
            },
            "nav-link-selected": {"background-color": "#2e7d32", "color": "white"},
        },
    )
    st.markdown("---")
    st.markdown(
        f"<p style='text-align:center;color:#90ee90;font-weight:bold;'>{texts['developed_by']}</p>",
        unsafe_allow_html=True,
    )

# --------- Welcome Page ---------
if selected_menu == texts["welcome_title"]:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            f"""
            <div style="height: 60vh; display:flex; flex-direction:column; justify-content:center;">
                <h1 style="font-size:3.5rem; margin-top: 5.2rem;">{texts['welcome_title']}</h1>
                <p style="font-size:1.5rem; max-width:700px;">{texts['welcome_subtitle']}</p>
                <br>
                <p style="font-size:1.1rem;">🌾 <b>Features:</b></p>
                <ul>
                    <li>AI-powered plant disease identification</li>
                    <li>Detailed treatment and prevention advice</li>
                    <li>Multi-language support</li>
                    <li>Crop planning calendar</li>
                    <li>Agricultural insights</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.image("Welcome_Image.png", 
                 caption="AI Plant Disease Detection")

# --------- AI Assistant Page ---------
elif selected_menu == texts["ai_assistant_title"]:
    st.markdown(f"<h1 style='text-align:center; color:#90ee90;'>{texts['ai_assistant_title']}</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2], gap="large")

    # 👤 Input Section
    with col1:
        name = st.text_input(texts["name_label"])
        weather = st.selectbox(texts["weather_label"], texts["weather_options"])
        soil = st.selectbox(texts["soil_label"], texts["soil_options"])

        st.markdown("### 📷 Upload Leaf Image")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    # 🧠 Prediction & Advice
    with col2:
        if uploaded_file:
            def preprocess_image(image):
                img = np.array(image)
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (224, 224))
                img = img / 255.0
                return np.expand_dims(img, axis=0)

            try:
                processed_img = preprocess_image(image)
                prediction = model.predict(processed_img)
                confidence = np.max(prediction) * 100
                class_index = np.argmax(prediction)
                disease_class = CLASSES[class_index]
                disease_name, treatment = plant_issues_dict[disease_class]

                def info_box(title, content, icon="🔍"):
                    st.markdown(f"""
                    <div style='background-color:#1a472a;padding:12px;border-left:5px solid #4caf50;border-radius:8px;margin-bottom:15px;'>
                    <h4 style='color:#90ee90;'>{icon} {title}</h4>
                    <p style='color:white;'>{content}</p>
                    </div>
                    """, unsafe_allow_html=True)

                if "healthy" in disease_class:
                    info_box("✅ Plant Status", "Healthy Plant", "🌿")
                else:
                    info_box("🦠 Disease Detected", f"{disease_name} ({confidence:.2f}%)")

                # Watering Advice
                watering = (
                    "Reduce watering frequency as rain provides sufficient moisture." if weather in ["Rainy", "बारिश"]
                    else "Water early morning or late evening to prevent evaporation." if weather in ["Sunny", "धूप"]
                    else "Maintain regular watering schedule based on plant needs."
                )
                info_box("💧 Watering Advice", watering)

                # Treatment Advice
                info_box("🧪 Recommended Treatment", treatment)

                # Top Predictions
                st.markdown("### 📊 Top Predictions")
                sorted_indices = np.argsort(prediction[0])[::-1][:5]
                for idx in sorted_indices:
                    prob = prediction[0][idx] * 100
                    readable_name = CLASSES[idx].split("___")[1].replace("_", " ")
                    if "healthy" in CLASSES[idx]:
                        readable_name = "Healthy"

                    st.markdown(f"""
                    <div style="margin: 10px 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>{readable_name}</span>
                            <span>{prob:.1f}%</span>
                        </div>
                        <div style="background: #555; border-radius: 5px; height: 20px;">
                            <div style="background: #4caf50; width: {int(prob)}%; height: 100%; border-radius: 5px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Translation
                translator = Translator()
                try:
                    translated = translator.translate(treatment, dest=languages[selected_lang]).text
                except:
                    translated = treatment

                st.markdown(f"#### 🌐 Translated Advice ({selected_lang})")
                st.info(translated)

                # Audio
                tts = gTTS(translated, lang=languages[selected_lang])
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                st.audio(buf.read(), format="audio/mp3")

                # Download summary
                summary_csv = "Name,Language,Weather,Soil,Condition,Confidence,Treatment\n"
                summary_csv += f"{name},{selected_lang},{weather},{soil},{disease_name},{confidence:.2f}%,{treatment}\n"
                b64 = base64.b64encode(summary_csv.encode()).decode()
                st.markdown(f'<a href="data:file/csv;base64,{b64}" download="plant_diagnosis_summary.csv"> {texts["download_csv"]}</a>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

        else:
            st.info(texts["info_no_upload"])
            st.image("banner-2.jpg", caption="Upload an image for analysis", use_container_width=True)

# --------- Crop Planting Calendar Page ---------
elif selected_menu == texts["crop_calendar_title"]:
    st.markdown(f"<h1 style='text-align:center; color:#90ee90;'>{texts['crop_calendar_title']}</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color:#1a472a;padding:15px;border-radius:10px;margin-bottom:20px;'>
    <h4 style='color:#90ee90;'>📅 Seasonal Crop Recommendations</h4>
    <p style='color:white;'>Plan your planting schedule based on India's seasonal cycles. Here's a quick guide:</p>
    </div>
    """, unsafe_allow_html=True)

    crop_data = {
        "Kharif (June–Oct)": ["Rice", "Maize", "Soybean", "Cotton", "Groundnut", "Bottle Gourd"],
        "Rabi (Oct–March)": ["Wheat", "Barley", "Mustard", "Peas", "Carrot", "Spinach"],
        "Zaid (March–June)": ["Watermelon", "Cucumber", "Pumpkin", "Fodder Crops"]
    }

    for season, crops in crop_data.items():
        with st.expander(f"🌾 {season} Crops"):
            for crop in crops:
                st.markdown(f"- {crop}")

    # ---------- Live Chart Section ----------
    import pandas as pd
    import plotly.express as px

    st.markdown("""
    <div style='background-color:#004d40;padding:15px;border-radius:10px;margin-top:20px;'>
    <h4 style='color:#90ee90;'>📊 Seasonal Crop Distribution Chart</h4>
    <p style='color:white;'>Visualize crop categories across India's seasonal cycles.</p>
    </div>
    """, unsafe_allow_html=True)

    # Prepare data for chart
    crop_chart_data = pd.DataFrame({
        "Season": ["Kharif"] * 6 + ["Rabi"] * 6 + ["Zaid"] * 4,
        "Crop": ["Rice", "Maize", "Soybean", "Cotton", "Groundnut", "Bottle Gourd",
                 "Wheat", "Barley", "Mustard", "Peas", "Carrot", "Spinach",
                 "Watermelon", "Cucumber", "Pumpkin", "Fodder Crops"]
    })

    fig = px.histogram(crop_chart_data, x="Season", color="Crop",
                       title="Seasonal Crop Distribution",
                       labels={"Crop": "Crop Type", "Season": "Season"},
                       barmode="group")

    st.plotly_chart(fig, use_container_width=True)
    
    # --------- Agriculture Insights Page ---------
elif selected_menu == texts["crop_stats_title"]:
    st.markdown(f"<h1 style='text-align:center; color:#90ee90;'>{texts['crop_stats_title']}</h1>", unsafe_allow_html=True)

    # 🌾 Key Trends
    st.markdown("""
    <div style='background-color:#1a472a;padding:15px;border-radius:10px;margin-bottom:20px;'>
    <h4 style='color:#90ee90;'>📊 Key Trends in Indian Agriculture</h4>
    <ul style='color:white;'>
        <li>🌱 Shift toward organic and sustainable farming</li>
        <li>📈 Rise in precision agriculture using IoT and satellite data</li>
        <li>🚜 Mechanization and smart equipment adoption</li>
        <li>🛰️ Use of remote sensing for crop health and yield prediction</li>
        <li>📦 Growth of direct-to-consumer agri-markets</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # 📈 Precision Farming Table
    st.markdown("""
    <div style='background-color:#0288d1;padding:15px;border-radius:10px;margin-bottom:20px;'>
    <h4 style='color:white;'>📈 Precision Farming Impact (2025)</h4>
    </div>
    """, unsafe_allow_html=True)

    impact_table = pd.DataFrame({
        "Technology": ["Satellite Monitoring", "Soil Sensors", "AI Analytics", "Drones", "Blockchain"],
        "Adoption Rate": ["73%", "65%", "68%", "55%", "47%"],
        "Yield Boost": ["11–15%", "8–12%", "10–14%", "7–10%", "2–6%"],
        "Sustainability Benefit": [
            "Reduced water & chemical use",
            "Efficient nutrient management",
            "Lower emissions",
            "Targeted pesticide use",
            "Supply chain transparency"
        ]
    })
    st.dataframe(impact_table)

    # 🌍 Climate & Market Insights
    st.markdown("""
    <div style='background-color:#6a1b9a;padding:15px;border-radius:10px;margin-top:20px;'>
    <h4 style='color:white;'>🌍 Climate & Market Insights</h4>
    <ul style='color:white;'>
        <li>🌦️ Climate volatility is reshaping sowing windows</li>
        <li>📉 Market prices fluctuate due to global supply chain shifts</li>
        <li>🧠 Predictive analytics help farmers plan better</li>
        <li>📲 Mobile platforms enable real-time mandi price tracking</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # 📈 Live Market Price Charts
    st.markdown("""
    <div style='background-color:#0288d1;padding:15px;border-radius:10px;margin-top:20px;'>
    <h4 style='color:white;'>📈 Live Market Price Charts</h4>
    <p style='color:white;'>Explore real-time mandi prices for major crops across India. Data sourced from Agmarknet and e-NAM platforms.</p>
    </div>
    """, unsafe_allow_html=True)

    agmarknet_url = "https://agmarknet.ceda.ashoka.edu.in/"
    st.components.v1.iframe(agmarknet_url, height=600, scrolling=True)

    st.markdown("""
    <p style='color:white;'>🔗 For detailed commodity-wise prices, visit the 
    <a href='https://www.enam.gov.in/web/dashboard/live_price' target='_blank' style='color:#90ee90;'>
    e-NAM Live Price Dashboard</a>.</p>
    """, unsafe_allow_html=True)


 # --------- Prevention Tips Page ---------
elif selected_menu == texts["prevention_title"]:
    st.markdown(f"<h1 style='text-align:center; color:#90ee90;'>{texts['prevention_title']}</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2], gap="large")

    # 📸 Seasonal checklist + image
    with col1:
        st.image(
            "C:/Users/Florence/Downloads/farmer_new/farmer_new/Ai_Green_Rebels.png",
            caption="Organic Farming Practices",
            use_container_width=True
        )

        st.markdown("""
        <div style='background-color:#1a472a;padding:15px;border-radius:10px;margin-top:20px;'>
        <h4 style='color:#90ee90;'>🕒 Seasonal Checklist</h4>
        """, unsafe_allow_html=True)

        seasons = {
            "🌸 Spring": ["Prepare soil with compost", "Start seedlings indoors", "Prune fruit trees"],
            "☀️ Summer": ["Monitor for pests", "Water deeply during dry spells", "Apply mulch to retain moisture"],
            "🍂 Fall": ["Harvest remaining crops", "Plant cover crops", "Clean and store tools"],
            "❄️ Winter": ["Plan next season's garden", "Order seeds", "Maintain equipment"]
        }
        for season, tasks in seasons.items():
            with st.expander(season):
                for task in tasks:
                    st.markdown(f"- {task}")
        st.markdown("</div>", unsafe_allow_html=True)

    # 🌱 Prevention tips styled in boxes
    with col2:
        def styled_box(title, color, icon, tips):
            st.markdown(f"""
            <div style='background-color:{color};padding:15px;border-radius:10px;margin-bottom:20px;'>
            <h4 style='color:white;'>{icon} {title}</h4>
            """, unsafe_allow_html=True)
            for tip in tips:
                st.markdown(f"- {tip}")
            st.markdown("</div>", unsafe_allow_html=True)

        styled_box("General Prevention", "#2e7d32", "🛡️", [
            "🌱 Rotate crops annually to prevent soil-borne diseases",
            "💧 Water at the base of plants to keep foliage dry",
            "🌞 Ensure adequate spacing for air circulation",
            "🧤 Remove infected plant material promptly",
            "🪲 Use row covers to block pests"
        ])

        styled_box("Soil Management", "#0288d1", "🧪", [
            "🌿 Add compost to enrich soil",
            "🔍 Test soil pH and adjust as needed",
            "🔄 Use cover crops to prevent erosion",
            "🚫 Avoid compacting wet soil"
        ])

        styled_box("Organic Treatments", "#6a1b9a", "🍃", [
            "🧄 Garlic & chili spray for insects",
            "🌿 Neem oil as a natural pesticide",
            "🦠 Bt for caterpillar control",
            "🧪 Baking soda for fungal prevention (1 tbsp/gallon)"
        ])
