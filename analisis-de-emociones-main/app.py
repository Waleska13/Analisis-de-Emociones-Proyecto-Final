import streamlit as st
import joblib
import spacy
import string

st.set_page_config(page_title="Detector de Estado de Ánimo", page_icon="🌿", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Lora&display=swap');
.stApp { background: #f7f2e4 !important; }
html, body, p, label, div, span { font-family: 'Lora', serif !important; color: #2f5233 !important; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #2f5233 !important; }
.titulo { text-align:center; font-family:'Playfair Display',serif; font-size:2.6rem; font-weight:700; color:#2f5233 !important; margin-bottom:0; }
.subtitulo { text-align:center; font-style:italic; color:#5b7a5e !important; margin-top:4px; }
.hoja { position:fixed; font-size:90px; opacity:.35; z-index:0; pointer-events:none; }
.h1 { top:40px; left:20px; transform:rotate(-30deg); }
.h2 { top:40px; right:20px; transform:rotate(30deg) scaleX(-1); }
.h3 { bottom:20px; left:20px; transform:rotate(200deg); }
.h4 { bottom:20px; right:20px; transform:rotate(150deg); }
.tarjeta { background:#fdfcf5; border:1px solid #d9e2c8; border-radius:22px; padding:24px 28px; box-shadow:0 6px 20px rgba(47,82,51,.08); margin:18px 0; }
.stTextArea textarea { background:#fdfcf5 !important; border:1.5px solid #c9d6b5 !important; border-radius:16px !important; color:#2f5233 !important; }
.stButton > button { background:#2f5233 !important; border:none !important; border-radius:30px !important; padding:10px 34px !important; width:100%; }
.stButton > button p { color:#fdfcf5 !important; font-family:'Playfair Display',serif !important; font-size:1.1rem; }
.stButton > button:hover { background:#3f6b45 !important; }
.resultado { text-align:center; font-family:'Playfair Display',serif; font-size:2rem; }
.fila { display:flex; align-items:center; gap:12px; margin:10px 0; }
.etq { width:110px; font-weight:600; }
.barra { flex:1; background:#e6ecd9; border-radius:20px; height:14px; overflow:hidden; }
.relleno { background:linear-gradient(90deg,#7fa66a,#2f5233); height:100%; border-radius:20px; }
.pct { width:50px; text-align:right; }
.separador { text-align:center; color:#7fa66a !important; font-size:1.4rem; margin-top:30px; }
.pie { text-align:center; font-style:italic; color:#5b7a5e !important; }
header, footer { visibility:hidden; }
</style>
<div class="hoja h1">🌿</div><div class="hoja h2">🌿</div>
<div class="hoja h3">🍃</div><div class="hoja h4">🍃</div>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar():
    nlp = spacy.load("es_core_news_sm")
    modelo = joblib.load("LogisticRegression_Balanced.pkl")
    vectorizador = joblib.load("vectorizador.pkl")
    return nlp, modelo, vectorizador

try:
    nlp, modelo, vectorizador = cargar()
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.stop()

def limpiar(texto):
    doc = nlp(texto.lower())
    return " ".join(t.lemma_ for t in doc if not t.is_stop and t.text not in string.punctuation and not t.is_space)

EMOJIS = {"alegría":"😊","tristeza":"😢","enojo":"😠","miedo":"😨","sorpresa":"😲","neutral":"😐","amor":"💚","ansiedad":"😟"}

st.markdown('<p class="titulo">Detector de Estado de Ánimo</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitulo">Escribe cómo te sientes y descubre tu emoción 🌱</p>', unsafe_allow_html=True)

texto = st.text_area("¿Cómo te sientes hoy?", height=140, placeholder="Hoy me siento...")
analizar = st.button("Analizar emoción")

if analizar:
    if not texto.strip():
        st.warning("Por favor escribe algo primero 🌿")
    else:
        X = vectorizador.transform([limpiar(texto)])
        pred = modelo.predict(X)[0]
        emoji = EMOJIS.get(str(pred).lower(), "🌿")
        html = f'<div class="tarjeta"><p class="resultado">{emoji} {pred}</p>'
        if hasattr(modelo, "predict_proba"):
            probs = modelo.predict_proba(X)[0]
            for clase, p in sorted(zip(modelo.classes_, probs), key=lambda x: -x[1]):
                html += (f'<div class="fila"><span class="etq">{clase}</span>'
                         f'<div class="barra"><div class="relleno" style="width:{p*100:.0f}%"></div></div>'
                         f'<span class="pct">{p*100:.0f}%</span></div>')
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

st.markdown('<div class="separador">∼ 🌿 ∼</div>', unsafe_allow_html=True)
st.markdown('<p class="pie">Hecho con 💚 para tu bienestar mental</p>', unsafe_allow_html=True)
