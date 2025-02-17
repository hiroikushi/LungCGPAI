import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import pickle
from io import BytesIO

path = os.getcwd()
st.header('Prediction of identifying druggable mutations in lung cancer')
st.subheader('Patient data')

age = st.number_input('Age', min_value=0, max_value=100, value=0, step=1)

sex = st.radio('Sex', ['Woman', 'Man'], horizontal=True)

smokingperday = st.number_input('Cigarettes per day', min_value=0, max_value=200, value=0, step=1)
smokingyears = st.number_input('Years of smoking', min_value=0, max_value=100, value=0, step=1)

asbestos = st.radio('Asbestos exposure', ['No', 'Yes'], horizontal=True)

histology = st.radio('Histology', ['LUAD (lung adenocarcinoma)', 'LUSC (lung squamous cell carcinoma)',
                                   'NSCLC (non-small cell lung cancer), other', 'SCLC (small cell lung cancer)', 
                                   'LCNEC (large cell neuroendocrine carcinoma)', 'Other'])

st.markdown('Metastasis')
lymphmeta = st.toggle('Lymph node', value=False)
lungmeta = st.toggle('Lung', value=False)
pleuralmeta = st.toggle('Pleura', value=False)
livermeta = st.toggle('Liver', value=False)
bonemeta = st.toggle('Bone', value=False)
brainmeta = st.toggle('Brain', value=False)
peritonealmeta = st.toggle('Peritoneum', value=False)
kidneymeta = st.toggle('Kidney', value=False)
adrenalsmeta = st.toggle('Adrenal', value=False)
musclemeta = st.toggle('Muscle', value=False)
softmeta = st.toggle('Soft tissue', value=False)
ovarymeta = st.toggle('Ovary', value=False)

specimentype = st.radio('Specimen (type)', ['primary lesion', 'metastatic lesion', 'peripheral blood'])
specimensite = st.radio('Specimen (site)', ['lung', 'lymph node', 'pleura', 'liver', 'bone', 'brain', 'adrenal', 'other tissue', 'blood'])


st.subheader('Prediction')
st.markdown('Probability (%) that CGP tests will identigy druggable mutations:')
button = st.button('Predict')
if button:
    st.write('Predicting...')
    # Data preparation
    if sex == 'Woman':
        sexinput = 0
    else:
        sexinput = 1
    
    smoking = smokingperday * smokingyears // 20

    if smokingperday * smokingyears == 0:
        whethersmoked = 0
    else:
        whethersmoked = 1
    
    if asbestos == 'No':
        asbestosinput = 0
    else:
        asbestosinput = 1
    
    luad = 1 if histology == 'LUAD (lung adenocarcinoma)' else 0
    lusc = 1 if histology == 'LUSC (lung squamous cell carcinoma)' else 0
    nsclc = 1 if histology == 'NSCLC (non-small cell lung cancer), other' else 0
    sclc = 1 if histology == 'SCLC (small cell lung cancer)' else 0
    lcnec = 1 if histology == 'LCNEC (large cell neuroendocrine carcinoma)' else 0
    other = 1 if histology == 'Other' else 0

    metasite = int(lymphmeta) + int(lungmeta) + int(pleuralmeta) + int(livermeta) + int(bonemeta) + int(brainmeta) + int(peritonealmeta) + int(kidneymeta) + int(adrenalsmeta) + int(musclemeta) + int(softmeta) + int(ovarymeta)

    specimen = 1 if specimentype == 'peripheral blood' else 0
    specimen_site_primary = 1 if specimentype == 'primary lesion' else 0
    specimen_site_metastatic = 1 if specimentype == 'metastatic lesion' else 0
    specimen_site_blood = 1 if specimentype == 'peripheral blood' else 0

    site_lung = 1 if specimensite == 'lung' else 0
    site_lymph = 1 if specimensite == 'lymph node' else 0
    site_pleura = 1 if specimensite == 'pleura' else 0
    site_liver = 1 if specimensite == 'liver' else 0
    site_bone = 1 if specimensite == 'bone' else 0
    site_brain = 1 if specimensite == 'brain' else 0
    site_adrenal = 1 if specimensite == 'adrenal' else 0
    site_other = 1 if specimensite == 'other tissue' else 0
    site_blood = 1 if specimensite == 'blood' else 0

    input = [sexinput, age, whethersmoked, smoking, smokingyears, smokingperday, asbestosinput, specimen, metasite, 
             int(lymphmeta), int(lungmeta), int(livermeta), int(bonemeta), int(brainmeta), int(pleuralmeta), int(peritonealmeta), int(kidneymeta), int(adrenalsmeta), int(musclemeta), int(softmeta), int(ovarymeta), 
             lcnec, luad, lusc, nsclc, other, sclc, specimen_site_metastatic, specimen_site_primary, specimen_site_blood, 
             site_adrenal, site_bone, site_liver, site_lymph, site_other, site_pleura, site_blood, site_brain, site_lung]



    # Prediction
    pred = 0
    fold = 5
    for i in range(fold):
        model = pickle.load(open(f'{path}/lungallmodel/xgb_241202_all_cv_{i}_calib.pkl', 'rb'))
        pred += model.predict_proba([input])[:, 1][0] / fold
    pred *= 100

    if pred < 0.01:
        pred = 0.01
    elif pred > 99.99:
        pred = 99.99

    st.write('Prediction done!')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'**<p style="color:red; font-size: 24px; ">Result: {pred:.2f}%</p>**', unsafe_allow_html=True)

    sizes = [pred, 100 - pred]
    explode = (0.1, 0)
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.pie(sizes, explode=explode, startangle=90, counterclock=False, colors=['limegreen', 'lightgrey'])
    ax.axis('equal')
    
    buf = BytesIO() 
    fig.savefig(buf, format="png")
    with col2:
        st.image(buf)

    st.markdown('This result cannot be used for clinical diagnosis. Please consider performing CGP tests at a physician\'s discretion.')
