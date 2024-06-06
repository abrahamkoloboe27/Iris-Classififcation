import streamlit as st 
import joblib
import pandas as pd
import plotly.express as px

st.header("Iris Classification", divider="rainbow")
st.image("images/iris_photo.jpg", use_column_width=True)
with st.sidebar : 
        st.markdown("""
        ## Auteur
        :blue[Abraham KOLOBOE]
        * Email : <abklb27@gmail.com>
        * WhatsApp : +229 91 83 84 21
        * Linkedin : [Abraham KOLOBOE](https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning)
                    """)
if st.sidebar.toggle("Description") : 
    st.write("Cette application Streamlit permet de classer les iris en utilisant un modèle pré-entraîné.")
    st.markdown("""
    Le jeu de données sur les fleurs d'iris, également appelé jeu de données sur les iris de Fisher, 
                est un ensemble de données multivariées introduit par le statisticien, eugéniste et 
                biologiste britannique Ronald Fisher dans son article de 1936 intitulé "The use of multiple 
                measurements in taxonomic problems" comme un exemple d'analyse discriminante linéaire. 
                Parfois appelé jeu de données sur les iris d'Anderson car Edgar Anderson a collecté les 
                données pour quantifier la variation morphologique des fleurs d'iris de trois espèces apparentées. 
                Deux des trois espèces ont été collectées dans la péninsule de Gaspé, "toutes dans le même pâturage, 
                cueillies le même jour et mesurées en même temps par la même personne avec le même appareil". 
                L'article de Fisher a été publié dans la revue "Annals of Eugenics", suscitant une controverse sur 
                l'utilisation continue du jeu de données sur les iris dans l'enseignement des techniques statistiques 
                aujourd'hui.

    L'ensemble de données se compose de 50 échantillons de chacune des trois espèces d'iris (Iris setosa, 
                Iris virginica et Iris versicolor). Quatre caractéristiques ont été mesurées pour chaque échantillon : 
                la longueur et la largeur des sépales et des pétales, en centimètres. En se basant sur la combinaison de
                 ces quatre caractéristiques, Fisher a développé un modèle discriminant linéaire pour distinguer les
                 espèces les unes des autres.
                """)
    st.write("Le modèle a été entraîné sur le dataset Iris, qui contient des mesures de différentes caractéristiques des iris.")
    st.write("Le dataset comprend les colonnes suivantes : SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species.")

model = joblib.load('model/Iris_model.pkl')
col_1, col_2 =  st.columns(2, gap="medium")
with col_1 : 
    with st.form("Caractéristiques") : 
        SepalLengthCm = st.number_input("Sepal Length in Cm", value=5.843333, min_value=4.3, max_value= 8.0)
        SepalWidthCm  = st.number_input("Sepal Width in Cm",value=3.054000, min_value=2.0, max_value= 4.5)
        PetalLengthCm  = st.number_input("Petal Length in Cm", value=3.758667, min_value=1.0, max_value= 7.0)
        PetalWidthCm  = st.number_input("Petal Width in Cm",value=1.198667, min_value=0.1, max_value= 2.6)
        submit = st.form_submit_button("Prédire ! ")

if submit :    
    dic = {
        'SepalLengthCm': [SepalLengthCm], 
        'SepalWidthCm': [SepalWidthCm], 
        'PetalLengthCm': [PetalLengthCm], 
        'PetalWidthCm': [PetalWidthCm]
    }
    pd_dic = pd.DataFrame(dic)

    try:
        predicted_class = model.predict(pd_dic)[0]
        proba_predict = model.predict_proba(pd_dic)
        fig = px.bar(x=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], 
                     y=proba_predict[0]*100, labels={'x':'Classe', 'y':'Probabilité'}, 
                     title='Probabilité de chaque classe', color=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
        with col_2 : 
            st.plotly_chart(fig, use_container_width=True)
        st.success(f"Classe prédite : {predicted_class}")
        st.image(f"images/{predicted_class}.jpg", use_column_width=True, caption=f"Iris {predicted_class}")
    except Exception as e:
        st.write("Erreur lors de la prédiction :", str(e))
