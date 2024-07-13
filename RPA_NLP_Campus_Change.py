'''

Version: 2.0
Created date: Thursday Jul 05 16:30:00 2024
ClassName: ADM_CASO_CAMBIO_CAMPUS


-----Version history-------
Version     Developer               Date                Description                             Changes
2.0         Daniel Pereira      06-07-2024         RPA para realizar cambio de campus.       A esta versión se le implementó un modelo de clasificación entrenado especialmente para cambios de campus.
----------------------------

'''


from simple_salesforce import Salesforce, SalesforceLogin
import pandas as pd
import torch
from transformers import BertTokenizer, BertForTokenClassification, pipeline
from collections import OrderedDict
from datetime import datetime, timedelta
from pytz import UTC
import configparser
import traceback
import mysql.connector
from mysql.connector import errorcode
from email.message import EmailMessage
import requests
import smtplib
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import json
import re
import nltk
from nltk.corpus import stopwords


# PROCESO E INSTANCIA
PROCESO = "ADM_CAMBIO_DE_CAMPUS"
INSTANCIA = "SAND"

# CREDENCIALES
USUARIO_UAT = ' '
CONTRASENA_UAT = ' '
TOKEN_SEGURIDAD_UAT = ' '
ORGANIZATION_ID = ''

# AUTENICACIÓN CON SALESFORCE
id_de_sesion, instance = SalesforceLogin(organizationId=ORGANIZATION_ID, username=USUARIO_UAT, password=CONTRASENA_UAT, security_token=TOKEN_SEGURIDAD_UAT, domain="test")
sf = Salesforce(instance=instance, session_id=id_de_sesion)

#nltk.download('stopwords')

#--------------------------------------------FUNCIONES-------------------------------------------------------------------

def limpiar_diccionarios(registros, llave_principal="", separador="."):
    items = []
    for llave, valor in registros.items():
        llave_nueva = f"{llave_principal}{separador}{llave}" if llave_principal else llave
        if isinstance(valor, OrderedDict):
            items.extend(limpiar_diccionarios(valor, llave_nueva, separador=separador).items())
        else:
            items.append((llave_nueva, valor))
    return OrderedDict(items)

def f_enviarConsulta(query):
    df_response_total = pd.DataFrame()
    procesamiento = True
    contador = 1
    
    while procesamiento:
        if contador == 1:
            response = sf.query(query)
        else:
            try:
                response = sf.query_more(response["nextRecordsUrl"], True)
            except Exception:
                procesamiento = False
                break
        
        registros = response["records"]
        registros_limpios = [limpiar_diccionarios(registro) for registro in registros]
        df_response = pd.DataFrame(registros_limpios)
        df_response_total = pd.concat([df_response_total, df_response], ignore_index=True)
        
        contador += 1
    
    return df_response_total


# Preprocesar el texto
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remover digitos
    text = re.sub(r'\s+', ' ', text)  # Remover espacios
    text = re.sub(r'[^\w\s]', '', text)  # Remover puntuación
    stop_words = set(stopwords.words('spanish'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


#FUNCION PARA EXTRAER LA CIUDAD DEL CASO
def extract_city(new_text):
    # Cargar el modelo
    model = tf.keras.models.load_model('/Users/Daniel/Documents/ITESM/Intern/RPAs/campus_ner_model_improved.keras')
    
    # Cargar el tokenizer
    with open('/Users/Daniel/Documents/ITESM/Intern/RPAs/tokenizer.json') as f:
        tokenizer_json = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
    
    # Cargar el label encoder
    label_encoder_classes = np.load('/Users/Daniel/Documents/ITESM/Intern/RPAs/label_encoder.npy', allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_encoder_classes
    
    # Preprocesar el texto
    preprocessed_text = preprocess_text(new_text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    max_len = 60
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    
    # Predecir el campus
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction, axis=1)
    predicted_campus = label_encoder.inverse_transform(predicted_label)
    
    return predicted_campus[0]

def extraer_descripciones(series):
    return series.apply(extract_city)


#FUNCIÓN PARA BUSCAR EL ID DEL CAMPUS SOLICITADO EN EL CASO
def buscar_campus(campus, catalogo):
    return catalogo[catalogo.str.contains(campus, case=False, na=False)]


#CONSULTA PARA EXTRAER LOS CASOS CON LA CLASIFICACIÓN CORRESPONDIENTE
consulta_casos = """SELECT Id, ADM_Solicitud__c, ADM_Reporta__c, ContactId, Description FROM Case WHERE ADM_Clasificacion_caso__r.ADM_clave__c = 'CC158' AND Status IN ('Nuevo', 'Asignado', 'Actualizado') AND Owner.Email = 'daniel.pereira@tec.mx' AND ContactId != ''"""

try:
    casos = f_enviarConsulta(consulta_casos)
    casos.rename(columns={"Id": "CaseId"}, inplace=True)
    casos.drop(columns=["attributes.type", "attributes.url"], inplace=True)
    
    #CONSULTA PARA EXTRAER LOS CAMPOS NECESARIOS DE DE LA APP
    contact_ids = ",".join([f"'{contact_id}'" for contact_id in casos["ContactId"].unique()])
    consulta_aplicaciones = f"""SELECT Id, hed__Applicant__c, ADM_Etapa__c, Programa_Academico__r.ADM_Agrupacion_Academica__c, ADM_Configuracion_de_oferta__c, Nivel__c, Programa_Academico__c, Programa_Academico__r.Program_Plan__c, Programa_Academico__r.ADM_Conf_de_agrupacion_academica__c, ADM_Configuracion_nivel_periodo__c, ADM_Estatus__c, Programa_Academico__r.Name FROM hed__Application__c WHERE hed__Applicant__c IN ({contact_ids})"""
    aplicaciones = f_enviarConsulta(consulta_aplicaciones)

    #CONSULTA PARA EXTRAER LAS CONFIGURACIONES DE OFERTA CANDIDATAS PARA EL CAMPUS NUEVO
    configuraciones_ids = ",".join([f"'{config_id}'" for config_id in aplicaciones["ADM_Configuracion_nivel_periodo__c"].unique()])
    niveles = ",".join([f"'{nivel}'" for nivel in aplicaciones["Nivel__c"].unique()])
    consulta_configuraciones_oferta = f"""SELECT Name, Id, ADM_Configuracion_Nivel_Periodo__c, ADM_Nivel__c FROM ADM_Configuracion_Oferta__c WHERE ADM_Configuracion_Nivel_Periodo__c IN ({configuraciones_ids}) AND ADM_Nivel__c IN ({niveles})"""
    configuraciones_oferta = f_enviarConsulta(consulta_configuraciones_oferta)

    #CONSULTAS PARA EXTRAER LAS AGRUPACIONES ACADÉMICAS CANDIDATAS PARA EL CAMPUS NUEVO
    nombres_agrupaciones = ",".join([f"'{name}'" for name in aplicaciones["Programa_Academico__r.Name"].unique()])
    oferta_ids = ",".join([f"'{oferta_id}'" for oferta_id in configuraciones_oferta["Id"].unique()])
    consulta_agrupaciones_academicas = f"""SELECT Id, Name, ADM_Conf_de_agrupacion_academica__c, ADM_Configuracion_de_oferta__c FROM Program_Offering__c WHERE Name IN ({nombres_agrupaciones}) AND ADM_Configuracion_de_oferta__c IN ({oferta_ids})"""
    agrupaciones_academicas = f_enviarConsulta(consulta_agrupaciones_academicas)
except Exception as e:
    print("Error en consulta")
    print(e)
    sys.exit(1)


descripciones = casos['Description']
ciudades = extraer_descripciones(descripciones) #BUSCANDO LA CIUDAD NOMBRADA


for index, row in casos.iterrows():
    contact_id = row["ContactId"]
    caso_id = row["CaseId"]

    df_solicitud_actual = aplicaciones[aplicaciones['hed__Applicant__c'] == contact_id]

    if df_solicitud_actual.empty:
        print(f"No data found for contact id: {contact_id}")
        continue

    etapa = df_solicitud_actual['ADM_Etapa__c'].iloc[0]
    estatus = df_solicitud_actual['ADM_Estatus__c'].iloc[0]

    #VALIDANDO QUE SE PUEDA REALIZAR EL CAMBIO
    if etapa == "Entrega de requisitos" and estatus == "Activa":
        nivel = df_solicitud_actual["Nivel__c"].iloc[0]
        configuracion = df_solicitud_actual["ADM_Configuracion_nivel_periodo__c"].iloc[0]

        df_oferta_filtrada = configuraciones_oferta[(configuraciones_oferta["ADM_Configuracion_Nivel_Periodo__c"] == configuracion) & (configuraciones_oferta["ADM_Nivel__c"] == nivel)]
        
        #BUSCANDO EL CAMPUS NUEVO EN EL CATALOGO DE LA OFERTA
        oferta_de_campus = df_oferta_filtrada['Name']
        oferta_de_campus_id = df_oferta_filtrada['Id']

        ciudad = ciudades[index]
        campus_catalogo = buscar_campus(ciudad, oferta_de_campus)

        if not campus_catalogo.empty:
            indice = campus_catalogo.index[0]
            Id_campus_nuevo = oferta_de_campus_id[indice]
            programa_acad_viejo = df_solicitud_actual['Programa_Academico__c'].iloc[0]
            conf_agrupacion = df_solicitud_actual['Programa_Academico__r.ADM_Conf_de_agrupacion_academica__c'].iloc[0]
            nombre_conf_agrupacion_vieja = df_solicitud_actual['Programa_Academico__r.Name'].iloc[0]

            df_solicitud_nueva = agrupaciones_academicas[(agrupaciones_academicas['Name'] == nombre_conf_agrupacion_vieja) & (agrupaciones_academicas['ADM_Configuracion_de_oferta__c'] == Id_campus_nuevo)]

            # ACTUALIZANDO LOS CAMPOS DE LA APP CON LOS NUEVOS VALORES 
            if not df_solicitud_nueva.empty:
                conf_agrupacion_campus_nuevo = df_solicitud_nueva['ADM_Conf_de_agrupacion_academica__c'].iloc[0]
                programa_acad_nuevo = df_solicitud_nueva['Id'].iloc[0]
                Id_App = df_solicitud_actual['Id'].iloc[0]
                sf.hed__Application__c.update(Id_App, {'ADM_Configuracion_de_oferta__c': Id_campus_nuevo,'ADMConfiguracion_Agrupacion_Academica__c': conf_agrupacion_campus_nuevo,'Programa_Academico__c': programa_acad_nuevo})
                sf.Case.update(caso_id, {'Status': 'Atendido'})

