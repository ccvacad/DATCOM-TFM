#Libraries
import numpy as np
import pandas as pd

from collections import Counter
from scipy.stats import entropy
from scipy.stats import spearmanr

import spacy

import sys
import os
import torch
import ast

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.pandas2ri as rpy2_pandas_converter

from transformers import BertForSequenceClassification, BertTokenizer
from deep_translator import GoogleTranslator
import torch


src_path = os.path.join('./', 'Empathy-Mental-Health', 'src')

# Añade la ruta 'src' al sys.path
sys.path.append(src_path)
from empathy_classifier import EmpathyClassifier

""" from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline """

def analizar_sentimientos_r(user_dialog_turns: list, agent_dialog_turns: list):
    """
    Procesa los turnos de diálogo del usuario y del agente para analizar sentimientos
    utilizando el paquete syuzhet de R.

    Args:
        user_dialog_turns (list): Lista de cadenas de texto de los turnos del usuario.
        agent_dialog_turns (list): Lista de cadenas de texto de los turnos del agente.

    Returns:
        tuple: Una tupla que contiene dos pandas DataFrames:
               - df_user_resultados: DataFrame con los resultados del análisis de sentimientos del usuario.
               - df_agent_resultados: DataFrame con los resultados del análisis de sentimientos del agente.
    """

    # Importar librerías de R
    base = importr('base')
    syuzhet = importr('syuzhet')
    dplyr = importr('dplyr')

    # Convertir listas de Python a vectores de caracteres de R
    r_user_dialog_turns = base.as_character(ro.StrVector(user_dialog_turns))
    r_agent_dialog_turns = base.as_character(ro.StrVector(agent_dialog_turns))

    # Definir el código R como una cadena de texto multilínea
    r_code = """
    # Procesar emociones del usuario
    user_resultados <- lapply(user_dialog_turns, function(texto) {
      sentimientos <- get_sentiment(texto, method = "nrc", language = "spanish")
      categorias <- get_nrc_sentiment(texto, language = "spanish")
      data.frame(texto = texto, sentimiento = sentimientos, categorias)
    })

    # Procesar emociones del agente
    agent_resultados <- lapply(agent_dialog_turns, function(texto) {
      sentimientos <- get_sentiment(texto, method = "nrc", language = "spanish")
      categorias <- get_nrc_sentiment(texto, language = "spanish")
      data.frame(texto = texto, sentimiento = sentimientos, categorias)
    })

    # Convertir listas a data frames
    df_user_resultados <- bind_rows(user_resultados)
    df_agent_resultados <- bind_rows(agent_resultados)
    """

    # Asignar variables de Python al entorno R
    ro.globalenv['user_dialog_turns'] = r_user_dialog_turns
    ro.globalenv['agent_dialog_turns'] = r_agent_dialog_turns

    # Ejecutar el código R
    ro.r(r_code)

    # Utilizar localconverter para asegurar la conversión de R DataFrame a pandas DataFrame
    # Corrected: Use rpy2_pandas_converter.converter instead of ro.pandas2ri.converter
    with localconverter(ro.default_converter + rpy2_pandas_converter.converter):
        df_user_resultados = ro.globalenv['df_user_resultados']
        df_agent_resultados = ro.globalenv['df_agent_resultados']

    return df_user_resultados, df_agent_resultados
 
def emotional_entropy(df):
    '''Entropia emocional

    :param sentences: List of sentences
    :type sentences: list
    :return: Entropy value between 0 and 1
    :rtype: float
    '''

    # Asegurar que solo usamos columnas de emociones (numéricas)
    emotion_cols = df.select_dtypes(include=[np.number]).columns

    # Calcular la entropía para cada fila
    entropies = []
    for _, row in df[emotion_cols].iterrows():
        probs = row.values
        total = np.sum(probs)
        if total == 0:
            entropies.append(0.0)
        else:
            norm_probs = probs / total
            ent = entropy(norm_probs) / np.log(len(probs))  # Normalizar entropía
            entropies.append(ent)

    # Promedio de las entropías
    return np.mean(entropies)

def emotional_matching(df_user, df_agent):
    # Validar que los DataFrames tienen la misma forma
    assert df_user.shape == df_agent.shape, "Los DataFrames deben tener la misma forma"
    assert all(df_user.columns == df_agent.columns), "Las columnas de los DataFrames deben coincidir"

    correlations = []

    for idx, (user_row, agent_row) in enumerate(zip(df_user.iterrows(), df_agent.iterrows())):
        user_vals = user_row[1].values.astype(float)
        agent_vals = agent_row[1].values.astype(float)

        if np.sum(user_vals) == 0 or np.sum(agent_vals) == 0:
            correlations.append(0.0)
        else:
            corr, _ = spearmanr(user_vals, agent_vals)
            correlations.append(corr if not pd.isna(corr) else 0.0)

    return np.mean(correlations)

def split_text(text, max_length=5000):
    """Divide un texto en fragmentos más pequeños para cumplir con el límite de caracteres."""
    fragments = []
    while len(text) > max_length:
        # Buscar el último espacio dentro del límite para evitar cortar palabras
        split_index = text[:max_length].rfind(" ")
        if split_index == -1:  # Si no hay espacios, cortar directamente
            split_index = max_length
        fragments.append(text[:split_index])
        text = text[split_index:].strip()
    fragments.append(text)  # Agregar el fragmento restante
    return fragments

def agreeableness(sentences: list):
    '''Agreeableness

    :param sentences: List of sentences
    :type sentences: list
    :return: Agreeableness value between 0 and 1
    :rtype: float
    '''

    # Initialization of the model values
    model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality", num_labels=5)
    tokenizer = BertTokenizer.from_pretrained('Minej/bert-base-personality', do_lower_case=True)
    model.config.label2id = {
        "Extroversion": 0,
        "Neuroticism": 1,
        "Agreeableness": 2,
        "Conscientiousness": 3,
        "Openness": 4,
    }
    model.config.id2label = {
        "0": "Extroversion",
        "1": "Neuroticism",
        "2": "Agreeableness",
        "3": "Conscientiousness",
        "4": "Openness",
    }

    def personality_detection(model_input: str) -> dict:
        '''
        Performs personality prediction on the given input text

        Args: 
            model_input (str): The text conversation 

        Returns:
            dict: A dictionary where keys are speaker labels and values are their personality predictions
        '''

        if len(model_input) == 0:
            ret = {
                "Extroversion": float(0),
                "Neuroticism": float(0),
                "Agreeableness": float(0),
                "Conscientiousness": float(0),
                "Openness": float(0),
            }
            return ret
        else:
            dict_custom = {}
            preprocess_part1 = model_input[:len(model_input)]
            dict1 = tokenizer.encode_plus(preprocess_part1, max_length=512, padding=True, truncation=True)  # Truncar a 512 tokens
            dict_custom['input_ids'] = [dict1['input_ids'], dict1['input_ids']]
            dict_custom['token_type_ids'] = [dict1['token_type_ids'], dict1['token_type_ids']]
            dict_custom['attention_mask'] = [dict1['attention_mask'], dict1['attention_mask']]
            outs = model(torch.tensor(dict_custom['input_ids']), token_type_ids=None, attention_mask=torch.tensor(dict_custom['attention_mask']))
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)
            ret = {
                "Extroversion": float(pred_label[0][0]),
                "Neuroticism": float(pred_label[0][1]),
                "Agreeableness": float(pred_label[0][2]),
                "Conscientiousness": float(pred_label[0][3]),
                "Openness": float(pred_label[0][4]),
            }
            return ret

    results = []

    for sentence in sentences:
        # Dividir el texto en fragmentos si excede el límite de caracteres
        text_fragments = split_text(sentence, max_length=5000)
        translated_fragments = []

        for fragment in text_fragments:
            try:
                translated_fragment = GoogleTranslator(source='es', target='en').translate(fragment)
                translated_fragments.append(translated_fragment)
            except Exception as e:
                print(f"Error al traducir el fragmento: {fragment[:50]}... - {str(e)}")

        text_input = " ".join(translated_fragments)  # Combinar los fragmentos traducidos

        personality = personality_detection(text_input)
        results.append({
            'sentence': sentence,
            'Extroversion': personality['Extroversion'],
            'Neuroticism': personality['Neuroticism'],
            'Agreeableness': personality['Agreeableness'],
            'Conscientiousness': personality['Conscientiousness'],
            'Openness': personality['Openness']
        })

    results = pd.DataFrame(results)
    return results

def linguistic_matching(user_sentences: list, agent_sentences: list):
    '''Linguistic Matching

    :param user_sentences: List of user sentences
    :type user_sentences: list
    :param agent_sentences: List of agent sentences
    :type agent_sentences: list
    :return: Linguistic matching value between 0 and 1
    :rtype: float
    '''
    
    nlp = spacy.load("es_core_news_md")

    categories = ["pronombres_personales", "pronombres_impersonales", "artículos", "conjunciones", "preposiciones", "verbos_auxiliares", "adverbios_frecuentes", "negaciones", "cuantificadores"]


    def get_pos_counts(text):
        doc = nlp(text)
        total_tokens = len([t for t in doc if t.is_alpha])

        counts = {
            "pronombres_personales": sum(1 for t in doc if t.pos_ == "PRON" and t.dep_ in {"nsubj", "obj", "iobj"}),
            "pronombres_impersonales": sum(1 for t in doc if t.pos_ == "PRON" and t.lemma_ in {"uno", "alguien", "nadie", "cualquiera"}),
            "artículos": sum(1 for t in doc if t.pos_ == "DET" and t.tag_ == "DA"),
            "conjunciones": sum(1 for t in doc if t.pos_ in {"CCONJ", "SCONJ"}),
            "preposiciones": sum(1 for t in doc if t.pos_ == "ADP"),
            "verbos_auxiliares": sum(1 for t in doc if t.pos_ == "AUX"),
            "adverbios_frecuentes": sum(1 for t in doc if t.pos_ == "ADV" and t.lemma_ in {"muy", "ya", "nunca", "siempre", "también"}),
            "negaciones": sum(1 for t in doc if t.dep_ == "neg" or t.lemma_ in {"no", "nunca", "jamás", "tampoco"}),
            "cuantificadores": sum(1 for t in doc if t.lemma_ in {"todo", "mucho", "poco", "varios", "algunos", "más", "menos", "cada"})
        }

        proportions = {k: v / total_tokens if total_tokens else 0 for k, v in counts.items()}
        return proportions

    results = []
    for i in range(len(user_sentences)):
        props1 = get_pos_counts(user_sentences[i])
        props2 = get_pos_counts(agent_sentences[i])

        lsm_scores = {}
        for category in categories:
            p1 = props1.get(category, 0.0)
            p2 = props2.get(category, 0.0)
            lsm = 1 - abs(p1 - p2) / (p1 + p2 + 0.0001)  # 0.0001 para evitar división por cero
            lsm_scores[category] = lsm

        lsm_avg = np.mean(list(lsm_scores.values()))
        results.append({
            'user_sentence': user_sentences[i],
            'agent_sentence': agent_sentences[i],
            'linguistic_matching_score': lsm_avg,
            'lsm_scores': lsm_scores
        })
    results = pd.DataFrame(results)
    return results
 
def empathy(seeker_posts, response_posts, 
                     ER_model_path='Empathy-Mental-Health/output/reddit_ER.pth', IP_model_path='Empathy-Mental-Health/output/reddit_IP.pth', EX_model_path='Empathy-Mental-Health/output/reddit_EX.pth', 
                     device=None):
    """
    Clasifica empatía para listas de seeker_posts y response_posts.
    Retorna una lista de diccionarios con los resultados.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    empathy_classifier = EmpathyClassifier(
        device,
        ER_model_path=ER_model_path,
        IP_model_path=IP_model_path,
        EX_model_path=EX_model_path,
    )

    results = []
    for i in range(len(seeker_posts)):
        (
            logits_empathy_ER, predictions_ER,
            logits_empathy_IP, predictions_IP,
            logits_empathy_EX, predictions_EX,
            logits_rationale_ER, predictions_rationale_ER,
            logits_rationale_IP, predictions_rationale_IP,
            logits_rationale_EX, predictions_rationale_EX
        ) = empathy_classifier.predict_empathy(
            [seeker_posts[i]], [response_posts[i]]
        )

        results.append({
            'seeker_post': seeker_posts[i],
            'response_post': response_posts[i],
            'ER_label': predictions_ER[0],
            'IP_label': predictions_IP[0],
            'EX_label': predictions_EX[0],
            'score_avg': np.mean([[predictions_ER[0], predictions_IP[0], predictions_EX[0]]]),
            'ER_rationale': predictions_rationale_ER[0].tolist(),
            'IP_rationale': predictions_rationale_IP[0].tolist(),
            'EX_rationale': predictions_rationale_EX[0].tolist()
        })
    results = pd.DataFrame(results)
    return results


def compute_metrics(user_dialog_turns, agent_dialog_turns):
    """
    Computes various metrics based on user and agent dialog turns.
    Returns a dictionary with the computed metrics.
    """
    df_user, df_agent = analizar_sentimientos_r(user_dialog_turns, agent_dialog_turns)
    
    emotional_entropy_value = emotional_entropy(df_agent.iloc[:, 2:])
    emotional_matching_value = emotional_matching(df_user.iloc[:, 2:], df_agent.iloc[:, 2:])
    
    agreeableness_results = agreeableness(agent_dialog_turns)
    linguistic_matching_results = linguistic_matching(user_dialog_turns, agent_dialog_turns)

    empathy_results = empathy(user_dialog_turns, agent_dialog_turns)

    return {
        'emotional_entropy': emotional_entropy_value,
        'emotional_matching': emotional_matching_value,
        'agreeableness': agreeableness_results.loc[:, 'Agreeableness'].mean(),
        'linguistic_matching': linguistic_matching_results.loc[:, 'linguistic_matching_score'].mean(),
        'empathy': empathy_results.loc[:, 'score_avg'].mean()
    }

def main():
    # Leer CSV
    data = pd.read_csv("resultados/simulaciones_conversacion.csv")
    
    results = []
    for row in data.itertuples():
        print(f"Processing row {row.Index}")
        
        # Convertir strings a listas/objetos
        youth = ast.literal_eval(row.youth)
        mentescopin = ast.literal_eval(row.mentescopin)
        
        # Calcular métricas (debe retornar un dict con las 5 métricas)
        result = compute_metrics(youth, mentescopin)
        
        print(f"Metrics for row {row.Index}: {result}")
        
        # Guardar en lista de resultados
        results.append(result)

    # Convertir lista de dicts en DataFrame
    results_df = pd.DataFrame(results)

    # Concatenar resultados con el DataFrame original
    data = pd.concat([data, results_df], axis=1)

    # Guardar en nuevo CSV
    data.to_csv("resultados/simulaciones_conversacion_resultados.csv", index=False)

if __name__ == "__main__":    
    main()