import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import torch
from transformers import AutoModel, AutoTokenizer, pipeline
from pre_proccess import carregaDados


def extrai_bert_embeddings(textos):
  print("Carregando o modelo BERT pré-treinado ('bert-base-uncased')...")
  model_name = "bert-base-uncased"
  tokenizer = AutoTokenizer.from_pretrained(model_name)  #baixa o tokenizador pre treinado
  model = AutoModel.from_pretrained(model_name) #Baixa a rede neural pré-treinada do BERT

  model.eval()  # coloca em modo de inferência (desativa dropout/treino)
  embeddings = []

  print(
      f"Extraindo vetores [CLS] contextuais para {len(textos)} mensagens..."
  )
  with torch.no_grad():  # economiza memória RAM e acelera o processamento
    for texto in textos:
      # O BERT precisa da frase inteira (com maiúsculas, pontuações e stopwords)
      inputs = tokenizer(
          texto,
          return_tensors="pt",
          truncation=True,
          padding=True,
          max_length=128,
      )
      outputs = model(**inputs)

      # pega a posição [:, 0, :], que é o vetor do token especial [CLS]
      cls_vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
      embeddings.append(cls_vector)

  return np.array(embeddings)


def avalia_zero_shot(df_amostra):    #faz a classificacao zero-shot, e avalia, sem nenhum treino no nosso dataset

  print(
      "\n======================================================================"
  )
  print(f"--- PARTE 1: Classificação Zero-Shot ({len(df_amostra)} amostras) ---")
  print(
      "======================================================================"
  )

  classifier = pipeline(
      "zero-shot-classification", model="typeform/distilbert-base-uncased-mnli"
  )

  y = df_amostra["label"].tolist()
  predicoes = []

  candidate_labels = ["spam message", "normal message"]       #mapeamento pra evitar confusao com a palavra "ham"
  mapeamento = {"spam message": "spam", "normal message": "ham"}

  print("Executando inferência Zero-Shot...")
  for texto in df_amostra["text"]:
    res = classifier(texto, candidate_labels=candidate_labels)
    rotulo_previsto = mapeamento[res["labels"][0]]
    predicoes.append(rotulo_previsto)

  acc = accuracy_score(y, predicoes)
  f1_macro = f1_score(y, predicoes, average="macro")

  print(f"\n[Resultado Zero-Shot]")
  print(f"Acurácia: {acc:.4f}")
  print(f"Macro F1-Score: {f1_macro:.4f}\n")


def avalia_classificadores_bert(df):
  print(
      "\n======================================================================"
  )
  print("--- PARTE 2: Avaliação Estatística (BERT Embeddings + 10-Fold CV) ---")
  print(
      "======================================================================"
  )

  # 1. extrai a matriz X com os vetores dinâmicos do BERT
  X = extrai_bert_embeddings(df["text"].tolist())
  y = df["label"]

  # 2. Define os classificadores tradicionais
  modelos = {
      "KNN": KNeighborsClassifier(n_neighbors=7, metric="cosine"),
      "SVM": SVC(C=1, kernel="linear"),
      "Árvore": DecisionTreeClassifier(
          criterion="gini", max_depth=30, random_state=42
      ),
  }

  cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
  notas_dos_modelos = {}

  print("\n--- MÉDIAS E DESVIOS (DESEMPENHO GERAL COM BERT EMBEDDINGS) ---")
  for nome, modelo in modelos.items():
    notas = cross_val_score(
        modelo, X, y, cv=cv, scoring="f1_macro", n_jobs=-1
    )
    notas_dos_modelos[nome] = notas
    print(
        f"{nome:<8} | Média F1-Macro: {notas.mean():.4f} | Desvio Padrão:"
        f" {notas.std():.4f}"
    )

  print("\n--- COMPARAÇÃO ESTATÍSTICA (TESTE T PAREADO + BONFERRONI) ---")
  pares = [("KNN", "SVM"), ("KNN", "Árvore"), ("SVM", "Árvore")]
  limite_bonferroni = 0.05 / len(pares)
  print(f"Nível de significância (Bonferroni): {limite_bonferroni:.4f}\n")

  for m1, m2 in pares:
    estatistica, p = stats.ttest_rel(
        notas_dos_modelos[m1], notas_dos_modelos[m2]
    )

    if p < limite_bonferroni:
      melhor = m1 if estatistica > 0 else m2
      print(f"{m1} vs {m2}")
      print(
          f"   -> p-valor: {p:.5f} | Veredito: Diferença Significativa"
          f" (Vencedor: {melhor})"
      )
    else:
      print(f"{m1} vs {m2}")
      print(
          f"   -> p-valor: {p:.5f} | Veredito: Estatisticamente Equivalentes"
          " (Empate)"
      )


if __name__ == "__main__":
  # Carrega os dados usando a sua função padrão
  df = carregaDados()

  #zero shot
  df_amostra = df.sample(n=30, random_state=42)
  avalia_zero_shot(df_amostra)

  #parte 2 (extracao + classificadores + teste T)

  #avalia_classificadores_bert(df) #testar no google collab