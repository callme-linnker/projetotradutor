import pymupdf
from nltk import sent_tokenize  # pip install nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from enginetradutor import traduzir_doc, traduzir_paragraf
from navegarpag import carregar_pagina

modelo_nome = "Helsinki-NLP/opus-mt-tc-big-en-pt"
tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
model = AutoModelForSeq2SeqLM.from_pretrained(modelo_nome)


# Move o modelo pra GPU se tiver
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
x = 0
doc = pymupdf.open("Cozzolino_Audio-Visual_Person-of-Interest_DeepFake_Detection_CVPRW_2023_paper.pdf")

def busc_frase(frase, blocos):
    listablock = []
    for block in blocos:
        if frase.lower() in block[4].lower():
            listablock.append(block)
    if not listablock:
        print("Frase não encontrada na página")   
        return []
    if listablock:
        return listablock


while True:
    print("Comandos: \n1 - Buscar frase:Q\n2 - Próxima página:W\n3 - Página anterior:E\n4 - Sair:C\n5 - Buscar página:T\n6 - TRADUZIR DOCUMENTO(INTEIRO):Z")
    comando = input("> ").strip().lower()

    if comando == "c":
        break

    elif comando == "q":
        print(f"Página atual: {x}")
        frase = input("Digite a frase: ")
        teste = busc_frase(frase, blocos)
        texto_traduzido = traduzir_paragraf(teste, tokenizer, model, device)
        print(f"Tradução: {texto_traduzido}")
    elif comando == "w":
        if x < len(doc) - 1:
            x += 1
            print(f"Página atual: {x}")

    elif comando == "e":
        if x > 0:
            x -= 1
            print(f"Página atual: {x}")
    elif comando == "t":
        navegar = int(input("Insira o número da página: ").strip())
        x = navegar - 1
        print(f"Página atual: {x}")
    elif comando == "z":
        info_blocopag = traduzir_doc(doc, tokenizer, model, device)
        print(info_blocopag)
    blocos = carregar_pagina(doc, x)
    
    