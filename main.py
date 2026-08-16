import pymupdf
from nltk import sent_tokenize  # biblioteca de PLN
from transformers import AutoTokenizer
#import torch
from enginetradutor import traduzir_paragraf, traduzir_doc
from navegarpag import carregar_pagina
import ctranslate2 as c2
from teste import escanear_hq, traduzir_hq

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")
model = c2.Translator("helsinki_pt_quantizado", device="cuda", compute_type="int8")

x = 0
doc = pymupdf.open("handsonmachinelearning.pdf")
#doc = pymupdf.open("Cozzolino_Audio-Visual_Person-of-Interest_DeepFake_Detection_CVPRW_2023_paper.pdf")
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


blocos = carregar_pagina(doc, x)  # Inicializa blocos antes de usar
while True:
    print("Comandos: \n1 - Buscar frase:Q\n2 - Próxima página:W\n3 - Página anterior:E\n4 - Sair:C\n5 - Buscar página:T\n6 - TRADUZIR DOCUMENTO(INTEIRO):Z\n7 - TRADUZIR HQ(DEMO INICIAL MUITO BETA):X")
    comando = input("> ").strip().lower()

    if comando == "c":
        break

    elif comando == "q":
        print(f"Página atual: {x}")
        frase = input("Digite a frase: ")
        teste = busc_frase(frase, blocos)  # Agora blocos está definido
        texto_traduzido = traduzir_paragraf(teste, tokenizer, model)
        print(f"Tradução: {texto_traduzido}")
    elif comando == "w":
        if x < len(doc) - 1:
            x += 1
            print(f"Página atual: {x}")
            blocos = carregar_pagina(doc, x)  # Atualiza blocos para a nova página

    elif comando == "e":
        if x > 0:
            x -= 1
            print(f"Página atual: {x}")
            blocos = carregar_pagina(doc, x)  # Atualiza blocos para a nova página
    elif comando == "t":
        navegar = int(input("Insira o número da página: ").strip())
        x = navegar - 1
        print(f"Página atual: {x}")
        blocos = carregar_pagina(doc, x)  # Atualiza blocos para a nova página
    elif comando == "z":
        info_blocopag = traduzir_doc(doc, tokenizer, model)
        print(info_blocopag)
    elif comando == "x":
        print("Etapa 1: escaneando HQ (só OCR, sem tradução)...")
        escanear_hq(doc)  # salva ocr_resultado.json
        print("Revise o arquivo ocr_resultado.json e use 'v' para traduzir.")

    elif comando == "v":
        print("Etapa 2: traduzindo a partir do JSON...")
        traduzir_hq(doc, tokenizer, model)
