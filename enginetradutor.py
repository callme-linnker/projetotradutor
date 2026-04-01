from nltk import sent_tokenize
from teste2 import reescrever_pag
import torch
#
##def detectar_colunas(blocos, largura_pagina):
#    meio = largura_pagina / 2
#    col_esq, col_dir = [], []
#
#    for block in blocos:
#        x0, y0, x1, y1, texto, *_ = block
#        if not texto.strip():
#            continue
#        centro_bloco = (x0 + x1) / 2
#        if centro_bloco < meio:
#            col_esq.append(block)
#        else:
#            col_dir.append(block)
#
#    col_esq.sort(key=lambda b: b[1])
#    col_dir.sort(key=lambda b: b[1])
#    return col_esq + col_dir
#


#def traduzir_paragraf(teste, tokenizer, model, device):
#
#    for block in teste:
#        texto_orig = block[4]
#        texto1 = texto_orig.replace("\n", " ")
#        frases = sent_tokenize(texto1)
#        inputs = tokenizer(frases, return_tensors="pt", padding=True, truncation=True).to(device)
#        outputs = model.generate(**inputs)
#        traducao = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#        traducao_texto = " ".join(traducao)
#    return print(traducao_texto)


from nltk.tokenize import sent_tokenize

def traduzir_paragraf(teste, tokenizer, model, device):

    texto_total = ""

    for block in teste:
        texto_total += block[4] + " "

    texto_total = texto_total.replace("\n", " ")

    frases = sent_tokenize(texto_total)

    inputs = tokenizer(
        frases,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    outputs = model.generate(**inputs)

    traducao = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    traducao_texto = " ".join(traducao)

    return traducao_texto


def traduzir_bloco(texto_orig, tokenizer, model, device):
    """Traduz um único bloco e libera memória logo em seguida."""
    texto = texto_orig.replace("\n", " ").strip()
    if not texto:
        return None

    frases = sent_tokenize(texto)

    # Processa as frases em mini-batches para não explodir a VRAM
    MINI_BATCH = 4
    traducoes = []

    for i in range(0, len(frases), MINI_BATCH):
        mini = frases[i:i + MINI_BATCH]

        # qual o num máx q essa porra recebe de token ?
        inputs = tokenizer(
            mini,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
            )

        parcial = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        traducoes.extend(parcial)

        # Libera tensores do mini-batch imediatamente
        del inputs, outputs
        torch.cuda.empty_cache()

    return " ".join(traducoes)


def traduzir_doc(doc, tokenizer, model, device):
    batch = []
    paginas_no_lote = 0

    for pag_num, pagina in enumerate(doc):
        #largura_pagina = pagina.rect.width
        blocos = pagina.get_text("blocks", sort=True)

        for block in blocos:
            x0, y0, x1, y1, texto_orig, *_ = block

            traducao_texto = traduzir_bloco(texto_orig, tokenizer, model, device)
            if traducao_texto is None:
                continue

            batch.append((traducao_texto, (x0, y0, x1, y1), pag_num))

        paginas_no_lote += 1

        if paginas_no_lote == 3:
            reescrever_pag(batch, doc)
            batch = []
            paginas_no_lote = 0
            torch.cuda.empty_cache()  # limpeza extra entre lotes

    if batch:
        reescrever_pag(batch, doc)

    print("Tradução concluída.")