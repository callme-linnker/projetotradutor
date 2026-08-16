from nltk import sent_tokenize
from teste2 import reescrever_pag
#import torch
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

from nltk.tokenize import sent_tokenize

def traduzir_paragraf(teste, tokenizer, model):
    texto_total = ""

    for block in teste:
        # block[4] é o texto. 
        # IMPORTANTE: tirar as quebras de linha de DENTRO do bloco antes de somar
        limpo = block[4].replace("\n", " ").strip()
        texto_total += limpo + " "

    # Garantir que espaços duplos sumam
    texto_total = " ".join(texto_total.split())

    frases = sent_tokenize(texto_total)
    
    lista_tokens = []
    for f in frases:
        ids = tokenizer.encode(f)
        lista_tokens.append(tokenizer.convert_ids_to_tokens(ids))

    # Traduz o lote de frases (muito mais rápido e preciso)
    resultados = model.translate_batch(lista_tokens)
    
    # Junta as frases traduzidas
    traducao_acumulada = []
    for r in resultados:
        # Pega a melhor hipótese (hypotheses[0])
        tokens_saida = r.hypotheses[0]
        ids_saida = tokenizer.convert_tokens_to_ids(tokens_saida)
        traducao_acumulada.append(tokenizer.decode(ids_saida, skip_special_tokens=True))

    return " ".join(traducao_acumulada)



#def traduzir_bloco(texto_orig, tokenizer, model):
#    """Traduz um único bloco e libera memória logo em seguida."""
#    texto = texto_orig.replace("\n", " ").strip()
#    if not texto:
#        return None
#
#    frases = sent_tokenize(texto)
#
#    # Processa as frases em mini-batches para não explodir a VRAM
#    MINI_BATCH = 4
#    traducoes = []
#
#    for i in range(0, len(frases), MINI_BATCH):
#        mini = frases[i:i + MINI_BATCH]
#
#        # qual o num máx q essa porra recebe de token ?
#        inputs = tokenizer(
#            mini,
#            return_tensors="pt",
#            padding=True,
#            truncation=True,
#        ).to(device)
#
#        with torch.no_grad():
#            outputs = model.generate(
#                **inputs,
#            )
#
#        parcial = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#        traducoes.extend(parcial)
#
#        # Libera tensores do mini-batch imediatamente
#        del inputs, outputs
#        torch.cuda.empty_cache()
#
#    return " ".join(traducoes)
#
#
#def traduzir_doc(doc, tokenizer, model):
#    batch = []
#    paginas_no_lote = 0
#
#    for pag_num, pagina in enumerate(doc):
#        #largura_pagina = pagina.rect.width
#        blocos = pagina.get_text("blocks", sort=True)
#
#        for block in blocos:
#            x0, y0, x1, y1, texto_orig, *_ = block
#
#            traducao_texto = traduzir_bloco(texto_orig, tokenizer, model, device)
#            if traducao_texto is None:
#                continue
#
#            batch.append((traducao_texto, (x0, y0, x1, y1), pag_num))
#
#        paginas_no_lote += 1
#
#        if paginas_no_lote == 3:
#            reescrever_pag(batch, doc)
#            batch = []
#            paginas_no_lote = 0
#            torch.cuda.empty_cache()  # limpeza extra entre lotes
#
#    if batch:
#        reescrever_pag(batch, doc)
#
#    print("Tradução concluída.")

from nltk.tokenize import sent_tokenize

def traduzir_bloco(texto_orig, tokenizer, translator):
    """Traduz um único bloco usando CTranslate2 (sem Torch)."""
    # Limpeza básica do texto do bloco
    texto = texto_orig.replace("\n", " ").strip()
    if not texto or len(texto) < 2:
        return None

    # O NLTK separa em frases para não sobrecarregar a IA
    frases = sent_tokenize(texto)
    
    # Prepara os tokens (O CTranslate2 ama listas de tokens)
    # Convertemos o texto em tokens que o modelo entende
    tokens_entrada = [
        tokenizer.convert_ids_to_tokens(tokenizer.encode(f)) 
        for f in frases
    ]

    # Tradução em Batch (Muito rápido e leve na RAM)
    # max_batch_size ajuda a controlar o uso de memória em PCs de 4GB
    resultados = translator.translate_batch(
        tokens_entrada, 
        max_batch_size=4,        # Equivale ao seu MINI_BATCH
        beam_size=5,             # Qualidade da tradução
        repetition_penalty=1.2   # Evita loops de repetição
    )

    traducoes_finais = []
    for r in resultados:
        # r.hypotheses[0] é a melhor tradução encontrada
        tokens_saida = r.hypotheses[0]
        ids_saida = tokenizer.convert_tokens_to_ids(tokens_saida)
        
        # O decode limpa espaços e tokens especiais
        frase_pronta = tokenizer.decode(ids_saida, skip_special_tokens=True)
        traducoes_finais.append(frase_pronta)

    return " ".join(traducoes_finais)


def traduzir_doc(doc, tokenizer, translator):
    batch_para_escrita = []
    
    for pag_num, pagina in enumerate(doc):
        # Pega os blocos de texto da página
        blocos = pagina.get_text("blocks", sort=True)

        for block in blocos:
            # O PyMuPDF retorna (x0, y0, x1, y1, "texto", block_no, block_type)
            x0, y0, x1, y1, texto_orig = block[:5]

            # Pula se for imagem ou bloco vazio
            if not texto_orig.strip():
                continue

            traducao_texto = traduzir_bloco(texto_orig, tokenizer, translator)
            
            if traducao_texto:
                # Guarda a tradução e a posição para escrever depois
                batch_para_escrita.append((traducao_texto, (x0, y0, x1, y1), pag_num))

        # A cada 3 páginas, salvamos o progresso (opcional, mas bom para memória)
        if (pag_num + 1) % 3 == 0:
            if batch_para_escrita:
                reescrever_pag(batch_para_escrita, doc)
                batch_para_escrita = []
            print(f"Página {pag_num + 1} processada...")

    # Traduz o que sobrou no último lote
    if batch_para_escrita:
        reescrever_pag(batch_para_escrita, doc)

    print("Tradução concluída com CTranslate2!")
