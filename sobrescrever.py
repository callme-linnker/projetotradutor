from enginetradutor import traduzir_texto
import pymupdf

def reescrever_pag(batch_para_escrita, doc):
    """
    Recebe uma lista de (texto, (x0, y0, x1, y1), num_pagina)
    e aplica as edições no documento original.
    """
    for texto_traduzido, coords, pag_num in batch_para_escrita:
        page = doc[pag_num]
        
        # Cria o retângulo usando as coordenadas salvas no batch
        rect_do_bloco = pymupdf.Rect(coords)
        
        # 1. Apaga o texto original (Redação)
        page.add_redact_annot(rect_do_bloco, fill=(1, 1, 1)) # Fundo branco
        page.apply_redactions()
        
        # 2. Insere o novo texto traduzido
        # O htmlbox é ótimo porque ajusta o tamanho da fonte automaticamente
        page.insert_htmlbox(rect_do_bloco, texto_traduzido)
