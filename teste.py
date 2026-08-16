import gc
import json
import pymupdf
from rapidocr_onnxruntime import RapidOCR

import wordninja

def corrigir_texto(texto):
    # Só tenta separar se não tiver espaços (palavra colada)
    if " " not in texto:
        palavras = wordninja.split(texto)
        return " ".join(palavras)
    return texto


# Aponta explicitamente pros modelos de inglês
engine_ocr = RapidOCR(
    model_type="en",
    cls_use=False  # classificador de orientação não é necessário pra quadrinhos
)


def escanear_hq(doc, caminho_json="ocr_resultado.json"):
    """
    Etapa 1: só OCR, sem tradução. Salva JSON pra revisão.
    """
    resultado_total = {}

    for i, page in enumerate(doc):
        print(f"Escaneando página {i+1}/{len(doc)}...")

        dpi = 150
        pix = page.get_pixmap(dpi=dpi)
        escala_x = page.rect.width / pix.width
        escala_y = page.rect.height / pix.height

        # RapidOCR aceita bytes PNG direto
        img_bytes = pix.tobytes("png")
        result, _ = engine_ocr(img_bytes)

        blocos_pagina = []
        if result:
            for linha in result:
                coords_px = linha[0]   # [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
                texto     = linha[1]
                texto = corrigir_texto(texto)  # adiciona essa linha
                confianca = linha[2]

                if len(texto.strip()) < 2:
                    continue
                if confianca < 0.5:    # descarta leituras ruins
                    continue

                # Converte pixels → pontos PDF
                x0 = coords_px[0][0] * escala_x
                y0 = coords_px[0][1] * escala_y
                x2 = coords_px[2][0] * escala_x
                y2 = coords_px[2][1] * escala_y

                blocos_pagina.append({
                    "texto":     texto,
                    "coords":    [x0, y0, x2, y2],
                    "confianca": round(confianca, 3)
                })

        resultado_total[str(i)] = blocos_pagina

        del pix
        gc.collect()

    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(resultado_total, f, ensure_ascii=False, indent=2)

    print(f"\nOCR concluído — salvo em '{caminho_json}'")
    print("Revise o arquivo antes de traduzir!")


def traduzir_hq(doc, tokenizer, model,
                caminho_json="ocr_resultado.json",
                caminho_saida="hq_traduzida_final.pdf"):
    """
    Etapa 2: lê o JSON revisado e sobrescreve no PDF.
    """
    with open(caminho_json, "r", encoding="utf-8") as f:
        dados = json.load(f)

    for pag_str, blocos in dados.items():
        pag_num = int(pag_str)
        page = doc[pag_num]
        print(f"Traduzindo página {pag_num + 1}...")

        for bloco in blocos:
            texto_orig = bloco["texto"]
            x0, y0, x2, y2 = bloco["coords"]

            tokens_en = tokenizer.convert_ids_to_tokens(
                tokenizer.encode(texto_orig)
            )
            resultado_ia = model.translate_batch([tokens_en], beam_size=5)
            tokens_pt = resultado_ia[0].hypotheses[0]
            ids_saida = tokenizer.convert_tokens_to_ids(tokens_pt)
            texto_traduzido = tokenizer.decode(ids_saida, skip_special_tokens=True)

            rect = pymupdf.Rect(x0, y0, x2, y2)
            page.add_redact_annot(rect, fill=(1, 1, 1))
            page.apply_redactions()
            page.insert_htmlbox(
                rect,
                f'<div style="text-align:center; font-family:sans-serif; '
                f'font-size:8px; color:black;">{texto_traduzido}</div>'
            )

    doc.save(caminho_saida)
    print(f"HQ salva em '{caminho_saida}'")