# Projeto Tradutor

Projeto em Python que lê e traduz PDFs usando modelo de IA. 
Funciona localmente no computador e está em fase de desenvolvimento.

## Como usar

1. **Configurar o ambiente virtual**:
   - Crie um ambiente virtual com o comando:
     ```bash
     python -m venv pymupdf
     ```
   - Ative o ambiente virtual:
     - No Linux/MacOS:
       ```bash
       source pymupdf/bin/activate
       ```
     - No Windows:
       ```bash
       pymupdf\Scripts\activate
       ```

2. **Instalar dependências**:
   - Instale as dependências listadas no arquivo `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

3. **Baixar o modelo Transformer**:
   - Certifique-se de que o modelo Transformer necessário está disponível no caminho especificado no código. Caso contrário, faça o download do modelo apropriado e ajuste o caminho no arquivo de configuração. O modelo utilizado foi Helsinki-NLP/opus-mt-tc-big-en-pt

4. **Executar o projeto**:
   - Rode o script principal em sua IDE de preferência ou diretamente no terminal:
     ```bash
     python main.py
     ```

> **Nota**: Atualmente, o caminho do modelo e do documento PDF deve ser ajustado diretamente no código-fonte. Planejamos adicionar um arquivo de configuração para facilitar essa personalização em versões futuras.
