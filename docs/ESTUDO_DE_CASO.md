# Estudo de caso — Docora

## Problema

Equipes de atendimento precisam localizar regras e procedimentos em vários documentos. Uma resposta útil deve apresentar a origem da informação e reconhecer quando a base não contém o que foi perguntado.

O Docora aborda esse problema com uma aplicação de consulta documental em português e um cenário demonstrativo de e-commerce.

## Solução

- Ingestão de Markdown, TXT e PDFs com texto selecionável.
- Divisão em trechos com título, seção, linhas ou página e identificador determinístico.
- Recuperação lexical combinando TF-IDF de palavras e de caracteres.
- Resposta extrativa local e síntese generativa opcional com referências validadas.
- Interface para perguntas, documentos próprios, inspeção das fontes e exportação da consulta.
- API HTTP para integração com outros sistemas.

## Decisões técnicas

| Decisão | Justificativa |
| --- | --- |
| Recuperação lexical local | Base pequena, execução reproduzível em CPU e consulta sem credenciais externas. |
| Segmentação por seção | Preservar o contexto de cada regra comercial e tornar a fonte mais precisa. |
| Palavras e caracteres | Combinar termos exatos com variações de escrita e acentuação. |
| Bases próprias isoladas da demonstração | Evitar misturar políticas de documentos enviados com regras da loja fictícia. |
| Fontes com posição | Permitir localizar a informação original no arquivo. |
| Saída estruturada do LLM | Separar o texto da resposta e a lista de referências para validação. |
| Tratamento de perguntas sem fonte | Manter a resposta limitada ao conteúdo disponível. |

## Avaliação

O conjunto de referência contém 28 perguntas elaboradas para a base Lume Casa. As 20 consultas com resposta documental incluem uma fonte e seção esperadas; os demais casos avaliam ausência de informação e encaminhamento.

A execução de referência atende aos 28 critérios. Nas consultas com fonte esperada, a seção correta aparece entre os três primeiros trechos em 100% dos casos. O resultado é específico desse conjunto e não representa validação externa.

Os testes automatizados também cobrem um PDF real criado como fixture, identificação de página, validação de referências inexistentes, isolamento dos documentos e respostas da API. A integração com o provedor é testada por contrato, com cliente simulado.

## Fluxo de uso

1. Escolher a loja demonstrativa ou enviar documentos próprios.
2. Fazer uma pergunta sobre o conteúdo.
3. Ler os trechos ou a síntese, conforme o modo selecionado.
4. Abrir as fontes para conferir seção, posição e texto original.

## Stack

Python, scikit-learn, Streamlit, FastAPI, Pydantic, pypdf e SDK da OpenAI. Os testes usam pytest, TestClient e o ambiente de testes do Streamlit.

O projeto demonstra construção de uma aplicação de atendimento com documentos, avaliação de recuperação e integração entre interface, serviço HTTP e provedor generativo.
