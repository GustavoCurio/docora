# Avaliação de recuperação

`questions.json` é o conjunto original: 20 consultas com seção esperada, cinco
negativos e três encaminhamentos. `heldout_questions.json` é um conjunto
**held-out separado**, congelado antes de mudar a recuperação: 23 consultas com
fonte, 12 negativos e três encaminhamentos. Inclui paráfrases, linguagem coloquial,
erros leves e perguntas que combinam seções. A redação conhece a base fictícia;
não é uma amostra independente de clientes. Não usar esse arquivo para ajustar
aliases, pesos ou cortes. Depois de examinar resultados, não há garantia de
independência de futuras decisões: novos experimentos exigem outro conjunto.

`development_questions.json` contém oito negativos adicionais, separados do
held-out, para calibrar o corte semântico junto ao conjunto original.
`scripts/calibrate_retrieval.py` lê **somente** esses dois arquivos e registra a
grade de cortes em `reports/calibration_development.json`. O corte 0,85 e a janela
fixa de 0,03 foram congelados antes da avaliação híbrida no held-out. Os aliases
TF-IDF e o corpus permanecem os mesmos do baseline.

O contrato está em `questions.schema.json` (JSON Schema 2020-12). Cada caso tem
`question` e `expected_status`; `id` é opcional. Casos `evidence_found` declaram
`source` e `section`, ou `acceptable_sources`, uma lista de pares equivalentes
para fins desta métrica. Nos casos multisseção, encontrar **qualquer** alternativa
conta como acerto; não significa cobrir a pergunta inteira. `not_found` e
`needs_human` não declaram fontes. Os casos sobre reembolso avaliam a localização
da política de troca/devolução, sem supor prazo de restituição ou aprovação.

```bash
python -m docora evaluate --suite original --retriever tfidf --output reports/baseline_tfidf_original.json
python -m docora evaluate --suite heldout --retriever tfidf --allow-failures --output reports/baseline_tfidf_heldout.json
python -m docora evaluate --suite all --retriever tfidf --allow-failures
```

A execução é sempre extrativa e local, sem chamar a OpenAI. Por padrão, falhas
nos critérios encerram o comando com código 1; `--allow-failures` permite medições
exploratórias com código 0 e preserva todas as falhas no relatório. `all` grava um
objeto com `suites`, sem misturar as populações. Sem `--output`, grava no diretório
temporário do sistema. Relatórios registram hashes dos casos e do corpus, horário,
recuperador solicitado/ativo e eventual retorno ao TF-IDF.

## Regras das métricas

- **Recall@1/3/5:** fração dos casos com fonte esperada com pelo menos um par
  documento/seção aceitável nas primeiras 1/3/5 posições, após os mesmos filtros
  de relevância da aplicação. A unidade é a pergunta, não cada trecho relevante.
- **MRR@5:** média do inverso da posição da primeira fonte aceitável; zero se
  ausente entre as cinco primeiras. O limite de cinco é explícito.
- **Precisão de not_found:** abstenções corretas divididas por todas as respostas
  `not_found`, inclusive falsos negativos de perguntas respondíveis.
- **Falso positivo OOD:** `evidence_found` em casos `not_found`, dividido pelo
  total de casos `not_found`. Encaminhamentos não contam como evidência.
- **Fidelidade lexical extrativa:** separa o texto real da resposta em sentenças
  por pontuação final e quebras de linha, removendo apenas introdução fixa,
  cabeçalhos com referência e divisórias. No extrativo novo, cada sentença tem
  sua própria referência inline; o leitor também aceita referências herdadas
  de cabeçalhos para comparar os relatórios do extrativo anterior.
  Após normalização de acentos, caixa e remoção das stopwords existentes, exige
  pelo menos 80% dos termos únicos no **texto do trecho citado**. Texto sem
  referência conta como não sustentado. A média é ponderada pelo número de
  sentenças. Ausência de sentenças gera `null`, nunca 100% artificial.
- **Critério atendido:** status correto e fonte aceitável nas até três fontes
  efetivamente apresentadas pela resposta, preservando o critério anterior.

Fidelidade lexical não mede pertinência à pergunta, inferência semântica,
contradição ou resposta completa. Uma seção errada copiada fielmente pode obter
100%; por isso as métricas de recuperação e OOD são apresentadas juntas.

## Resposta por sentenças e rerankers

Os relatórios `reports/items56_before_*.json` registram o extrativo anterior com
as mesmas checagens de resposta do novo. `items5_after_*.json` usa o novo seletor,
sem reranking; `items6_rules_*.json` e `items6_crossencoder_*.json` comparam as duas
opções em TF-IDF e híbrido. Os casos, corpus, aliases e cortes não foram alterados.
O held-out serve à decisão de habilitação, sem ajuste de parâmetros após medir.

`answer_checks` mede citação inline válida por sentença, fragmento literal na
fonte citada (espaços normalizados), fonte esperada efetivamente citada no corpo,
até seis sentenças/120 palavras e ausência de duplicatas exatas. A média de
palavras inclui apenas respostas `evidence_found`. Os limites contam o texto,
sem os marcadores `[Fn]`. No baseline, as referências por cabeçalho dão suporte
lexical, mas não contam como citação inline por sentença. São checagens
estruturais e de proveniência; não medem completude, contradições ou implicação.

A avaliação faz uma busca por pergunta e reutiliza o mesmo ranking para a
resposta e R@k/MRR, evitando medir duas inferências como se fossem uma única
consulta. Guardrails são executados antes da busca. Ambos os rerankers recebem
até dez candidatos já filtrados, independentemente do corte de saída.

```bash
python -m docora prepare-model
python -m docora prepare-reranker
DOCORA_ALLOW_MODEL_DOWNLOAD=0 python -m docora evaluate --suite all --retriever hybrid --reranker none --allow-failures --output /tmp/sem-reranking.json
DOCORA_ALLOW_MODEL_DOWNLOAD=0 python -m docora evaluate --suite all --retriever hybrid --reranker rules --allow-failures --output /tmp/regras.json
DOCORA_ALLOW_MODEL_DOWNLOAD=0 python -m docora evaluate --suite all --retriever hybrid --reranker crossencoder --allow-failures --output /tmp/ce.json
python scripts/benchmark_reranking.py --reranker crossencoder --output /tmp/recursos-ce.json
python -m pytest -q -m slow
```

`items6_runtime_*.json` mede processos novos Linux com E5 e Streamlit AppTest;
o arquivo `crossencoder_cold` começa sem CE, mas com o E5 em disco. Cada opção
executa três rodadas das 35 perguntas held-out sem guardrail, incluindo OOD.
Latência de busca, resposta completa e reranking não vazio são separadas.
Também há três medições diretas de dez pares. Diferenças de RSS entre processos
incluem ruído de alocação; as regras não adicionam modelo. A medição local não
é teste de carga ou medição do contêiner do Community Cloud.
