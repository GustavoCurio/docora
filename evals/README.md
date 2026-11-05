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
  cabeçalhos com referência e divisórias. Cada sentença herda a citação do bloco.
  Após normalização de acentos, caixa e remoção das stopwords existentes, exige
  pelo menos 80% dos termos únicos no **texto do trecho citado**. Texto sem
  referência conta como não sustentado. A média é ponderada pelo número de
  sentenças. Ausência de sentenças gera `null`, nunca 100% artificial.
- **Critério atendido:** status correto e fonte aceitável nas até três fontes
  efetivamente apresentadas pela resposta, preservando o critério anterior.

Fidelidade lexical não mede pertinência à pergunta, inferência semântica,
contradição ou resposta completa. Uma seção errada copiada fielmente pode obter
100%; por isso as métricas de recuperação e OOD são apresentadas juntas.
