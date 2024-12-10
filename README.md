![image](https://github.com/Henrique-Peter/fraud_detection/blob/main/images/case_fraude.png)


## 📌 Visão Geral
Este projeto visa a identificação de compras digitais fraudulentas, que podem causar muita dor de cabeça para os usuários, além de prejuízo para a instituição financeira em questão. Para isso, criaremos um modelo de Machine Learning que supere os resultados de um modelo previamente testado com os mesmos dados. O modelo antigo é apresentado no dataset, então conseguimos avaliar o desempenho dele e depois comparar com o novo modelo treinado, apresentando a diferença não só em métricas como precision, f1-score e AUC-ROC, mas também em valores financeiros dentro de um cenário real, mostrando quanto a mais de lucro podemos gerar para a empresa. Utilizei algumas técnicas como Análise Exploratória Bivariada para gerar algumas hipóteses e entender as relações das variáveis com o fato de ser ou não fraude, também utilizei o MLFlow na parte de ajuste dos hiperparâmetros para ter o melhor controle possível sobre todos os testes que fiz, e por fim ainda apliquei o método SHAP para conseguir visualizar e explicar quais variáveis tiveram maior peso na decisão final do modelo.

## 💼 Entendendo um pouco mais sobre fraudes em transações financeiras

### Importância da Identificação de Fraudes
A identificação de transações fraudulentas é uma preocupação central para instituições financeiras em todo o mundo. Fraudes financeiras podem resultar em perdas significativas, afetando a lucratividade e a confiança dos clientes na instituição. A fraude ocorre quando transações são realizadas de forma ilegal ou sem a autorização adequada, e identificar esses casos é crucial para proteger tanto os ativos financeiros da empresa quanto os de seus clientes.

### Impactos Financeiros
As transações fraudulentas têm um impacto financeiro direto e devastador. Para este projeto específico, a instituição financeira ganha 10% do valor de cada pagamento aprovado corretamente. No entanto, cada fraude aprovada resulta na perda de 100% do valor do pagamento. Isso significa que, para cada transação fraudulenta, a empresa não apenas deixa de ganhar a comissão de 10%, mas também perde todo o valor transacionado. Essa dinâmica ressalta a importância de uma identificação eficaz das fraudes, pois mesmo um pequeno número de fraudes pode rapidamente anular os lucros obtidos por muitas transações legítimas.

### Exemplificação dos Impactos
Para ilustrar, considere o seguinte cenário: se uma empresa aprova 100 transações legítimas de 1.000 reais cada, ela ganha 10.000 reais (10% de 1.000 * 100). No entanto, se uma única transação fraudulenta de 1.000 reais for aprovada, a perda é desse mesmo valor, anulando o ganho de 10 transações legítimas. Portanto, a identificação e prevenção de fraudes são fundamentais para manter a saúde financeira da instituição.

### Principais KPIs do Projeto

#### KPIs de Eficiência e Eficácia do Modelo

*Taxa de Detecção de Fraudes (Precision)*
- Descrição: Percentual de transações identificadas como fraudulentas que realmente são fraudes.
- Fórmula: (Número de Fraudes Corretamente Identificadas / Número Total de Transações Identificadas como Fraudes) * 100%

*Taxa de Cobertura de Fraudes (Recall)*
- Descrição: Percentual de transações fraudulentas identificadas corretamente pelo sistema em relação ao total de fraudes.
- Fórmula: (Número de Fraudes Corretamente Identificadas / Número Total de Fraudes) * 100

*Acurácia do Modelo*
- Descrição: Percentual de todas as transações (fraudulentas e não fraudulentas) que foram classificadas corretamente.
- Fórmula: (Número de Transações Corretamente Classificadas / Número Total de Transações) * 100

*Taxa de Falsos Positivos*
- Descrição: Percentual de transações não fraudulentas que foram incorretamente classificadas como fraudes.
- Fórmula: (Número de Falsos Positivos / Número Total de Transações Não Fraudulentas) * 100

*Taxa de Falsos Negativos*
- Descrição: Percentual de transações fraudulentas que não foram detectadas pelo sistema.
- Fórmula: (Número de Falsos Negativos / Número Total de Transações Fraudulentas) * 100


#### KPIs Financeiros

*Valor Total de Fraudes Prevenidas*
- Descrição: Quantidade de dinheiro que foi salva ao identificar e bloquear transações fraudulentas.
- Fórmula: Soma do valor de todas as transações fraudulentas bloqueadas

*Custo de Fraudes Não Detectadas*
- Descrição: Quantidade de dinheiro perdida devido a fraudes que não foram identificadas e bloqueadas.
- Fórmula: Soma do valor de todas as transações fraudulentas que passaram pelo sistema

*Taxa de Fraude (Fraud Rate)*
- Descrição: Percentual de transações que são fraudulentas em relação ao total de transações processadas.
- Fórmula: (Número Total de Transações Fraudulentas / Número Total de Transações) * 100

## 📊 Análise do Modelo Inicial

O modelo inicial (`score_fraude_modelo`) definitivamente tem diversos pontos a serem melhorados, e podemos analisar isso por uma métrica super importante, a Curva ROC, como é mostrada na imagem abaixo. 

1. Interpretação da AUC (área sob a curva):

- Uma AUC de 0.73 indica que o modelo tem uma capacidade moderada de discriminar entre fraudes e não fraudes. A AUC varia de 0 a 1, onde 1 representa um modelo perfeito e 0.5 representa um modelo sem poder discriminativo (equivalente a uma classificação aleatória).
- Com 0.73, o modelo atual é melhor do que uma classificação aleatória, mas ainda há espaço significativo para melhorias.

2. Curva ROC:

- A curva ROC em si mostra a relação entre a taxa de verdadeiros positivos (sensibilidade) e a taxa de falsos positivos (1 - especificidade) em diferentes limiares de classificação.
- A curva está acima da linha diagonal (linha de não discriminação), o que é bom, mas a inclinação poderia ser mais acentuada para indicar uma melhor performance.

*Pontos a Considerar:*
- Melhoria do Modelo: Com uma AUC de 0.73, o modelo está acertando mais do que errando, mas é crucial entender onde ele está falhando. Analisar casos específicos de falsos positivos e falsos negativos pode ajudar a refinar o modelo.
- Balanceamento dos Dados: Precisamos certificar de que os dados de fraude e não fraude estão balanceados ou precisaremos usar técnicas de balanceamento para evitar viés no treinamento.
- Ajuste de Hiperparâmetros: Ajustar os hiperparâmetros do modelo pode levar a uma melhor performance.
- Features Adicionais: Certamente partiremos para a engenharia de novas features ou a transformação de features existentes para melhorar a performance do modelo.

![image](https://github.com/Henrique-Peter/fraud_detection/blob/main/images/roc_auc.png)

## 🛠 Pré-processamento 
O pré-processamento de dados é crucial em projetos de Machine Learning. Utilizei o Pipeline do Scikit-learn para garantir eficácia e reprodutibilidade. 

*Considerações:*
- Todos os valores da coluna `valor_compra` estão em dólares;
- Não teremos a possibilidade de data leakage, ou seja, todos os dados que estão no dataframe inicial do projeto sempre serão recebidos e calculados antes do evento de "Fraude" ocorrer.
   
*O que será feito no pré-processamento:*
- Excluir a coluna `score_fraude_modelo`, visto que ela servia como modelo baseline, e agora não será mais usada;
- Excluir a coluna `data_compra` para não adicionar a complexidade temporal citada durante a parte de análise dos dados;
- Excluir a coluna `produto`, por ter uma alta cardinalidade (são mais de 127 mil valores únicos, dentro de 150 mil registros);
- Excluir a coluna `score_8` pois isso ajudará a diminuir o ruído (coluna com valores com quase nenhuma distinção entre fraude e não fraude, não agregando muito para o treinamento do modelo);
- Separar a coluna `categoria_produto` em menos valores únicos, mantendo as 1000 categorias (aproximadamente 11% das categorias) que representam 85% das fraudes, e agrupando o restante das categorias como "Outros";
- Realizar algo parecido com a coluna `pais`, mantendo apenas Brasil e Argentina (BR e AR) por representarem mais de 97% dos registros, e agrupar o restante dos países como "Outros";
- Preencher os valores nulos de score com a mediana, por não seguirem uma distribuição normal;
- Preencher os valores nulos de `entrega_doc_2` com 0, visto que no início do projeto a informação era que valores nulos seriam a mesma coisa que "não entregou" (além disso preciso mudar os valores "Y" e "N" das colunas `entrega_doc_2` e `entrega_doc_3` para 0 e 1, assim o modelo conseguirá interpretar corretamente o que é entrega ou não);
- Criar uma feature `was_null` que indicará os registros de `entrega_doc_2` que eram nulos antes da transformação;
- Fazer target encoder na coluna `categoria_produto` devido à alta cardinalidade (mesmo que eu diminua para 1001 valores únicos, são muitos valores). Além disso, usar técnica de cross validation junto com o encoder, pois como temos muitas categorias, não queremos mais de uma com o mesmo valor (empatadas);
- Fazer one hot encoder nas demais variáveis categóricas.

## 🤖 Modelagem e Avaliação

Primeiramente eu treinei os seguintes modelos: Balanced RF, Light GBM, XGBoost e Decision Tree. Pelos resultados, optei por seguir com o Light GBM, e então usei o RandomizedSearchCV para otimizar os hiperparâmetros. As métricas de avaliação incluem: log-loss, precision, recall, f1-score e ROC-AUC .

## 📈 Insights e Conclusões

Passando por todo o projeto, podemos resumir os resultados em 4 principais pontos, comparando o modelo baseline com o novo modelo treinado:

1. Métricas financeiras (considerando dados de teste)
- Threshold: passou de 73 para 57
- Ganhos: passou de USD 80330 para USD 86128
- Perdas: passou de USD 25353 para USD 18070
- Lucro: passou de USD 54977 para USD 68058

2. Matriz de confusão e taxas de fraude e aprovação (considerando dados de teste)
- Falsos negativos: passou de 503 para 383 (diminuir esse quadrante da matriz é especialmente importante, pois é aquele que causa prejuízo dobrado para a instituição, perdendo a comissão de 10% e ainda tendo que pagar 100% do valor para o cliente)
- Taxa de fraude permaneceu a mesma (2%), enquanto a de aprovação passou de 74% para 77%

3. Métricas de desempenho
- Log Loss: passou de 8.6 para 7.3 (diminuir essa métrica indica que o modelo está fazendo previsões melhores)
- Precisão: passou de 0.13 para 0.17 (indica que o novo modelo conseguiu classificar mais fraudes que realmente eram fraudes)
- Recall: passou de 0.67 para 0.75 (indica que o novo modelo conseguiu identificar melhor as fraudes que realmente eram fraudes)
- F1-score: passou de 0.22 para 0.27 (como o F1-score é a média harmônica da precisão e do recall, tendo aumentado ambos, essa métrica também cresceu)
- ROC-AUC: passou de 0.73 para 0.85 (o novo modelo está conseguindo generalizar melhor, entendendo as diferenças entre fraudes e não fraudes)

4. O que realmente importa no fim do dia... qual o lucro a mais que foi gerado?
- Como calculado acima, considerando as médias de compras por mês, e a média dos valores das compras, o modelo inicial, por ter uma razão de lucro de 68%, conseguiria um lucro mensal de aproximadamente USD 221968. Enquanto o modelo novo, com uma razão de lucro de 79%, conseguiria um lucro mensal de aproximadamente USD 257874
- Por mês, isso representa uma diferença de USD 35906, e por ano de USD 430872 a mais que a instituição irá ganhar de lucro. Isso representa um aumento de lucros anuais em 16%.

Ou seja, o novo modelo treinado não só melhora a capacidade de detecção de compras fraudulentas, como também aumenta consideravelmente os lucros da empresa.

## 🚧 Próximos Passos

Os próximos passos vão para um pouco além do projeto em si, e aqui eu destaco dois principais pontos:

*Como você pode garantir que o desempenho do modelo no laboratório vai ser um proxy para o desempenho do modelo em produção?*

Garantir que o desempenho do modelo no laboratório seja um proxy para o desempenho do modelo em produção envolve várias práticas e considerações durante o desenvolvimento e a validação do modelo. Alguns exemplos práticos não citados no projeto são:

Dados Representativos:
- Assegurando que os dados de treinamento e teste sejam representativos do ambiente de produção. Isso inclui a mesma distribuição de fraudes e não fraudes, bem como características e padrões de comportamento dos usuários.

Simulação do Ambiente de Produção:
- Criando um ambiente de teste que simule o ambiente de produção, incluindo latências, volumes de dados, e padrões de acesso.

Monitoramento Contínuo:
- Implementando um sistema de monitoramento para verificar o desempenho do modelo em produção. Monitorar métricas de desempenho, como precisão, recall, F1-score e taxas de falsos positivos/negativos.
- Monitorar também dados de entrada para detectar mudanças na distribuição dos dados que podem afetar o desempenho do modelo.

Análise de Erros:
- Realizar uma análise detalhada dos casos de erro do modelo (tanto falsos positivos quanto falsos negativos) para entender suas limitações e melhorar a robustez do modelo.

Atualização Contínua do Modelo:
- Atualizar o modelo regularmente com novos dados e retrainá-lo conforme as mudanças nos padrões de comportamento e nas características das fraudes.

Teste de Estresse:
- Submetendo o modelo a testes de estresse para garantir que ele pode lidar com picos de carga e volumes de dados maiores sem degradação significativa do desempenho.


*O segundo ponto seria: Se o modelo precisar responder online, no menor tempo possível, o que isso mudaria no desenvolvimento?*

Infraestrutura de Baixa Latência:
- Servidores de Alto Desempenho: Utilizando servidores com CPUs de alta performance e baixa latência. LightGBM é altamente eficiente em CPUs e pode não precisar de GPUs.
- Infraestrutura Otimizada: Considerar o uso de serviços de nuvem otimizados para machine learning, como Amazon SageMaker, Google AI Platform, ou Azure Machine Learning, que oferecem instâncias otimizadas para baixa latência.

Deploy do Modelo:
- Serviços de Inferência: Implementando o LightGBM como um serviço de inferência, utilizando frameworks que suportam baixa latência como Flask, FastAPI, ou microserviços em Kubernetes.
- Serialização do Modelo: Salvar o modelo treinado em um formato eficiente como .txt ou .bin e carregar na memória durante a inferência para reduzir o tempo de carregamento.

Monitoramento e Logging:
- Tempo de Resposta: Monitorar continuamente o tempo de resposta para identificar possíveis gargalos e otimizar o desempenho.
- Logging Otimizado: Implementar logging eficiente para rastrear e depurar a performance sem adicionar sobrecarga significativa.
