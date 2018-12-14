# Vaga para Estágio - Data Scientist - Teste Classificação

Minha solução para o problema proposto no Processo Seletivo da Birdie, Estágio Data Scientist, 2018

## Task
Usando as técnicas que lhe forem mais familiares e amigáveis (linguagem, frameworks, algoritmos), faça um classificador de títulos de ofertas de e-commerces para diferenciar entre smartphone / não-smartphone para o conjunto de títulos que pode ser baixado no nesse [link](https://github.com/bzamith/processo_seletivo-birdie/blob/master/data_estag_ds.tsv)

A partir do classificador, gere um arquivo de resposta onde para cada título é indicado a categoria (smartphone / não-smartphone) predita pelo classificador.

Publique o código para geração do classificador e a lista de categorias preditas em um repositório github/bitbucket/gitlab e nos envie o link.


## Suposições
* Aprendizado não supervisionado
* NLP (Natural Language Processing)
* Apenas duas classes: "smartphone" e "não smartphone"
* Todos os celulares são smartphones
* Tablets não são smartphones

## Implementação
* Python 3 (_não compatível com versões anteriores_)
* KMeans (clustering) com k = 2 (_também é permitida a decisão pelo melhor k de maneira automatizada_)
* Pré-processamento dos dados (_também é permitida a execução sem o pré-processamento_)
* Atributos criados com base no bag of words 

## Execução
No diretório onde se encontra o arquivo [data_estag_ds.tsv](https://github.com/bzamith/processo_seletivo-birdie/blob/master/data_estag_ds.tsv), executar:

> python3 classifier.py

Output: [post_classifier.tsv](https://github.com/bzamith/processo_seletivo-birdie/blob/master/post_indentifier.tsv)

## Lógica
Estou trabalhando com um dataset não rotulado, sendo assim, existem algumas opções:
- [ ] Criar regras de classificação com base em conhecimento próprio (ocorrência de palavras, por exemplo)
- [ ] Rotular manualmente as instâncias óbvias e usar Aprendizado semi-supervisionado
- [x] Aprendizado não supervisionado

Em minha solução, podemos dizer que o Aprendizado de Máquina propriamente dito ocorre de maneira não supervisionada, com o clustering por KMeans.

No entanto, há um pré-processamento dos dados baseado em ocorrência de palavras. Isso evita que sejam criados muitos clusters com instâncias que obviamente não são smartphones. Por exemplo, instâncias que possuem a palavra "tablet" são automaticamente classificadas como "não smartphones". Isso pode gerar inconsistências, mas os resultados mostraram-se melhores do que quando não aplicado o pré-processamento.

Depois de feito esse pré-processamento, as instâncias que já foram classificadas são removidas do dataset que passará pelo KMeans. Agora é preciso criar atributos para o classificador. Optei por usar o [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), no modelo Bag of Words. 

Por fim, é preciso definir o "k" (número de clusters). Podemos fazer isso de duas maneiras:
- [x] k = 2, assumindo que um cluster será de smartphones e outro de não smartphones
- [ ] Melhor k com base no melhor [score](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html). Para isso, o dataset tem que ser dividido em treino e teste. Serão gerados k diferentes clusters e apenas um será de smartphones, todos os outros serão de não smartphones.

As duas opções foram implementadas, mas a primeira obteve melhores resultados que a segunda e por isso fiz uso dela.

A saída de predição do KMeans se dá em classes "0" ou "1" etc com base no cluster em que as instâncias caíram. Para decidir qual classe é a de smartphones, peguei uma instância que eu sabia que tratava-se de um smartphone, devido justamente à ocorrência da palavra "smartphone". Se essa instância conhecida caiu no cluster 1, por exemplo, então esse é o cluster da classe "smartphone". Isso também pode gerar insconsistências. Se essa instãncia conhecida for rotulada erroneamente pelo classificador, então o resultado estará todo invertido. Isso só poderia ser evitado se estivéssemos lidando com Aprendizado Semi-supervisionado.

Os resultados gerados pelo classificador são combinados com os resultados pré-processados e extraídos para um arquivo tsv.

### Nota

Do meu entendimento, por tratar-se de um algoritmo de clustering, o nome "classificador" talvez seja uma maneira não ideal de referir-se à solução. Todavia, como o resultado é tratado como classes ("smartphone" e "não smartphone"), assumi que não haveria problema em tratar a solução como um "classificador". 
