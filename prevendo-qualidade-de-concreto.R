# SEJA BEM VINDO(A)!

# Neste projeto, o objetivo é treinar REDES NEURAIS ARTIFICIAIS, IA, APRENDIZAGEM SUPERVISIONADA E Ñ SUPERVISIONADA.

# =========================== PROBLEMA DE NEGÓCIO =========================== 

# Trabalhamos em uma grande empresa da área de construção civial, que terá que construir uma 
# barragem que fornecerá energia 10% de toda energia hidrelétrica do país

# O DESAFIO: é encontrarmos as melhores combinações de concreto para ser utilizado na construção
# Então faremos a modelagem da resistência do concreto.

# O cálculo da resistência se dá por: água, cimento, areia e outros)
# Link do Dataset: https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength

# As variáveis:
# slag - Escória de alto-forno
# ash - cinzas volantes
# superplastic - superplastificante
# courseagg - areia grossa
# fineagg - areia fina
# strength - resistência à compressão do concreto (TARGET)

# ATENÇÃO:  margem de acerto tem que ser acima de 90% pois em caso de má qualidade, desastres
# ambientais podem ocorrer.

# ============================================================== 

# Definindo o workspace
setwd('C:/Users/jonat/OneDrive/__ARQUIVOS DE CURSOS/_____PORTFOLIO/R')
getwd()

# Carregando o dataset
concrete <- read.csv('concrete.csv')

# ============ 1 - ANÁLISE EXPLORATÓRIA # ============

# Visualizando o dataset
View(concrete)

# Descrição dos dados
str(concrete)

# ============ 2 - TRATAMENTO DOS DADOS # ============

# Normalizando a escala dos dados
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

# Agora aplicaremos a normalização em todo o dataset
# Criamos um dataframe(as.data.frame) e aplicamos a função lapply (dataset e função normalize)
concrete_norm <- as.data.frame(lapply(concrete, normalize))

# Verificando como ficou nosso dataframe
summary(concrete_norm)

# Verificando se o range ficou entre 0 e 1
summary(concrete_norm$strength)

# Comparando com dados antes da normalização
summary(concrete$strength)

# ============ 3 - CRIANDO MÁQUINA PREDITIVA ============

# Separando dados de treino e dados de teste
concrete_train <- concrete_norm[1:773, ] # 70%
concrete_test <- concrete_norm[774:1030, ] # 30%

# Treinando o modelo
install.packages('neuralnet') # <== pacote de treinamento de redes neurais
library(neuralnet)

# Usando uma Rede Neural com apenas uma camada oculta de neurônios matemáticos
set.seed(12345) 
concrete_model <- neuralnet(formula = strength ~ cement + slag +
                              ash + water + superplastic + 
                              coarseagg + fineagg + age,
                            data = concrete_train)

print(concrete_model)

# Visualizando os resultados e a porcentagem de erros em "error"
print(concrete_model)

# Gráfico da rede
plot(concrete_model)

# Avaliação da performance
model_results <- compute(concrete_model, concrete_test[1:8])
print(model_result)

# Verificando a acertividade dos resultados
predicted_strength <- model_results$net.result
print(predicted_strength)

# Verificando a correlação entre o valor previsto e o exato
# Lembrando que a estimação mínima desejada é 90% devidos aos problemas decorrentes
cor(predicted_strength, concrete_test$strength)

# Otimização do modelo com o aumento de 2 camadas ocultas em 5 e 4 neurônios
set.seed(12345)
concrete_model2 <- neuralnet(strength ~ cement + slag +
                               ash + water + superplastic + 
                               coarseagg + fineagg + age,
                             data = concrete_train, hidden =  c(5,4))
# hidden = c(x, x) => se refere ao número de camadas ocultas(2) e quantos neurônios cada uma tem(x, x)

# Gráfico da rede
plot(concrete_model2)

# Analisando o resultado
model_result2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_result2$net.result

# Verificando a correlação entre o valor previsto e o exato
cor(predicted_strength2, concrete_test$strength)
model_result2

# Créditos Eduardo Rocha

