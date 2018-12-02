
#1o filtrar ate conseguir olhar para uma imagem e perceber
# = Pre-processamento
#- CLAHE
#- bilateral filtering 
#- histogram matching
#
#2o pixel based features 
#- lbp
#- intensidade regional e gradientes 
#- Matriz hessiana
#
#n o fim feature selection algorithms vao pegar nas features e dizer o que e 
#m erda e o que e fixe
#
#
# X para cada pixel extrai as features 
#
# funcao feature 1  - sai uma imagem
# funcao feature 2 - sai outra imagem
#
# feature 1 - extrair os pixeis amostra
# feature 2 - extrair os pixeis amostra

# e preciso escalar as features com StandardScaler().fit_transform(X)
# Sendo que X e a matriz das features 

# Cross-validation  - para verificar se os classificadores são robustos
# Procurar na net como fazer cross-validation para rodar os validation set