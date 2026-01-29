# Academic Lab v2 — Guia para Leigos (Backend e Frontend)

Este documento explica, **sem código**, como o sistema funciona. A ideia é que qualquer pessoa, mesmo sem experiência técnica, consiga compreender o fluxo do programa.

---

## 1) Como funciona o BACKEND (de forma simples)

### Visão geral
O backend é o “cérebro” do sistema. É ele que:
- Recebe o ficheiro CSV
- Analisa os dados
- Detecta problemas (P01–P35)
- Decide o que mostrar ao aluno
- Guarda o progresso da sessão

Pense no backend como um tutor automático que lê o CSV, identifica erros e guia o aluno.

---

### Passo a passo do fluxo no backend

#### **1. Upload do CSV**
Quando o aluno envia o CSV:
1. O backend lê o ficheiro.
2. Calcula estatísticas básicas:
   - Número de linhas e colunas
   - Percentagem de valores em falta
   - Outliers (valores extremos)
   - Duplicados
   - Correlações altas
3. Cria uma **sessão** (com ID único) para guardar tudo o que acontece com aquele aluno.

#### **2. Diagnóstico inicial (P01–P35)**
Com as estatísticas em mãos, o backend identifica problemas de qualidade e prepara uma lista, por exemplo:
- P01: valores em falta
- P03: outliers
- P09: classes desbalanceadas

Estes problemas são organizados por gravidade (CRITICAL, HIGH, MEDIUM, LOW).

#### **3. O tutor responde no chat**
Depois do diagnóstico, o backend envia a primeira resposta ao aluno:
- Lista dos problemas encontrados
- Sugestão de por onde começar

A partir daí, o aluno conversa com o tutor (chat). O backend interpreta cada mensagem e decide a resposta seguinte.

#### **3.1 Ajuda técnica (especialista) quando o aluno está bloqueado**
Se o aluno pedir ajuda técnica, o backend ativa um modo **especialista**.

Nesse modo, o tutor:
- Dá uma explicação mais técnica e direta
- Mostra um pequeno exemplo em Python
- Dá uma opinião profissional (trade‑offs e quando escolher cada opção)
- Mantém o foco no problema atual (ex: P03, P01, etc.)

Depois disso, o fluxo volta ao normal.

#### **4. Progresso da sessão**
O backend guarda o “estado” da sessão:
- Quais problemas já foram resolvidos
- Qual problema está ativo
- Se o aluno compreendeu a explicação
- Se é necessário re-upload do CSV

Ou seja: o backend **lembra-se** do progresso do aluno.

#### **5. Reupload do CSV**
Quando o aluno envia um CSV atualizado:
1. O backend repete a análise do novo ficheiro.
2. Compara a lista de problemas antes e depois.
3. Diz claramente o que foi resolvido e o que ainda falta.

#### **6. Outliers: 3 soluções integradas**
Para o problema P03 (outliers), o aluno tem três maneiras de lidar:

1. **Dismiss (false alarm)**
   - O aluno diz: “estes outliers são normais no meu domínio”.
   - O backend marca como “resolvido por decisão”.

2. **Contexto de domínio**
   - O aluno dá contexto do que é normal (ex: valores esperados entre 10 e 1000).
   - O backend recalcula os outliers usando esse contexto.
   - Se o contexto for estranho (ex: intervalo não cobre os dados), o backend **avisa**.

3. **Threshold (sensibilidade)**
   - O aluno ajusta quão sensível é o detector.
   - Sensibilidade alta = muitos outliers
   - Sensibilidade baixa = menos outliers

**Regra principal:** se existir contexto (min/max), ele tem prioridade sobre a sensibilidade.

#### **7. Avisos educacionais**
Se o aluno fornece contexto que não faz sentido com os dados:
- O backend não bloqueia (é educativo)
- Apenas avisa com uma mensagem clara

Exemplo:
“⚠ O intervalo [150, 200] não cobre nenhum valor real do dataset.”

---

### Resumo do backend em 1 frase
**O backend recebe o CSV, identifica problemas, guia o aluno, reavalia versões novas e adapta a análise com contexto e sensibilidade.**

---

## 2) Como funciona o FRONTEND (de forma simples)

### Visão geral
O frontend é a “cara” do sistema. É aquilo que o aluno vê e usa:
- Página de upload
- Chat com o tutor
- Botões de ação
- Modais para reupload e contexto

Pense no frontend como a sala de aula, onde o aluno interage com o tutor.

---

### Passo a passo do fluxo no frontend

#### **1. Upload inicial**
- O aluno escolhe um ficheiro CSV
- O frontend envia o ficheiro para o backend
- O chat começa automaticamente

#### **2. Conversa no chat**
- O aluno escreve perguntas
- O tutor responde com explicações
- As respostas aparecem como “bolhas” no chat

**Ajuda técnica com modal:**
- O aluno pode clicar em “Ask for technical help”
- Abre um modal para escrever a dúvida
- A resposta vem no mesmo idioma da pergunta
- Trechos de código aparecem formatados com botão “Copy”

#### **3. Reupload (nova versão do CSV)**
- O aluno clica em “Re-evaluate CSV”
- Abre um modal com instruções simples
- O aluno faz as correções no Jupyter e envia o novo CSV

#### **4. Controlo de outliers (P03)**
Se o problema P03 existir, o frontend mostra um painel especial:

- **Botão “Mark as false alarm”**
  → o aluno explica por que é normal

- **Botão “Provide domain context”**
  → o aluno explica o intervalo normal (min/max)

- **Slider de sensibilidade**
  → ajusta a detecção de outliers

Este painel vem **fechado** por padrão e pode ser aberto/fechado pelo aluno.
Ao abrir, existe um texto curto a explicar para que serve cada opção.

E se houver warnings (ex: intervalo estranho), aparecem logo abaixo:

“⚠ Context warnings: …”

#### **5. Feedback imediato**
Sempre que o aluno:
- Dismiss
- Dá contexto
- Muda sensibilidade

O frontend:
- Atualiza o chat
- Mostra o próximo problema
- Mostra avisos, se necessário

#### **6. Recomeçar do zero**
- Existe um botão “Start new session”
- O chat e o estado atual são limpos
- O aluno começa novamente com um novo upload

---

### Resumo do frontend em 1 frase
**O frontend permite que o aluno envie CSVs, converse com o tutor e tome decisões (dismiss/contexto/sensibilidade) de forma simples e visual.**

---

## Conclusão (para leigos)

O sistema funciona como um tutor inteligente:
1. Lê o CSV
2. Detecta problemas
3. Explica como resolver
4. Reavalia novas versões
5. Ajusta a análise conforme o aluno explica o contexto

Tudo foi desenhado para ser **educacional**, não apenas “corrigir erros”.
O objetivo é que o aluno aprenda o porquê das decisões, não apenas siga comandos.

---

**Ficheiro:** bible.md  
**Objetivo:** servir como “manual leigo” para entender o fluxo completo do sistema.
