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

#### **3.1 Ajuda técnica automática (especialista)**
Se o aluno estiver bloqueado ou pedir ajuda técnica, o sistema deteta isso
automaticamente e envia a conversa para um modo **especialista**.

Nesse modo, o tutor:
- Dá uma explicação mais técnica e direta
- Mostra um pequeno exemplo em Python
- Dá uma opinião profissional (trade‑offs e quando escolher cada opção)
- Mantém o foco no problema atual (ex: P01, P02, etc.)

Depois disso, o fluxo volta ao normal sem o aluno ter de clicar em botões.

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

#### **6. Estilo de resposta (rápido ou detalhado)**
O aluno pode escolher a forma de resposta:
- **Rápida (default)** → respostas mais curtas e diretas
- **Detalhada** → respostas mais completas e explicativas

O backend guarda essa preferência na sessão e ajusta o tutor e o especialista de acordo.

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

**Ajuda técnica automática:**
- O aluno escreve a dúvida normalmente no chat
- O sistema deteta a intenção de ajuda técnica
- A resposta vem no mesmo idioma da pergunta
- Trechos de código aparecem formatados com botão “Copy”

#### **3. Reupload (nova versão do CSV)**
- O aluno clica em “Re-evaluate CSV”
- Abre um modal com instruções simples
- O aluno faz as correções no Jupyter e envia o novo CSV

#### **4. Estilo de resposta (Rápida vs Detalhada)**
Na página principal existe um seletor com dois níveis:
- **Fast (default)** → respostas curtas e objetivas
- **Detailed** → respostas mais completas

O aluno pode trocar a qualquer momento.

#### **5. Notas pessoais (bloco de notas)**
Existe um botão de **Notes** que abre um painel de anotações:
- O aluno pode criar várias notas
- Cada nota tem título
- As notas são **guardadas localmente** no navegador
- Dá para exportar uma nota como `.txt`

Isso ajuda o aluno a guardar explicações ou resumo das aulas.

#### **6. Recomeçar do zero**
Existe um botão “Start new session”.
Quando o aluno clica, aparece um **modal de confirmação**:
- “Tem a certeza? Os dados da sessão serão perdidos.”
Se confirmar, o frontend limpa o chat e cria uma nova sessão no backend.

#### **6. Recomeçar do zero**
- Existe um botão “Start new session”
- O chat e o estado atual são limpos
- O aluno começa novamente com um novo upload

---

### Resumo do frontend em 1 frase
**O frontend permite enviar CSVs, conversar com o tutor, alternar o nível de detalhe das respostas, guardar notas pessoais e recomeçar a sessão com segurança.**

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
