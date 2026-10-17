# Tech Challenge - Fase 3

Protótipo acadêmico de assistente hospitalar. Dados e protocolos são fictícios: não são recomendações médicas. Nenhum dado real do hospital foi fornecido.

## Executar no Colab

1. Abra `llm_work.ipynb` no Google Colab (Arquivo > Fazer upload de notebook).
2. Selecione GPU T4 em Ambiente de execução > Alterar tipo de ambiente.
3. Execute as células em ordem. O notebook recria os módulos dentro de `/content/tech_challenge_fase3`, instala dependências, prepara dados, treina e demonstra o fluxo.
4. Salve uma cópia no Drive e baixe o ZIP final. O disco temporário do Colab desaparece ao encerrar a sessão.

Sem GPU, execute apenas preparação e testes com `python -m src.data` e `python -m unittest discover -s tests -v`. O treinamento baixa o modelo público Qwen/Qwen2.5-0.5B-Instruct do Hugging Face, sem chave de API. Verifique licença e model card antes de reutilizar.

## Organização

- `src/data.py`: curadoria, anonimização demonstrativa, partição por protocolo e prontuários SQLite.
- `src/train.py`: fine-tuning QLoRA e avaliação antes/depois em protocolo separado.
- `src/assistant.py`: pipeline LangChain, consultas parametrizadas, fluxo LangGraph e auditoria.
- `data/`: 60 exemplos sintéticos; 40 treino, 10 validação, 10 teste.
- `docs/relatorio_tecnico.md`: metodologia, diagrama, limites e campos para resultados reais.
- `outputs/`: adaptador, métricas, logs e versões do ambiente após execução.

O notebook contém cópia dos módulos para execução autônoma. Se alterar os módulos, regenere o notebook com `python build_notebook.py` antes de entregar. Dependências de treinamento são fixadas; LangChain/LangGraph têm faixas de versão e o ambiente efetivo é registrado com pip freeze. A compatibilidade GPU ainda precisa ser validada no Colab.

## Segurança e avaliação

As regras de anonimização não cobrem nomes livres, datas, endereços ou todos os identificadores. Use apenas os exemplos sintéticos deste projeto. SQL não é gerado pela LLM. Não existem ferramentas que prescrevam, enviem alertas externos ou executem procedimentos. O fluxo identifica alertas registrados e cria uma resposta para revisão, sem enviar mensagens.

A LLM produz um rascunho em `raw` para inspeção acadêmica; o gate final exibe apenas fatos estruturados e regras fictícias. Isso limita a utilidade do assistente e não constitui validação médica. Logs registram ID sintético, hash da pergunta, fontes, modelo, resposta e status; não gravam o rascunho livre. Qualquer evolução para dados reais exige outro desenho de privacidade e validação.

Antes da entrega: execute o treino, examine métricas e respostas, preencha o relatório com resultados observados, adicione nomes do grupo, publique o repositório escolhido e grave vídeo de até 15 minutos. Não há resultados de treino pré-fabricados.
