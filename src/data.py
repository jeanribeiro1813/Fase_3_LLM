import json
import random
import re
import sqlite3
from contextlib import closing
from pathlib import Path

PROTOCOLS = [
    {'id': 'SYN-001', 'topic': 'exames', 'text': 'Protocolo fictício: consultar exames pendentes e encaminhar a lista ao médico responsável; não solicitar nem cancelar exames automaticamente.'},
    {'id': 'SYN-002', 'topic': 'receita', 'text': 'Modelo fictício de receita: campos medicamento, dose, via e duração permanecem em branco até preenchimento e assinatura do médico. O assistente não prescreve.'},
    {'id': 'SYN-003', 'topic': 'laudo', 'text': 'Modelo fictício de laudo: identificação pseudonimizada, exame, achados fornecidos, conclusão pendente e assinatura médica. Não inventar achados.'},
    {'id': 'SYN-004', 'topic': 'procedimento', 'text': 'Procedimento fictício: verificar identificação pseudonimizada, documentação e aprovação humana antes de qualquer execução.'},
    {'id': 'SYN-005', 'topic': 'alerta', 'text': 'Fluxo fictício: um alerta já registrado no prontuário deve ser encaminhado ao profissional responsável, sem diagnóstico ou tratamento automático.'},
    {'id': 'SYN-006', 'topic': 'tratamento', 'text': 'Conduta fictícia: informar que o tratamento depende da avaliação do médico e do protocolo institucional aprovado; não indicar medicamentos nem doses.'},
]

def anonymize(text):
    """Demonstração por regras; NÃO garante anonimização de texto clínico real."""
    text = re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[EMAIL]', text)
    text = re.sub(r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b', '[CPF]', text)
    text = re.sub(r'(?i)nome\s*:\s*[^;\n]+', 'Nome: [REMOVIDO]', text)
    return ' '.join(text.split())

def prepare(root='.'):
    root = Path(root)
    (root / 'data').mkdir(exist_ok=True)
    rows = []
    for p in PROTOCOLS:
        for verb in ['Explique', 'Resuma', 'Como funciona', 'Qual é a regra de', 'Descreva', 'Oriente sobre', 'Revise', 'Detalhe', 'Apresente', 'O que consta sobre']:
            rows.append({'group': p['id'], 'prompt': anonymize(f"{verb} {p['topic']} no hospital fictício?"),
                         'completion': f"{p['text']} Fonte: {p['id']}. Revisão humana obrigatória."})
    # Separar por protocolo evita paráfrases do mesmo exemplo nos dois conjuntos.
    random.Random(42).shuffle(rows)
    for name, groups in [('train', {'SYN-001', 'SYN-002', 'SYN-003', 'SYN-004'}), ('validation', {'SYN-005'}), ('test', {'SYN-006'})]:
        selected = [r for r in rows if r['group'] in groups]
        assert len({r['prompt'] for r in selected}) == len(selected)
        (root / 'data' / f'{name}.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in selected), encoding='utf-8')
    (root / 'data/protocols.json').write_text(json.dumps(PROTOCOLS, ensure_ascii=False, indent=2), encoding='utf-8')
    with closing(sqlite3.connect(root / 'data/hospital.db')) as db:
        db.execute('CREATE TABLE IF NOT EXISTS patients (id TEXT PRIMARY KEY, pending TEXT, alert INTEGER, updated TEXT)')
        db.executemany('INSERT OR REPLACE INTO patients VALUES (?,?,?,?)', [
            ('SYN-P001', 'Exame fictício A; Exame fictício B', 1, '2026-10-16T12:00:00Z'),
            ('SYN-P002', '', 0, '2026-10-16T12:00:00Z')])
        db.commit()
    return {name: len((root / 'data' / f'{name}.jsonl').read_text(encoding='utf-8').splitlines()) for name in ['train', 'validation', 'test']}

if __name__ == '__main__':
    print(prepare())
