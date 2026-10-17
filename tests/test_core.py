import tempfile
import unittest
from pathlib import Path
from src.data import anonymize, prepare
from src.assistant import patient_record

class CoreTests(unittest.TestCase):
    def test_anonymization(self):
        self.assertEqual(anonymize('Nome: Pessoa Teste; CPF 123.456.789-00 email x@y.com'), 'Nome: [REMOVIDO]; CPF [CPF] email [EMAIL]')

    def test_split(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(prepare(d), {'train':40, 'validation':10, 'test':10})
            groups = [{json.loads(r)['group'] for r in (Path(d)/'data'/f'{n}.jsonl').read_text(encoding='utf-8').splitlines()} for n in ['train','validation','test']]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_sql_rejects_invalid_identifier(self):
        with self.assertRaises(ValueError):
            patient_record("' OR 1=1 --")

if __name__ == '__main__':
    unittest.main()
