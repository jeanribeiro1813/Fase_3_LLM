import json
from pathlib import Path

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
SYSTEM = 'Você é um assistente acadêmico hospitalar. Use somente o contexto fornecido. Não prescreva, não invente fatos. Cite fontes e exija revisão humana.'

def train():
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer
    if not torch.cuda.is_available():
        raise RuntimeError('Ative GPU no Colab: Ambiente de execução > Alterar tipo de ambiente > T4 GPU.')
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map='auto',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16))
    model = prepare_model_for_kbit_training(model)
    ds = load_dataset('json', data_files={k: f'data/{k}.jsonl' for k in ['train', 'validation', 'test']})
    def format_row(r):
        return {'text': tok.apply_chat_template([{'role':'system','content':SYSTEM}, {'role':'user','content':r['prompt']}, {'role':'assistant','content':r['completion']}], tokenize=False)}
    ds = ds.map(format_row)
    args = SFTConfig(output_dir='outputs/checkpoints', max_length=384, num_train_epochs=3,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=2e-4, fp16=True, logging_steps=2, eval_strategy='epoch', save_strategy='epoch',
        report_to='none', seed=42, dataset_text_field='text', completion_only_loss=False)
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds['train'], eval_dataset=ds['validation'],
        processing_class=tok, peft_config=LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM'))
    baseline = trainer.evaluate(ds['test'])
    trainer.train()
    after = trainer.evaluate(ds['test'])
    trainer.save_model('outputs/adapter')
    tok.save_pretrained('outputs/adapter')
    results = {'base_test':baseline, 'finetuned_test':after, 'history':trainer.state.log_history,
               'note':'Teste sintético pequeno, protocolo não visto; loss não demonstra segurança clínica.'}
    Path('outputs/training_metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results

if __name__ == '__main__':
    train()
